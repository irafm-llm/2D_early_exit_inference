import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoModelForCausalLM, Gemma3nTextModel, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, TaskType
from peft import get_peft_model
from contextlib import nullcontext
from functools import partial
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score

from config import logging_dir
from utils import RMSNorm, calculate_gradient_norm

writer = SummaryWriter(logging_dir)


class CustomCollator:
    """Pads a batch of multi-sentence samples to a common sequence length.

    Each sample may contain a different number of tokens, so this collator
    right-pads all sequences to the longest one in the batch. It also builds
    the attention mask and preserves the per-token sentence assignment mask.

    Args:
        tokenizer:        HuggingFace tokenizer (used to look up the pad token id).
        column_name:      Dataset column that holds the token id list.
        mask_column_name: Dataset column that holds the sentence-index mask
                          (0 = padding / BOT token, 1..N = sentence index).
    """

    def __init__(self, tokenizer, column_name: str, mask_column_name: str):
        self.pad_token_idx = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
        self.column_name = column_name
        self.mask_column_name = mask_column_name

    def __call__(self, batch):
        input_ids, attention_mask, sentence_mask, label_idx, n_sentences, sentence_length = [], [], [], [], [], []
        # Determine the longest sequence in the batch for padding
        max_length = max(len(s[self.column_name]) for s in batch)
        for s in batch:
            pad_len = max_length - len(s[self.column_name])
            # Pad token ids with the tokenizer's pad token
            input_ids.append(torch.tensor([s[self.column_name] + [self.pad_token_idx] * pad_len]))
            # Attention mask: 1 for real tokens, 0 for padding
            attention_mask.append(torch.tensor([[1] * len(s[self.mask_column_name]) + [0] * pad_len]))
            # Sentence mask: sentence index per token, 0 for padding positions
            sentence_mask.append(torch.tensor([s[self.mask_column_name] + [0] * pad_len]))
            label_idx.append(s['label'])
            n_sentences.append(s['n_sentences'])
            sentence_length.append(s['sentence_length'])

        input_ids = torch.cat(input_ids)
        attention_mask = torch.cat(attention_mask)
        output = {'input_ids': input_ids, 'attention_mask': attention_mask, 'label_idx': torch.tensor(label_idx),
                  'sentence_mask': sentence_mask, 'n_sentences': torch.tensor(n_sentences),
                  'sentence_length': np.concatenate(sentence_length)}

        return output


class CyclicDataLoader:
    """Wraps a DataLoader to produce batches indefinitely.

    When the underlying iterator is exhausted it restarts from the beginning,
    so training loops driven by a fixed step count rather than epochs can call
    get_batch() without worrying about StopIteration.
    """

    def __init__(self, dl):
        self.dl = dl
        self.iterator = iter(dl)

    def get_batch(self):
        try:
            return next(self.iterator)
        except StopIteration:
            # End of dataset reached — restart the iterator for the next epoch
            self.iterator = iter(self.dl)
            return next(self.iterator)


class Gemma3nRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x) * self.weight.to(x.device)   # .float()
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class AdapterModule(torch.nn.Module):
    """Classification head"""
    def __init__(self, in_features, out_features, device, hidden_features=200, dropout=0.2, norm=None):
        super().__init__()
        self.c_fc    = torch.nn.Linear(in_features, hidden_features, device=device)
        self.act_fn  = torch.nn.SiLU()
        self.c_proj  = torch.nn.Linear(hidden_features, out_features, device=device)
        self.device  = device

        self.dropout = torch.nn.Dropout(dropout)

        if norm == 'LayerNorm':
            self.norm = torch.nn.LayerNorm(in_features, device=device)
        elif norm == 'RMSNorm':
            self.norm = RMSNorm(in_features, device=device)
        elif norm is None:
            self.norm = None
        else:
            raise ValueError(f'unknown norm type, supported values are - LayerNorm, RMSNorm or None, got: {norm}')

    def forward(self, x):
        x = x.to(self.device)

        if self.norm is not None:
            x = self.norm(x)

        x = self.dropout(x)
        x = self.c_fc(x)
        x = self.act_fn(x)
        x = self.c_proj(x)

        return x
   

class GatedCascadeModel(torch.nn.Module):
    def __init__(self, model_ckpt, label2idx, n_labels, adapter=None, peft_training=False):
        super(GatedCascadeModel, self).__init__()

        self.label2idx = label2idx
        self.model_ckpt = model_ckpt
            
        adapter_module = torch.nn.Linear if adapter is None else adapter

        # peft_training=True: backbone is frozen, load in FP16 to save VRAM.
        # peft_training=False: backbone is fully trained; must be FP32 so gradients
        #   are FP32 and GradScaler can unscale them (FP16 grads are unsupported by GradScaler).
        backbone_dtype = torch.float16 if peft_training else torch.float32

        if 'gemma' in model_ckpt.lower():
            model = Gemma3nTextModel.from_pretrained(model_ckpt, torch_dtype=backbone_dtype, device_map="auto")

            hidden_size = model.config.get_text_config().hidden_size
            num_hidden_layers = model.config.get_text_config().num_hidden_layers
            self.config = model.config.get_text_config()
            self.altup_unembed_projections = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
            ).to(model.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_ckpt, torch_dtype=backbone_dtype, device_map="auto")
            hidden_size = model.config.n_embd if hasattr(model.config, 'n_embd') else model.config.hidden_size
            num_hidden_layers = model.config.num_hidden_layers

        self.classification_adapters = torch.nn.ModuleDict(
                                        {f'fc{i+1}': adapter_module(hidden_size, n_labels, device=model.device)
                                     for i in range(num_hidden_layers+1)})

        if peft_training:
            lora_config = LoraConfig(
                r=64, 
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=16,
                lora_dropout=0.05
            )
            model = get_peft_model(model, lora_config)

        self.model = model
        self.device = model.device
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Only needed for Gemma3n AltUp stream merging
        self.norm = Gemma3nRMSNorm(self.hidden_size, eps=1e-06) if 'gemma' in model_ckpt.lower() else None
        print(f'Number of layers: {num_hidden_layers}')

    def forward(self, x, return_embedding=False):
        cascade_output = {}
        _, t = x['input_ids'].shape

        # Run the backbone; collect hidden states from every layer (including embedding layer)
        output = self.model.forward(input_ids=x['input_ids'], attention_mask=None, output_attentions=False, output_hidden_states=True)
        hidden_states = output.hidden_states  # tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_size)

        # sentence_length[i] = cumulative token count up to and including sentence i,
        # used to compute the mean-pool window for each sentence.
        # Shape after unsqueeze: (total_sentences_in_batch, 1)
        sentence_length = torch.tensor(x['sentence_length']).unsqueeze(1)

        # Total number of sentences across all samples in the batch
        n_sentences = torch.sum(x['n_sentences'])

        # Build a boolean positional mask of shape (n_sentences, seq_len, 1).
        # mask[s, p] is True when token position p belongs to sentence s, i.e. p < sentence_length[s].
        sentence_mask = (torch.arange(t).expand(n_sentences, t) < sentence_length).bool().unsqueeze(-1)
        sentence_mask[:, 0] = 0  # exclude beginning-of-text token from mean pooling

        # dtype of the adapter weights — used to cast embeddings before the linear projection
        adapter_weight_dtype = self.classification_adapters['fc1'].c_fc.weight.dtype
        # Small constant used in AltUp magnitude normalisation; created once to avoid
        # repeated allocation inside the per-layer loop
        epsilon_tensor = torch.tensor(1e-5)

        # Iterate over every layer's hidden state and produce a classification logit at each exit point
        for i, hidden_state in enumerate(hidden_states):
            if len(hidden_state.shape) == 4:
                # Gemma3n uses AltUp (Alternating Updates, https://arxiv.org/pdf/2301.13310):
                # the model maintains multiple parallel hidden-state streams.
                # We project each auxiliary stream back into the primary stream's space,
                # magnitude-normalise it to match the primary stream's scale, then average
                # all streams to obtain a single (batch, seq_len, hidden_size) tensor.
                target_magnitude = torch.mean(hidden_state[0] ** 2, dim=-1, keepdim=True) ** 0.5
                temp_hidden_states = [hidden_state[0]]  # stream 0 is the primary stream

                for j in range(1, self.config.altup_num_inputs):
                    # Project auxiliary stream j into the primary stream's space
                    # (mirrors jax.numpy.einsum("btp,pd->btd", hidden_state[j], W))
                    altup_unemb_proj: torch.Tensor = self.altup_unembed_projections[j - 1](hidden_state[j])
                    current_hidden_state = altup_unemb_proj.to(dtype=hidden_state.dtype, device=target_magnitude.device)
                    # Rescale so the projected stream has the same RMS magnitude as stream 0
                    new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
                    new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor.to(target_magnitude.device)))
                    current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
                    temp_hidden_states.append(current_hidden_state)

                # Average all streams and apply RMS norm to get the final hidden state
                hidden_state = torch.stack(temp_hidden_states)
                hidden_state = torch.mean(hidden_state, dim=0)
                hidden_state = self.norm(hidden_state)

            # --- Sentence embedding via masked mean pooling ---
            # Each sample in the batch may contain multiple sentences. We expand the batch
            # dimension so that every sentence gets its own row.
            # repeat_interleave duplicates sample i exactly n_sentences[i] times:
            #   shape: (batch, seq_len, hidden_size) -> (n_sentences, seq_len, hidden_size)
            sentence_embeddings = torch.repeat_interleave(hidden_state, repeats=x['n_sentences'].to(hidden_state.device), dim=0)

            # Zero out token positions that don't belong to this sentence (mask = 0),
            # then sum across the sequence dimension and divide by the sentence length
            # to obtain the mean of only the tokens in this sentence.
            # Result shape: (n_sentences, hidden_size)
            sentence_embeddings *= sentence_mask.to(sentence_embeddings.device)
            sentence_embeddings = sentence_embeddings.sum(dim=1) / sentence_length.to(sentence_embeddings.device)

            adapter_name = f'fc{i+1}'

            # Cast to adapter weight dtype if needed (e.g. FP16 backbone -> FP32 adapter)
            if sentence_embeddings.dtype != adapter_weight_dtype:
                sentence_embeddings = sentence_embeddings.type(adapter_weight_dtype)

            logits = self.classification_adapters[adapter_name].forward(sentence_embeddings)
            cascade_output[f'{adapter_name}_logits'] = logits.detach().cpu()

            if return_embedding:
                cascade_output[f'{adapter_name}_embedding'] = sentence_embeddings.cpu().detach().to(torch.float16).numpy()

            if 'label_idx' in x:
                # Expand document-level labels to sentence level so each sentence is supervised
                labels = torch.repeat_interleave(x['label_idx'].to(logits.device), repeats=x['n_sentences'].to(logits.device), dim=0)
                adapter_loss = F.cross_entropy(logits, labels)
                cascade_output[f'{adapter_name}_loss'] = adapter_loss

        return cascade_output

    
class CascadeDecoderModule:
    def __init__(self, model_ckpt, label2idx, training_config, n_labels=3):
        idx2label = {i: l for l, i in label2idx.items()}
        self.training_config = training_config
        self.idx2label = idx2label
        self.label2idx = label2idx
        self.model_ckpt = model_ckpt
        
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer.padding_side = "right"
        self.begin_of_text_idx = tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)[0]

        adapter = partial(AdapterModule,
                          hidden_features=training_config.adapter_hidden_size,
                          dropout=training_config.adapter_dropout,
                          norm=training_config.adapter_norm)
        
        self.tokenizer = tokenizer
        self.model = GatedCascadeModel(model_ckpt, label2idx, n_labels, adapter=adapter, peft_training=training_config.peft_training)
       
    def encode_sentences(self, sample, max_length):
        input_ids, sentence_mask, sentence_lengths = [self.begin_of_text_idx], [0], []
        cumulative_length = 1  # starts at 1 to account for the beginning-of-text token
        for sentence_id, sentence in enumerate(sample['sentences']):
            # Stop tracking sentences that start at or beyond max_length — they have no tokens
            # in the truncated sequence, so including them would create empty sentence embeddings
            # and inflate n_sentences incorrectly.
            if cumulative_length >= max_length:
                break
            token_ids = self.tokenizer(sentence, add_special_tokens=False)['input_ids']
            input_ids += token_ids
            sentence_mask += [sentence_id + 1] * len(token_ids)  # each token tagged with its sentence index
            cumulative_length += len(token_ids)
            # Clamp to max_length: if this sentence extends past the truncation boundary,
            # the embedding mean should only cover tokens that actually exist in the sequence.
            sentence_lengths.append(min(cumulative_length, max_length))

        return {'sentence_idx': input_ids[:max_length], 'sentence_mask': sentence_mask[:max_length],
                'n_sentences': len(sentence_lengths), 'sentence_length': sentence_lengths}
    
    def weighted_compound_loss(self, output, weight: float = 0.1, last_layer_weight: float = 0.8):
        """Combine individual losses from adapters.

        The final adapter (last transformer layer) is weighted by `last_layer_weight`.
        All earlier exit-point adapters are weighted by the smaller `weight` value.

        Parameters:
            output (dict):            dictionary with per-adapter losses from forward()
            weight (float):           weight applied to intermediate adapter losses
            last_layer_weight (float): weight applied to the final adapter loss
        """
        total_loss = 0
        for k, v in output.items():
            if k == f'fc{self.model.num_hidden_layers+1}_loss':
                total_loss += last_layer_weight * v
            elif 'loss' in k:
                total_loss += weight * v

        return total_loss

    def fit(self, dataset: DatasetDict, path: str): 
        """Run training and store model checkpoint.

        Args:
            dataset (DatasetDict): training/validation dataset
            path    (str):         output directory for the saved checkpoint
        """
        n_epochs = self.training_config.n_epochs
        criterion = self.weighted_compound_loss
       
        gradient_accumulation_steps = self.training_config.gradient_accumulation_steps
        logging_steps = self.training_config.logging_steps
        batch_size = self.training_config.train_batch_size
        validate = True if self.training_config.validation_steps and self.training_config.validation_steps > 0 else False

        idx_column, mask_column = ('sentence_idx', 'sentence_mask') 
        if validate:
            validation_ds = dataset['validation'].select(range(self.training_config.validation_steps)) 

        train_dl = DataLoader(dataset['train'], batch_size=batch_size,
                              collate_fn=CustomCollator(self.tokenizer, idx_column, mask_column))

        train_dl = CyclicDataLoader(train_dl)
        if self.training_config.train_steps:
            total_training_steps = self.training_config.train_steps
        else:
            n_samples = len(dataset['train'])
            total_training_steps = int((n_samples * n_epochs) / (batch_size * gradient_accumulation_steps))

        device = self.model.device   
        ctx = nullcontext() if device == 'cpu' else torch.autocast(device_type='cuda', dtype=torch.float16)
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.training_config.lr)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_training_steps), num_training_steps=total_training_steps)
        # GradScaler requires FP32 gradients. With peft_training=True the backbone is frozen
        # (no FP16 grads) and LoRA weights are FP32 — scaler works. With peft_training=False
        # the backbone is loaded in FP32 (see GatedCascadeModel) so grads are FP32 — scaler works.
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        iter_num = 0
        train_batch = train_dl.get_batch()

        while True:
            t1 = time.time()
            optimizer.zero_grad(set_to_none=True)

            for _ in range(gradient_accumulation_steps):
                train_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in train_batch.items()}

                with ctx:
                    output = self.model.forward(train_batch)
                    loss = criterion(output)
                    loss /= gradient_accumulation_steps

                # backward adapter loss
                scaler.scale(loss).backward()
                train_batch = train_dl.get_batch()

            scaler.unscale_(optimizer)
            grad_norm = calculate_gradient_norm(self.model)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            t2 = time.time()
            dt = t2 - t1            
            # Multiply back by gradient_accumulation_steps to recover the true batch loss
            # (loss was divided inside the accumulation loop to scale gradients correctly)
            print(f'loss: {loss.item() * gradient_accumulation_steps:.6f} time: {dt*1000:.2f}ms, grad_norm: {grad_norm:.6f}')
                
            if iter_num % logging_steps == 0:
                if validate:
                    # run the evaluation on validation slice
                    validation_preds, _ = self.evaluate_dataset(validation_ds, output_dir=path)
                    validation_accuracy = self.calculate_accuracy(validation_preds)
                    print(f'step: {iter_num} / {total_training_steps}, validation accuracy: {validation_accuracy:.4f}')

                log_output = f'step: {iter_num} / {total_training_steps}, train_total_loss: {loss.item() * gradient_accumulation_steps:.4f}'
                writer.add_scalar('train_total_loss', loss.item() * gradient_accumulation_steps, iter_num)
                adapter_loss = 0

                n_adapters = self.model.num_hidden_layers + 1
                for i in range(n_adapters):
                    # sum all adapter losses for logging (fc1 … fc{n_adapters})
                    adapter_loss += output.get(f'fc{i+1}_loss', 0)

                adapter_loss /= n_adapters

                log_output += f', mean_adapter_loss: {adapter_loss:.4f}'

                writer.add_scalar('mean_adapter_loss', adapter_loss, iter_num)
                print(log_output)

            iter_num += 1
            if iter_num >= total_training_steps:
                break

        torch.save(self.model.state_dict(), os.path.join(path, 'model_checkpoint.pt'))

    def evaluate_dataset(self, dataset: Dataset, output_dir: str = '.', save_embeddings: bool = False) -> pd.DataFrame:
        """Iterate over the dataset, collect per-sentence logits and optionally embeddings.

        Args:
            dataset (Dataset):        split to evaluate (test or validation)
            output_dir (str):         directory where embeddings.npy will be saved (only used when save_embeddings=True)
            save_embeddings (bool):   whether to collect and save per-layer sentence embeddings;
                                      disable during mid-training validation to save memory and time
        """
        self.model.eval()
        device = self.model.device
        # Mirror the training autocast context so evaluation works on both CPU and GPU
        ctx = nullcontext() if str(device) == 'cpu' else torch.autocast(device_type='cuda', dtype=torch.float16)
        idx_column, mask_column = ('sentence_idx', 'sentence_mask')
        # +1 because hidden_states includes the embedding layer output, giving num_hidden_layers+1 adapters
        n_adapters = self.model.num_hidden_layers + 1
        test_dl = DataLoader(dataset, batch_size=self.training_config.test_batch_size,
                             collate_fn=CustomCollator(self.tokenizer, idx_column, mask_column))

        preds = []
        embeddings = []
        sample_id = 0
        for _, batch in tqdm(enumerate(test_dl), total=len(test_dl)):
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            with torch.no_grad():
                with ctx:
                    output = self.model.forward(inputs, return_embedding=save_embeddings)

            sentence_embeddings = []
            pred = {}
            for adapter in range(1, n_adapters+1):
                logits = output[f'fc{adapter}_logits']
                pred_classes = np.argmax(logits, axis=-1)

                pred[f'fc{adapter}_logits'] = logits.tolist()
                pred[f'fc{adapter}_pred'] = pred_classes.tolist()
                if save_embeddings:
                    sentence_embeddings.append(output[f'fc{adapter}_embedding'])

            if save_embeddings:
                sentence_embeddings = np.transpose(np.array(sentence_embeddings), (1, 0, 2))
                embeddings.append(sentence_embeddings)

            pred['label'] = torch.repeat_interleave(batch['label_idx'].cpu(), repeats=batch['n_sentences'].cpu(), dim=0).tolist()
            pred['original_idx'] = np.concatenate([[sample_id+i]*n for i, n in enumerate(batch['n_sentences'])])
            pred['sentence_idx'] = np.concatenate([list(range(n)) for n in batch['n_sentences']])
            pred = pd.DataFrame(pred)

            preds.append(pred)
            sample_id += len(batch['n_sentences'])

        preds = pd.concat(preds, axis=0)

        metadata_dict = {}
        metadata_dict['split'] = 'test'
        metadata_dict['original_example_count'] = len(dataset)
        metadata_dict['model_id'] = self.model_ckpt
        metadata_dict['total_sentence_count'] = int(preds.shape[0])

        if save_embeddings:
            embeddings = np.concatenate(embeddings, axis=0, dtype=np.float16)
            with open(os.path.join(output_dir, 'embeddings.npy'), 'wb') as f:
                np.save(f, embeddings)
            preds['embedding_shape'] = [list(embeddings.shape[1:]) for _ in range(len(preds))]
            metadata_dict['embeddings_shape'] = embeddings.shape
            metadata_dict['dtype'] = str(embeddings.dtype)

        metadata_dict['sentence_info'] = preds[['original_idx', 'sentence_idx', 'label']].to_dict(orient="records")

        self.model.train()
        return preds, metadata_dict

    def calculate_accuracy(self, preds: pd.DataFrame, layer: int = None):
        if layer is None:
            layer = self.model.num_hidden_layers + 1  # default to the final exit point
        # Group by original document index; evaluate prediction at the last sentence of each document
        dfgroup = preds.groupby('original_idx')

        labels, predictions = [], []
        for _, group in dfgroup:
            last_sentence = max(group['sentence_idx'].tolist())
            row = group[group['sentence_idx'] == last_sentence].head(1)
            labels.append(row['label'])
            predictions.append(row[f'fc{layer}_pred'])

        return accuracy_score(labels, predictions)

