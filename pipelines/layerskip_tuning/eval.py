import os

# Select GPUs before importing torch/transformers to avoid CUDA init issues
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

import torch
import config
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

labels = list(config.label2idx.keys())


def map_label(sample):
    # Convert integer class_index to string label used in the prompt
    idx = sample['class_index']
    label = config.idx2label[idx]
    return {'label': label}


def formatting_prompts_func(example):
    # Build inference prompt — no label appended, model must generate it
    text = f"### Instruction: {example['text']}\n ### Response: "

    # Inject eos_token as a string before tokenization, because they are not always added
    # See: https://github.com/huggingface/transformers/issues/22794 and
    # https://github.com/huggingface/trl/issues/1623
    #if tokenizer.eos_token:  # usually something like "</s>" for GPT2 or "<|endoftext|>"
    #    text += f"{tokenizer.eos_token}"

    return {'text': text}


def generate_tokens_with_assistance(model, inputs, assistant_early_exit):
    # Greedy decode with LayerSkip early exit; 3 new tokens is enough for a single digit label
    outputs = model.generate(
        **inputs,
        assistant_early_exit=assistant_early_exit,
        do_sample=False,
        max_new_tokens=3,
    )
    return outputs.cpu()


def parse_output_label(output):
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)

    # Locate the response section; return None if the model didn't follow the template
    try:
        idx = output_str.index('Response: ') + len('Response: ')
    except ValueError:
        return {'label_text': None, 'pred_label': None}
    output_str = output_str[idx:].strip()

    # Find first matching label string in the generated text
    parsed_label = None
    for label in labels:
        if label in output_str:
            parsed_label = label
            break

    if parsed_label is None:
        return {'label_text': None, 'pred_label': None}

    return {'label_text': parsed_label, 'pred_label': config.label2idx[parsed_label]}


if __name__ == "__main__":
    dtype = torch.bfloat16 if config.bf16_training else torch.float32
    model_ckpt = f"{config.model_name.replace('/', '_')}_{config.dataset_name.replace('/', '_')}"

    # --- Model & Tokenizer ---
    model = AutoModelForCausalLM.from_pretrained(model_ckpt, device_map="auto", dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # --- Dataset ---
    dataset = load_dataset(config.dataset_name, split="test")
    dataset = dataset.rename_column('review_text', 'text')
    dataset = dataset.map(map_label)
    print(dataset)
    dataset = dataset.map(formatting_prompts_func)
    dataset = dataset.shuffle(seed=42)

    # --- Evaluation loop ---
    y_true, y_pred = [], []
    skipped = 0
    for sample in tqdm(dataset, total=len(dataset)):
        inputs = tokenizer(sample['text'], return_tensors="pt").to(model.device)
        output = generate_tokens_with_assistance(model, inputs, assistant_early_exit=config.eval_skip_layer)
        label_dict = parse_output_label(output)
        if label_dict['pred_label'] is None:
            skipped += 1
            continue
        y_pred.append(label_dict['pred_label'])
        y_true.append(sample['class_index'])

    print(f'accuracy score: {accuracy_score(y_true, y_pred)} (skipped {skipped} unparseable samples)')
    with open('metrics.txt', 'w') as f:
        f.write(f'accuracy score: {accuracy_score(y_true, y_pred)}\nskipped: {skipped}')
