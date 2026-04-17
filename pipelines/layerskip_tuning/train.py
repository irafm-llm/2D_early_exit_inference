import os

# Select GPUs before importing torch/transformers to avoid CUDA init issues
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import config
from custom_trainer import LayerSkipSFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from peft import get_peft_model
from trl import DataCollatorForCompletionOnlyLM, SFTConfig

tokenizer = AutoTokenizer.from_pretrained(config.model_name, add_eos_token=True)


def map_label(sample):
    # Convert integer class_index to string label expected by the prompt template
    idx = sample['class_index']
    label = config.idx2label[idx]
    return {'label_text': label}


def formatting_prompts_func(example):
    # Truncate input to max_length tokens to avoid exceeding model context
    input_ids = tokenizer(example['text'], add_special_tokens=False)['input_ids']
    input_text = tokenizer.decode(input_ids)[:config.max_length]
    text = f"### Instruction: {input_text}\n ### Response: {example['label_text']}"

    # Inject eos_token as a string before tokenization, because they are not always added
    # See: https://github.com/huggingface/transformers/issues/22794 and
    # https://github.com/huggingface/trl/issues/1623
    if tokenizer.eos_token:  # usually something like "</s>" for GPT2 or "<|endoftext|>"
        text += f"{tokenizer.eos_token}"

    return {'text': text}


if __name__ == "__main__":
    # --- Dataset ---
    print("[INFO] loading the dataset...")
    train_dataset = load_dataset(config.dataset_name, split="train")
    train_dataset = train_dataset.rename_column('review_text', 'text')
    train_dataset = train_dataset.map(map_label)
    train_dataset = train_dataset.map(formatting_prompts_func)
    train_dataset = train_dataset.shuffle(seed=42)

    print(f"output_root_dir: {config.output_root_dir}")
    print(f"hub_model_id: {config.hub_model_id}")

    # --- Model & Tokenizer ---
    print("[INFO] loading the model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto", dtype=torch.bfloat16)

    # Add pad/eos tokens if missing — required for correct loss masking and sequence termination
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.eos_token is None or tokenizer.eos_token == tokenizer.bos_token:
        tokenizer.add_special_tokens({"eos_token": "[EOS]"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.eos_token_id = tokenizer.eos_token_id

    # Only compute loss on the response tokens, not the instruction
    response_template = " ### Response:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    print(train_dataset)

    # --- Optional LoRA ---
    if config.peft_training:
        # Preserve the norm layer before PEFT wrapping — LayerSkip applies it manually at early exits
        norm = model.model.norm
        lora_config = LoraConfig(
                r=64,
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=16,
                lora_dropout=0.05)

        model = get_peft_model(model, lora_config)
        model.model.norm = norm
        print(model.print_trainable_parameters())

    # --- Training ---
    args = SFTConfig(
        do_train=True,
        bf16=config.bf16_training,
        max_seq_length=None,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        gradient_checkpointing=True,
        dataset_text_field='text',
        num_train_epochs=config.num_train_epochs,
        report_to="none",
        push_to_hub=False,
        hub_model_id=config.hub_model_id,
        output_dir=config.output_dir,
        save_steps=1000,
        save_total_limit=config.save_total_limit,
    )

    trainer = LayerSkipSFTTrainer(
        model,
        train_dataset=train_dataset,
        args=args,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(f"{config.model_name.replace('/', '_')}_{config.dataset_name.replace('/', '_')}")