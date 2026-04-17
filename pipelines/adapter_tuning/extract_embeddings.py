#!/usr/bin/env python3
"""
Extract embeddings from all layers of a causal language model for each sentence in the dataset.

Outputs per-sentence mean embeddings: embeddings.npy [n_sentences, n_layers, embedding_dim]

Refactored from: new_pipeline/sentence_processing_hf.py

Usage:
    python core/extract_embeddings.py --config amazon_reviews
    python core/extract_embeddings.py --config amazon_reviews --model meta-llama/Meta-Llama-3.1-8B-Instruct
    python core/extract_embeddings.py --config two_class_reviews --device cuda:1 --batch-size 4
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import logging
from datetime import datetime
import json
import gc
import pandas as pd
import argparse

from dataset_configs import DATASET_CONFIGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_id: str, device: str, cache_dir=None):
    """Load model and tokenizer from HuggingFace."""
    logger.info(f"Loading model {model_id} on device {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "cache_dir": cache_dir,
        "torch_dtype": torch.float16,
    }

    if device.startswith("cuda"):
        model_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    if not device.startswith("cuda"):
        model = model.to(device)

    model.eval()

    # Determine max context length
    model_max_length = getattr(model.config, "max_position_embeddings", None)
    if model_max_length is None:
        model_max_length = getattr(tokenizer, "model_max_length", 2048)
    if model_max_length > 32000:
        model_max_length = 8192
    tokenizer.model_max_length = model_max_length
    logger.info(f"Max sequence length: {model_max_length}")

    return model, tokenizer


def normalize_text(text):
    """Normalize text - remove newlines, tabs, standardize quotes."""
    text = text.replace("\n", "").replace("\t", "").replace("\r", "")
    text = text.replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'").replace("`", "'")
    text = text.replace("\\'", "'").replace('\\"', '"')
    text = " ".join(text.split())
    return text


def get_token_indices_for_sentence(sentence: str, prompt: str,
                                   tokenizer: AutoTokenizer,
                                   input_ids: torch.Tensor) -> List[int]:
    """Identify token indices corresponding to a sentence within the prompt."""
    seq_length = input_ids.shape[1]

    text_pos = prompt.find(sentence)
    if text_pos == -1:
        logger.warning(f"Sentence not found in prompt, using full sequence: '{sentence[:60]}...'")
        return list(range(seq_length))

    encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False, return_tensors="np")
    offset_mapping = encoding["offset_mapping"][0]

    token_indices = set()
    text_end = text_pos + len(sentence)

    for i, (start, end) in enumerate(offset_mapping):
        if start < text_end and end > text_pos and i < seq_length:
            token_indices.add(i)

    if not token_indices:
        logger.warning(f"No tokens found for sentence, using full sequence: '{sentence[:60]}...'")
        return list(range(seq_length))

    return sorted(list(token_indices))


def process_and_save(dataset, model, tokenizer, output_dir, split, max_length, batch_size=8):
    """
    Process dataset and save mean embeddings from all layers for each sentence.

    Output files:
        embeddings.npy  - [n_sentences, n_layers, embedding_dim]
        sentences.npy   - sentence texts
        labels.npy      - labels per sentence
        metadata.json   - processing metadata
    """
    logger.info(f"Processing {split} split - {len(dataset)} examples")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_subdir = Path(output_dir) / f"embeddings_{split}_{timestamp}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    embeddings_file = output_subdir / "embeddings.npy"
    sentences_file = output_subdir / "sentences.npy"
    labels_file = output_subdir / "labels.npy"
    metadata_file = output_subdir / "metadata.json"

    metadata = {
        "split": split,
        "original_example_count": len(dataset),
        "model_id": model.config._name_or_path,
        "dtype": "float16",
        "processing_time": datetime.now().isoformat(),
        "sentence_info": []
    }

    num_batches = (len(dataset) + batch_size - 1) // batch_size
    all_sentences = []
    all_embeddings = []
    all_labels = []
    sentence_count = 0

    for batch_idx in tqdm(range(num_batches), desc=f"Processing {split}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_data = [dataset[i] for i in range(start_idx, end_idx)]

        for i, item in enumerate(batch_data):
            row_idx = start_idx + i

            full_text = " ".join(item["sentences"])
            sentences = item["sentences"]
            label = item.get("label", None)

            clean_full_text = normalize_text(full_text)
            clean_sentences = [normalize_text(s) for s in sentences]

            prompt = PROMPT_TEMPLATE.format(text=clean_full_text)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    output_hidden_states=True,
                )

            hidden_states = [hs[0].cpu().numpy() for hs in outputs.hidden_states]
            hidden_states_array = np.stack(hidden_states)

            for sentence_idx, sentence in enumerate(clean_sentences):
                token_indices = get_token_indices_for_sentence(
                    sentence, prompt, tokenizer, inputs["input_ids"]
                )

                reduced_states = hidden_states_array[:, token_indices, :]
                avg_embeddings = np.mean(reduced_states, axis=1)

                all_embeddings.append(avg_embeddings)
                all_sentences.append(sentence)
                all_labels.append(label)

                metadata["sentence_info"].append({
                    "original_idx": row_idx,
                    "sentence_idx": sentence_idx,
                    "embedding_shape": list(avg_embeddings.shape),
                    **({"label": label} if label is not None else {})
                })
                sentence_count += 1

        torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"Saving {sentence_count} processed sentences")
    metadata["total_sentence_count"] = sentence_count

    stacked_embeddings = np.stack(all_embeddings)
    np.save(embeddings_file, stacked_embeddings)
    metadata["embeddings_shape"] = list(stacked_embeddings.shape)

    np.save(sentences_file, all_sentences)
    np.save(labels_file, all_labels)

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    total_size_mb = sum(os.path.getsize(f) for f in [embeddings_file, sentences_file, labels_file, metadata_file] if f.exists()) / (1024 * 1024)
    logger.info(f"Data saved to: {output_subdir} ({total_size_mb:.2f} MB, {sentence_count} sentences)")

    return str(output_subdir)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from all layers of a causal LM")
    parser.add_argument("--config", type=str, default="amazon_reviews",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset configuration name")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: output_hf/<model>/<dataset>)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda:0 if available)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"],
                        help="Dataset splits to process")
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    config = DATASET_CONFIGS[args.config]
    dataset_name = config["dataset_name"]

    # Set global prompt template
    global PROMPT_TEMPLATE
    PROMPT_TEMPLATE = config["prompt_template"]

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    model_short = args.model.split("/")[-1]
    output_dir = args.output_dir or f"output_hf/{model_short}/{dataset_name.split('/')[-1]}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model, device, args.cache_dir)
    max_length = tokenizer.model_max_length

    output_files = {}
    for split in args.splits:
        dataset = load_dataset(dataset_name, split=split, cache_dir=args.cache_dir)

        df = dataset.to_pandas()
        if 'label' not in df.columns:
            df['label'] = df['class_index'] - 1
        if 'text' not in df.columns:
            df['text'] = df['review_text']
        dataset = Dataset.from_pandas(df)

        output_file = process_and_save(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            split=split,
            max_length=max_length,
            batch_size=args.batch_size,
        )
        output_files[split] = output_file

    metadata = {
        "files": output_files,
        "model_id": args.model,
        "processing_time": datetime.now().isoformat(),
        "dtype": "float16",
    }
    with open(output_dir / "processing_metadata.txt", "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
