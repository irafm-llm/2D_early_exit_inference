#!/usr/bin/env python3
"""
Train per-layer classifiers on extracted embeddings.

For each layer, trains a 2-layer perceptron and evaluates on test data.
Results are organized by examples with per-sentence predictions.

Refactored from: new_pipeline/train_emb_v3_universal.py

Usage:
    python core/train_classifiers.py --model llama_3_1_8b --train-dir ... --test-dir ... --output-dir ...
    python core/train_classifiers.py --model gemma --train-dir ... --test-dir ... --output-dir ...
    python core/train_classifiers.py --list-models
"""

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import pickle
from collections import defaultdict
from tqdm import tqdm
import argparse

from model_configs import get_model_config, list_available_models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleClassifier(nn.Module):
    """2-layer perceptron: Linear -> ReLU -> Dropout -> Linear."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class SentenceDataset(Dataset):
    """Dataset for sentences from a single layer."""
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def preprocess_embeddings(embeddings, model_config):
    """Model-specific preprocessing of embeddings (e.g. Gemma extra dim squeeze)."""
    original_shape = embeddings.shape
    print(f"    Original shape: {original_shape}")

    if model_config.get("squeeze_extra_dim", False):
        if model_config.get("ndim_check", False) and embeddings.ndim > 2:
            squeeze_axis = model_config.get("squeeze_axis")
            shape_check = model_config.get("shape_check")
            if shape_check and embeddings.shape[squeeze_axis] in shape_check:
                print(f"    Applying squeeze on axis {squeeze_axis}")
                embeddings = embeddings.squeeze(squeeze_axis)
                print(f"    After squeeze: {embeddings.shape}")

    print(f"    Final shape: {embeddings.shape}")
    return embeddings


def load_data(data_dir, model_config):
    """Load embeddings, labels, and metadata from directory."""
    data_dir = Path(data_dir)

    embeddings = np.load(data_dir / "embeddings.npy")
    labels = np.load(data_dir / "labels.npy")

    embeddings = preprocess_embeddings(embeddings, model_config)

    # Convert 1-based to 0-based labels
    if labels.min() == 1:
        labels = labels - 1

    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    sentence_info = metadata["sentence_info"]

    print(f"Loaded:")
    print(f"  - {len(sentence_info)} sentences")
    print(f"  - {embeddings.shape[1]} layers")
    print(f"  - {embeddings.shape[2]} embedding dim")
    print(f"  - {len(np.unique(labels))} classes")

    expected_layers = model_config.get("expected_layers")
    expected_embed_dim = model_config.get("expected_embed_dim")

    if expected_layers and embeddings.shape[1] != expected_layers:
        print(f"  WARNING: Expected {expected_layers} layers, got {embeddings.shape[1]}")
    if expected_embed_dim and embeddings.shape[2] != expected_embed_dim:
        print(f"  WARNING: Expected {expected_embed_dim} embedding dim, got {embeddings.shape[2]}")

    return embeddings, labels, sentence_info


def train_classifier(embeddings, labels, num_classes, model_config):
    """Train classifier for a single layer."""
    input_dim = embeddings.shape[1]
    hidden_dim = model_config["hidden_dim"]
    dropout = model_config["dropout"]
    epochs = model_config["epochs"]
    batch_size = model_config["batch_size"]

    print(f"    Input shape: {embeddings.shape}, classes: {num_classes}")

    model = SimpleClassifier(input_dim, hidden_dim, num_classes, dropout).to(DEVICE)

    dataset = SentenceDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    optimizer_type = model_config.get("optimizer", "adamw").lower()
    lr = model_config["learning_rate"]
    wd = model_config.get("weight_decay", 1e-4)
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler_type = model_config.get("scheduler", "reduce_plateau").lower()
    scheduler = None
    if scheduler_type == "reduce_plateau":
        params = model_config.get("scheduler_params", {})
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for batch_emb, batch_lbl in dataloader:
            batch_emb = batch_emb.to(DEVICE)
            batch_lbl = batch_lbl.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_emb)
            loss = criterion(outputs, batch_lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        if scheduler:
            scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch < 5:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

    return model


def predict_all(model, embeddings, batch_size):
    """Get predictions and probabilities for all sentences."""
    model.eval()
    dataset = SentenceDataset(embeddings, np.zeros(len(embeddings)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch_emb, _ in dataloader:
            batch_emb = batch_emb.to(DEVICE)
            outputs = model(batch_emb)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    return all_predictions, all_probabilities


def organize_by_examples(predictions, probabilities, labels, sentence_info):
    """Organize predictions by examples (not individual sentences)."""
    examples = defaultdict(lambda: {"sentences": [], "true_label": None})

    for i, info in enumerate(sentence_info):
        example_id = info["original_idx"]
        sentence_idx = info["sentence_idx"]

        if examples[example_id]["true_label"] is None:
            examples[example_id]["true_label"] = int(labels[i])

        examples[example_id]["sentences"].append({
            "sentence_idx": sentence_idx,
            "prediction": int(predictions[i]),
            "probabilities": probabilities[i].tolist()
        })

    result = []
    for example_id in sorted(examples.keys()):
        examples[example_id]["sentences"].sort(key=lambda x: x["sentence_idx"])
        result.append({
            "example_id": example_id,
            "true_label": examples[example_id]["true_label"],
            "sentences": examples[example_id]["sentences"]
        })

    return result


def main():
    parser = argparse.ArgumentParser(description='Train per-layer classifiers on embeddings')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (llama_3_1_8b, llama_3_2_3b, qwen, gemma)')
    parser.add_argument('--train-dir', type=str, required=True,
                        help='Directory with training embeddings')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Directory with test embeddings')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--list-models', action='store_true',
                        help='List available model configurations')

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        return

    model_config = get_model_config(args.model)

    print("=== Train Classifiers ===\n")
    print(f"Model: {model_config['display_name']} ({args.model})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n1. Loading training data...")
    train_embeddings, train_labels, train_info = load_data(args.train_dir, model_config)

    print("\n2. Loading test data...")
    test_embeddings, test_labels, test_info = load_data(args.test_dir, model_config)

    num_layers = train_embeddings.shape[1]
    num_classes = len(np.unique(train_labels))

    print(f"\n3. Overview:")
    print(f"  - Layers: {num_layers}")
    print(f"  - Classes: {num_classes}")
    print(f"  - Embedding dim: {train_embeddings.shape[2]}")
    print(f"  - Device: {DEVICE}")

    results = {}

    print(f"\n4. Training classifiers for {num_layers} layers...")
    for layer_idx in range(num_layers):
        print(f"\n  Layer {layer_idx}:")

        layer_train = train_embeddings[:, layer_idx, :]
        layer_test = test_embeddings[:, layer_idx, :]

        classifier = train_classifier(layer_train, train_labels, num_classes, model_config)
        predictions, probabilities = predict_all(classifier, layer_test, model_config["batch_size"])
        examples = organize_by_examples(predictions, probabilities, test_labels, test_info)

        correct = sum(1 for ex in examples
                      if ex["sentences"][-1]["prediction"] == ex["true_label"])
        accuracy = correct / len(examples)
        print(f"    Accuracy: {accuracy:.4f} ({correct}/{len(examples)})")

        results[layer_idx] = examples

    # Save results
    output_path = output_dir / "training_results.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n5. Results saved to: {output_path}")

    # Save summary
    summary_path = output_dir / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Train Classifiers - {model_config['display_name']}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Hyperparameters: epochs={model_config['epochs']}, lr={model_config['learning_rate']}, "
                f"dropout={model_config['dropout']}, hidden_dim={model_config['hidden_dim']}\n")
        f.write(f"Layers: {num_layers}, Classes: {num_classes}, Embed dim: {train_embeddings.shape[2]}\n")
        f.write(f"Train sentences: {len(train_labels)}, Test sentences: {len(test_labels)}\n\n")
        f.write("Per-layer accuracy (last sentence method):\n")
        for layer_idx in range(num_layers):
            examples = results[layer_idx]
            correct = sum(1 for ex in examples
                          if ex["sentences"][-1]["prediction"] == ex["true_label"])
            accuracy = correct / len(examples)
            f.write(f"  Layer {layer_idx:2d}: {accuracy:.4f}\n")

    print(f"Summary: {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
