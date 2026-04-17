#!/usr/bin/env python3
"""
Prepare data for analysis - convert various input formats to the standard hierarchical format
expected by visualization and evaluation scripts.

Refactored from: new_pipeline_clean/scripts/prepare_data_for_analysis.py

Expected output format:
{
    layer_idx: [
        {
            "example_id": int,
            "true_label": int,
            "sentences": [
                {"prediction": int, "confidence": float, "probabilities": [float, ...]}
            ]
        }
    ]
}

Usage:
    python core/prepare_data.py --input raw_data.pkl --output training_results_v2.pkl
    python core/prepare_data.py --create-sample --output sample_data.pkl
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import json


def softmax(logits):
    """Numerically stable softmax."""
    if isinstance(logits, list):
        logits = np.array(logits)
    if logits.size == 0 or not np.isfinite(logits).all():
        num_classes = len(logits) if hasattr(logits, '__len__') else 3
        return np.ones(num_classes) / num_classes
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def detect_input_format(file_path):
    """Detect input data format."""
    print(f"Detecting format: {file_path}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, pd.DataFrame):
        print("Detected: Pandas DataFrame")
        return "dataframe", data
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            if "predictions_by_layer" in data[0] and "probabilities_by_layer" in data[0]:
                print("Detected: Imported format (list with predictions_by_layer)")
                return "imported", data
        print("Unknown list format")
        return "unknown", data
    elif isinstance(data, dict):
        first_key = list(data.keys())[0]
        if isinstance(data[first_key], list) and len(data[first_key]) > 0:
            first_example = data[first_key][0]
            if "sentences" in first_example and "true_label" in first_example:
                print("Detected: Hierarchical format (already correct)")
                return "hierarchical", data
        print("Detected: Dictionary format")
        return "dict", data
    else:
        print("Unknown format")
        return "unknown", data


def convert_dataframe_to_hierarchical(df):
    """Convert DataFrame to hierarchical structure."""
    print("Converting DataFrame to hierarchical format...")

    results = {}

    layer_cols = [col for col in df.columns if 'fc' in col and ('logits' in col or 'pred' in col)]
    layer_indices = sorted(list(set([int(col.split('_')[0].replace('fc', '')) for col in layer_cols])))

    print(f"   Found layers: {layer_indices}")

    has_sentence_id = 'sentence_id' in df.columns

    for layer_idx in tqdm(layer_indices, desc="Processing layers"):
        layer_examples = []

        for sample_id in df['sample_id'].unique():
            sample_data = df[df['sample_id'] == sample_id]
            if has_sentence_id:
                sample_data = sample_data.sort_values('sentence_id')

            if len(sample_data) == 0:
                continue

            sentences = []
            for _, row in sample_data.iterrows():
                logits_col = f'fc{layer_idx}_logits'

                if logits_col in row:
                    logits = row[logits_col]
                    if isinstance(logits, str):
                        try:
                            logits = eval(logits)
                        except Exception:
                            logits = [0.1, 0.8, 0.1]
                else:
                    logits = [0.1, 0.8, 0.1]

                probabilities = softmax(logits).tolist()
                prediction = np.argmax(probabilities)

                sorted_probs = sorted(probabilities, reverse=True)
                confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 0.5

                sentences.append({
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": probabilities
                })

            layer_examples.append({
                "example_id": sample_id,
                "true_label": int(sample_data.iloc[0]['label']),
                "sentences": sentences
            })

        results[layer_indices.index(layer_idx)] = layer_examples

    print(f"Converted {len(results)} layers with {len(results[0]) if results else 0} examples each")
    return results


def convert_imported_to_hierarchical(data):
    """Convert imported format (list with predictions_by_layer) to hierarchical structure."""
    print("Converting imported format to hierarchical format...")

    results = {}

    all_layers = set()
    for example in data:
        all_layers.update(example['predictions_by_layer'].keys())

    layer_ids = sorted(all_layers)
    print(f"   Found layers: {layer_ids}")

    for layer_idx, layer_id in enumerate(tqdm(layer_ids, desc="Processing layers")):
        layer_examples = []

        for example in data:
            if layer_id not in example['predictions_by_layer']:
                continue

            sentences_data = example.get('sentences', {}).get(layer_id, [])

            sentences = []
            for sent_data in sentences_data:
                probs = sent_data.get('probabilities', [])
                if isinstance(probs, np.ndarray):
                    probs = probs.tolist()

                sorted_probs = sorted(probs, reverse=True)
                confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

                sentences.append({
                    "prediction": int(sent_data.get('prediction', 0)),
                    "confidence": float(confidence),
                    "probabilities": probs
                })

            if not sentences:
                probs = example['probabilities_by_layer'][layer_id]
                if isinstance(probs, np.ndarray):
                    probs = probs.tolist()

                sorted_probs = sorted(probs, reverse=True)
                confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

                sentences.append({
                    "prediction": int(example['predictions_by_layer'][layer_id]),
                    "confidence": float(confidence),
                    "probabilities": probs
                })

            layer_examples.append({
                "example_id": example['example_id'],
                "true_label": int(example['true_label']),
                "sentences": sentences
            })

        results[layer_idx] = layer_examples

    print(f"Converted {len(results)} layers with {len(results[0]) if results else 0} examples each")
    return results


def create_sample_data():
    """Create sample data for testing."""
    print("Creating sample data for testing...")

    np.random.seed(42)
    num_layers = 8
    num_examples = 100
    num_sentences = 5
    num_classes = 3

    results = {}

    for layer_idx in range(num_layers):
        layer_examples = []
        for example_id in range(num_examples):
            true_label = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])

            sentences = []
            for _ in range(num_sentences):
                base_logits = np.random.normal(0, 1, num_classes)
                base_logits[true_label] += (layer_idx + 1) * 0.3

                probabilities = softmax(base_logits).tolist()
                prediction = np.argmax(probabilities)
                sorted_probs = sorted(probabilities, reverse=True)
                confidence = sorted_probs[0] - sorted_probs[1]

                sentences.append({
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": probabilities
                })

            layer_examples.append({
                "example_id": example_id,
                "true_label": true_label,
                "sentences": sentences
            })

        results[layer_idx] = layer_examples

    print(f"Created {num_layers} layers with {num_examples} examples each")
    return results


def save_results(results, output_path):
    """Save results to pickle file with metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved: {output_path}")

    metadata = {
        "created_at": pd.Timestamp.now().isoformat(),
        "num_layers": len(results),
        "num_examples": len(list(results.values())[0]) if results else 0,
        "num_sentences": len(list(results.values())[0][0]["sentences"]) if results and list(results.values()) else 0,
        "format_version": "v2"
    }

    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Prepare data for analysis')
    parser.add_argument('--input', type=str, default=None,
                        help='Input data file (pkl). If not provided, creates sample data.')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for training_results_v2.pkl')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create sample data instead of loading input file')

    args = parser.parse_args()

    print("=== Prepare Data for Analysis ===\n")

    if args.create_sample or not args.input:
        results = create_sample_data()
    else:
        print("1. Loading input data...")
        format_type, data = detect_input_format(args.input)

        if format_type == "error" or data is None:
            print("Failed to load input data. Creating sample data instead...")
            results = create_sample_data()
        elif format_type == "hierarchical":
            print("Data already in correct format!")
            results = data
        elif format_type == "imported":
            results = convert_imported_to_hierarchical(data)
        elif format_type == "dataframe":
            results = convert_dataframe_to_hierarchical(data)
        else:
            print("Unsupported format. Creating sample data instead...")
            results = create_sample_data()

    print("\n2. Saving results...")
    output_path = save_results(results, args.output)

    print("\n3. Validating results...")
    if results and len(results) > 0:
        first_layer = results[list(results.keys())[0]]
        print(f"   - Layers: {len(results)}")
        print(f"   - Examples: {len(first_layer)}")
        print(f"   - Sentences per example: {len(first_layer[0]['sentences'])}")
        print(f"   - Classes: {len(first_layer[0]['sentences'][0]['probabilities'])}")

    print(f"\nDone! Now you can run:")
    print(f"  python scripts/visualization/visualize_training_results_v2.py --input {output_path}")
    print(f"  python scripts/evaluation/evaluate_results_v2_parallel.py --input {output_path}")


if __name__ == "__main__":
    main()
