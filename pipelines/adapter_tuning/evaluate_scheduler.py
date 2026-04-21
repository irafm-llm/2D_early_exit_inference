#!/usr/bin/env python3
"""
Evaluate the 2D early-exit scheduler on per-layer / per-sentence predictions.

Consumes the hierarchical pkl produced by `train_classifiers.py`
(`{layer_idx: [{example_id, true_label, sentences: [{prediction, confidence, probabilities}]}]}`)
and sweeps a grid of (ignore_threshold, acc_threshold) combinations, producing:

- scheduler_heatmap.{png,pdf,svg}: accuracy (color) + computational savings (annotation)
- evaluation_report.txt
- best_thresholds.txt

The scheduler progressively processes sentences × layers of each example, accumulating
confidence for each primary label and stopping as soon as one crosses `acc_threshold`.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
import os

from dataset_configs import DATASET_CONFIGS

# Publication settings
PUBLICATION_DPI = 300
HEATMAP_SIZE = (14, 10)
FONT_SIZE = 12

# Optimized palette for heatmap readability
COLORS = {
    'heatmap_cmap': 'RdYlBu_r',    # Red-Yellow-Blue (reversed) - better text contrast
    'text_light': 'white',         # For dark backgrounds
    'text_dark': 'black',          # For light backgrounds
    'text_outline': 'white',       # For text outline
    'grid_color': '#666666',       # Darker gray for better visibility
    'primary': '#1f77b4',          # Blue
    'secondary': '#ff7f0e',        # Orange
    'best_highlight': '#00ff00',   # Bright green for best result
}

def setup_publication_style():
    """Set up matplotlib and seaborn styles for publication."""
    plt.rcParams.update({
        'figure.dpi': PUBLICATION_DPI,
        'savefig.dpi': PUBLICATION_DPI,
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 2,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'legend.fontsize': FONT_SIZE - 1,
        'figure.titlesize': FONT_SIZE + 4,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'text.usetex': False,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

    sns.set_style("whitegrid", {
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.8,
        "grid.color": COLORS['grid_color']
    })


def load_results(file_path):
    """Načte výsledky trénování."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def evaluate_with_scheduler(results, ignore_threshold, acc_threshold, primary_labels, skip_labels):
    """
    Evaluate one (ignore_threshold, acc_threshold) combination with the 2D scheduler.

    Returns:
        (accuracy, savings_ratio) where savings_ratio in [0, 1] (1 = 100% skipped ops)
    """

    def process_example(example_data, true_label):
        num_sentences = len(example_data[0]["sentences"])
        num_layers = len(example_data)
        total_operations = num_sentences * num_layers
        increment = max(1, num_layers // num_sentences)

        for s in range(num_sentences):
            layers_to_traverse = min(s * increment, num_layers)

            acc = {label: 0.0 for label in primary_labels}

            for l in range(layers_to_traverse):
                for s1 in range(s):
                    sentence = example_data[l]["sentences"][s1]
                    probs = np.array(sentence["probabilities"])

                    sorted_probs = np.sort(probs)[::-1]
                    confidence = sorted_probs[0] - sorted_probs[1]
                    pred_label = np.argmax(probs)

                    if pred_label in primary_labels and confidence > ignore_threshold:
                        acc[pred_label] += confidence

                    for primary_label in primary_labels:
                        if acc[primary_label] > acc_threshold:
                            operations_used = (l + 1) * (s1 + 1)
                            return primary_label, operations_used

        last_sentence = example_data[-1]["sentences"][-1]
        final_probs = np.array(last_sentence["probabilities"])
        final_pred = np.argmax(final_probs)

        return final_pred, total_operations

    total = 0
    correct = 0
    total_operations = 0
    used_operations = 0

    num_examples = len(results[0])

    for example_idx in range(num_examples):
        example_data = [results[layer][example_idx] for layer in range(len(results))]
        true_label = example_data[0]["true_label"]

        if true_label in skip_labels:
            continue

        pred_label, ops_used = process_example(example_data, true_label)

        num_sentences = len(example_data[0]["sentences"])
        num_layers = len(example_data)
        ops_total = num_sentences * num_layers

        total += 1
        if pred_label == true_label:
            correct += 1

        used_operations += ops_used
        total_operations += ops_total

    accuracy = correct / total if total > 0 else 0
    savings_ratio = 1 - (used_operations / total_operations) if total_operations > 0 else 0

    return accuracy, savings_ratio


def evaluate_threshold_combination(results, ignore_thresh, acc_thresh, primary_labels, skip_labels):
    """Worker for parallel sweep."""
    accuracy, savings = evaluate_with_scheduler(
        results, ignore_thresh, acc_thresh, primary_labels, skip_labels
    )
    return (ignore_thresh, acc_thresh, accuracy, savings)


def generate_scheduler_heatmap(results, primary_labels, skip_labels, output_dir,
                             min_acc_threshold=0.6, max_acc_threshold=2.4, colormap=None,
                             method_name=None, dataset_name=None,
                             ignore_threshold_min=0.0, ignore_threshold_max=0.5, ignore_threshold_step=0.1,
                             reference_point=None):
    """
    Sweep (ignore, acc) threshold grid in parallel and render a heatmap.

    reference_point: optional (ignore, acc) tuple to mark on the heatmap — typically the
        `optimal_thresholds` recorded in dataset_configs.py for comparison with the swept optimum.
    """
    print(f"\nGenerating scheduler heatmap")
    print(f"Acceptance threshold range: {min_acc_threshold:.1f} - {max_acc_threshold:.1f}")
    print(f"Ignore threshold range: {ignore_threshold_min:.1f} - {ignore_threshold_max:.1f} (step: {ignore_threshold_step:.1f})")
    print(f"Primary labels: {primary_labels}")
    print(f"Skipped labels: {skip_labels}")
    if reference_point is not None:
        print(f"Reference (config) optimum: ignore={reference_point[0]:.2f}, acc={reference_point[1]:.2f}")

    ignore_thresholds = np.arange(ignore_threshold_min, ignore_threshold_max + ignore_threshold_step/2, ignore_threshold_step)
    acc_range = max_acc_threshold - min_acc_threshold
    if acc_range > 2.0:
        acc_thresholds = np.arange(min_acc_threshold, max_acc_threshold + 0.1, 0.3)
    else:
        acc_thresholds = np.arange(min_acc_threshold, max_acc_threshold + 0.1, 0.2)

    accuracy_matrix = np.zeros((len(ignore_thresholds), len(acc_thresholds)))
    savings_matrix = np.zeros((len(ignore_thresholds), len(acc_thresholds)))

    total_combinations = len(ignore_thresholds) * len(acc_thresholds)

    combinations = []
    for i, ignore_thresh in enumerate(ignore_thresholds):
        for j, acc_thresh in enumerate(acc_thresholds):
            combinations.append((i, j, ignore_thresh, acc_thresh))

    print(f"Testing {total_combinations} threshold combinations in parallel...")

    n_jobs = max(1, os.cpu_count() - 1)
    print(f"Using {n_jobs} CPU cores for parallelization")

    results_list = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(evaluate_threshold_combination)(
            results, ignore_thresh, acc_thresh, primary_labels, skip_labels
        ) for i, j, ignore_thresh, acc_thresh in tqdm(combinations, desc="Preparing tasks")
    )

    print("Processing results...")
    for idx, (ignore_thresh, acc_thresh, accuracy, savings) in enumerate(results_list):
        i, j, _, _ = combinations[idx]
        accuracy_matrix[i, j] = accuracy
        savings_matrix[i, j] = savings

    best_idx = np.unravel_index(np.argmax(accuracy_matrix), accuracy_matrix.shape)
    best_ignore = ignore_thresholds[best_idx[0]]
    best_acc = acc_thresholds[best_idx[1]]
    best_accuracy = accuracy_matrix[best_idx]
    best_savings = savings_matrix[best_idx]

    print(f"\nBest Results:")
    print(f"  Accuracy: {best_accuracy:.4f}")
    print(f"  Ignore threshold: {best_ignore:.1f}")
    print(f"  Acceptance threshold: {best_acc:.1f}")
    print(f"  Computational savings: {best_savings*100:.1f}%")

    setup_publication_style()

    width = max(12, len(acc_thresholds) * 0.8)
    height = max(8, len(ignore_thresholds) * 1.5)
    fig, ax = plt.subplots(figsize=(width, height))

    chosen_cmap = colormap if colormap else COLORS['heatmap_cmap']
    im = ax.imshow(accuracy_matrix, cmap=chosen_cmap, aspect='auto',
                   interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Classification Accuracy', fontsize=FONT_SIZE+1)
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    total_cells = len(ignore_thresholds) * len(acc_thresholds)

    if total_cells <= 60:
        text_font_size = 10
    elif total_cells <= 90:
        text_font_size = 8
    else:
        text_font_size = 6

    for i in range(len(ignore_thresholds)):
        for j in range(len(acc_thresholds)):
            acc = accuracy_matrix[i, j]
            sav = savings_matrix[i, j]
            text = f"{acc:.3f}\n{sav*100:.0f}%"

            norm_value = (acc - accuracy_matrix.min()) / (accuracy_matrix.max() - accuracy_matrix.min())

            if norm_value > 0.7 or norm_value < 0.3:
                text_color = COLORS['text_light']
                outline_color = COLORS['text_dark']
            else:
                text_color = COLORS['text_dark']
                outline_color = COLORS['text_light']

            import matplotlib.patheffects as path_effects
            text_obj = ax.text(j, i, text,
                              ha='center', va='center',
                              color=text_color, fontsize=text_font_size,
                              weight='bold')
            text_obj.set_path_effects([path_effects.withStroke(linewidth=2, foreground=outline_color)])

    best_i, best_j = best_idx
    rect = plt.Rectangle((best_j-0.45, best_i-0.45), 0.9, 0.9,
                        fill=False, edgecolor=COLORS['best_highlight'], linewidth=3)
    ax.add_patch(rect)

    # Mark the config-recorded optimum, if provided, with a dashed magenta box
    if reference_point is not None:
        ref_ignore, ref_acc = reference_point
        # Snap to nearest grid cell — the grid is discrete so the reference point may not land exactly on a cell
        ref_i = int(np.argmin(np.abs(ignore_thresholds - ref_ignore)))
        ref_j = int(np.argmin(np.abs(acc_thresholds - ref_acc)))
        ref_rect = plt.Rectangle((ref_j-0.45, ref_i-0.45), 0.9, 0.9,
                                fill=False, edgecolor='magenta', linewidth=2, linestyle='--')
        ax.add_patch(ref_rect)

    if method_name and dataset_name:
        title = f'Scheduler Performance: {method_name} on {dataset_name}'
    elif method_name:
        title = f'Scheduler Performance: {method_name}'
    elif dataset_name:
        title = f'Scheduler Performance: {dataset_name}'
    else:
        title = 'Scheduler Performance Analysis'

    ax.set_title(title, fontsize=FONT_SIZE+2, pad=15)
    ax.set_xlabel(f'Acceptance Threshold', fontsize=FONT_SIZE)
    ax.set_ylabel('Ignore Threshold', fontsize=FONT_SIZE)

    ax.set_xticks(range(len(acc_thresholds)))
    x_labels = [f"{t:.1f}" for t in acc_thresholds]
    ax.set_xticklabels(x_labels, fontsize=FONT_SIZE)

    ax.set_yticks(range(len(ignore_thresholds)))
    y_labels = [f"{t:.1f}" for t in ignore_thresholds]
    ax.set_yticklabels(y_labels, fontsize=FONT_SIZE)

    ax.set_xticks(np.arange(-0.5, len(acc_thresholds), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ignore_thresholds), 1), minor=True)
    ax.grid(which="minor", color=COLORS['grid_color'], linestyle='-', linewidth=1.5, alpha=0.8)
    ax.grid(which="major", color=COLORS['grid_color'], linestyle=':', linewidth=0.8, alpha=0.6)

    plt.tight_layout()

    for fmt in ('png', 'pdf', 'svg'):
        output_path = output_dir / f'scheduler_heatmap.{fmt}'
        fig.savefig(output_path, dpi=PUBLICATION_DPI, bbox_inches='tight',
                   format=fmt, facecolor='white')

    plt.close()

    print(f"\nPublication-quality heatmap saved: {output_dir}/scheduler_heatmap.[png,pdf,svg]")

    return {
        'accuracy_matrix': accuracy_matrix,
        'savings_matrix': savings_matrix,
        'best_ignore': best_ignore,
        'best_acc': best_acc,
        'best_accuracy': best_accuracy,
        'best_savings': best_savings,
        'ignore_thresholds': ignore_thresholds,
        'acc_thresholds': acc_thresholds
    }


def analyze_dataset(results):
    """Inspect the pkl and guess primary/skip labels when no --config is given."""
    all_labels = []
    for layer_idx in sorted(results.keys()):
        layer_data = results[layer_idx]
        for example in layer_data:
            all_labels.append(example["true_label"])

    unique_labels = sorted(list(set(all_labels)))
    label_counts = {label: all_labels.count(label) for label in unique_labels}

    print("\nDataset Analysis:")
    print(f"Number of layers: {len(results)}")
    print(f"Number of examples: {len(results[0])}")
    print(f"Sentences per example: {len(results[0][0]['sentences'])}")
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} instances ({count/len(all_labels)*100:.1f}%)")

    if len(unique_labels) == 3 and 1 in unique_labels:
        primary_labels = [0, 2]
        skip_labels = [1]
        print("\nAutomatically detected 3-class classification with neutral class")
    elif len(unique_labels) == 2:
        primary_labels = unique_labels
        skip_labels = []
        print("\nAutomatically detected binary classification")
    else:
        primary_labels = unique_labels
        skip_labels = []
        print(f"\nDetected {len(unique_labels)}-class classification")

    return unique_labels, label_counts, primary_labels, skip_labels


def save_report(results, heatmap_results, label_counts, output_path, reference_point=None):
    """Save detailed evaluation report (publication format)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Scheduler Evaluation Report\n")
        f.write("===========================\n\n")

        f.write("1. DATASET INFORMATION\n")
        f.write(f"Number of layers: {len(results)}\n")
        f.write(f"Number of examples: {len(results[0])}\n")
        f.write(f"Sentences per example: {len(results[0][0]['sentences'])}\n")
        f.write("\nLabel distribution:\n")
        for label, count in label_counts.items():
            total = sum(label_counts.values())
            f.write(f"  Label {label}: {count} instances ({count/total*100:.1f}%)\n")

        f.write("\n2. OPTIMAL SCHEDULER CONFIGURATION (from sweep)\n")
        f.write(f"Classification accuracy: {heatmap_results['best_accuracy']:.4f}\n")
        f.write(f"Ignore threshold: {heatmap_results['best_ignore']:.1f}\n")
        f.write(f"Acceptance threshold: {heatmap_results['best_acc']:.1f}\n")
        f.write(f"Computational savings: {heatmap_results['best_savings']*100:.1f}%\n")

        if reference_point is not None:
            f.write("\n2b. REFERENCE OPTIMUM (from dataset_configs.optimal_thresholds)\n")
            f.write(f"Ignore threshold: {reference_point[0]:.2f}\n")
            f.write(f"Acceptance threshold: {reference_point[1]:.2f}\n")

        f.write("\n3. EVALUATION PARAMETERS\n")
        f.write(f"Ignore threshold range: {heatmap_results['ignore_thresholds'][0]:.1f} - "
                f"{heatmap_results['ignore_thresholds'][-1]:.1f}\n")
        f.write(f"Acceptance threshold range: {heatmap_results['acc_thresholds'][0]:.1f} - "
                f"{heatmap_results['acc_thresholds'][-1]:.1f}\n")
        f.write(f"Total combinations tested: "
                f"{len(heatmap_results['ignore_thresholds']) * len(heatmap_results['acc_thresholds'])}\n")

        f.write("\n4. PERFORMANCE STATISTICS\n")
        f.write(f"Mean accuracy (all combinations): "
                f"{np.mean(heatmap_results['accuracy_matrix']):.4f}\n")
        f.write(f"Mean computational savings (all combinations): "
                f"{np.mean(heatmap_results['savings_matrix'])*100:.1f}%\n")
        f.write(f"Maximum accuracy: {np.max(heatmap_results['accuracy_matrix']):.4f}\n")
        f.write(f"Maximum computational savings: {np.max(heatmap_results['savings_matrix'])*100:.1f}%\n")
        f.write(f"Minimum accuracy: {np.min(heatmap_results['accuracy_matrix']):.4f}\n")
        f.write(f"Standard deviation (accuracy): {np.std(heatmap_results['accuracy_matrix']):.4f}\n")
        f.write(f"Standard deviation (savings): {np.std(heatmap_results['savings_matrix'])*100:.1f}%\n")

        f.write("\n5. KEY FINDINGS FOR PUBLICATION\n")
        acc_matrix = heatmap_results['accuracy_matrix']
        sav_matrix = heatmap_results['savings_matrix']
        f.write(f"• Best accuracy-savings trade-off: {heatmap_results['best_accuracy']:.3f} accuracy "
                f"with {heatmap_results['best_savings']*100:.1f}% computational reduction\n")
        f.write(f"• Threshold sensitivity: accuracy range [{np.min(acc_matrix):.3f}, {np.max(acc_matrix):.3f}]\n")
        f.write(f"• Computational efficiency: savings range [{np.min(sav_matrix)*100:.1f}%, {np.max(sav_matrix)*100:.1f}%]\n")
        f.write(f"• Optimal operating point: ignore={heatmap_results['best_ignore']:.1f}, "
                f"accept={heatmap_results['best_acc']:.1f}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate 2D early-exit scheduler on adapter_tuning output')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to hierarchical pkl produced by train_classifiers.py (e.g. training_results.pkl)')
    parser.add_argument('--output', type=str, default='evaluation_v2',
                        help='Output directory for evaluation results')
    parser.add_argument('--config', type=str, default=None, choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset config name from dataset_configs.py — populates primary_labels, '
                             'skip_labels and marks the reference optimum from optimal_thresholds on the heatmap')
    parser.add_argument('--min-acc-threshold', type=float, default=0.6,
                        help='Minimum acceptance threshold (default: 0.6)')
    parser.add_argument('--max-acc-threshold', type=float, default=2.4,
                        help='Maximum acceptance threshold (default: 2.4)')
    parser.add_argument('--primary-labels', type=int, nargs='+', default=None,
                        help='Primary labels (overrides --config, e.g., --primary-labels 0 2)')
    parser.add_argument('--skip-labels', type=int, nargs='+', default=None,
                        help='Labels to skip (overrides --config, e.g., --skip-labels 1)')
    parser.add_argument('--colormap', type=str, default=None,
                        help='Colormap for heatmap (e.g., RdYlBu_r, plasma, coolwarm, viridis)')
    parser.add_argument('--method-name', type=str, default=None,
                        help='Method name for figure title')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='Dataset name for figure title (overrides --config)')
    parser.add_argument('--ignore-threshold-min', type=float, default=0.0,
                        help='Minimum ignore threshold (default: 0.0)')
    parser.add_argument('--ignore-threshold-max', type=float, default=0.5,
                        help='Maximum ignore threshold (default: 0.5)')
    parser.add_argument('--ignore-threshold-step', type=float, default=0.1,
                        help='Step size for ignore threshold range (default: 0.1)')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=== Scheduler Evaluation (adapter_tuning) ===")

    print("\n1. Loading results...")
    results = load_results(args.input)
    print(f"   Loaded {len(results)} layers with {len(results[0])} examples each")

    print("\n2. Resolving labels and thresholds...")

    # --config fills primary_labels, skip_labels, dataset_name and reference optimum;
    # explicit CLI flags still override.
    config_primary, config_skip, config_reference, config_dataset_name = None, None, None, None
    if args.config is not None:
        cfg = DATASET_CONFIGS[args.config]
        label_cfg = cfg["label_config"]
        config_primary = label_cfg["primary_labels"]
        config_skip = label_cfg["skip_labels"]
        opt = cfg.get("optimal_thresholds")
        if opt is not None:
            config_reference = (opt["ignore_threshold"], opt["acc_threshold"])
        config_dataset_name = cfg["dataset_name"].split("/")[-1]
        print(f"   Loaded config '{args.config}': primary={config_primary}, skip={config_skip}")
        if config_reference is not None:
            print(f"   Reference optimum from config: ignore={config_reference[0]}, acc={config_reference[1]}")

    # Fall back to dataset auto-detection only when --config not given and CLI labels missing
    _, label_counts, auto_primary, auto_skip = analyze_dataset(results)

    primary_labels = args.primary_labels if args.primary_labels else (config_primary if config_primary is not None else auto_primary)
    skip_labels = args.skip_labels if args.skip_labels is not None else (config_skip if config_skip is not None else auto_skip)
    dataset_name = args.dataset_name if args.dataset_name else config_dataset_name

    print(f"\nConfiguration used:")
    print(f"  Primary labels: {primary_labels}")
    print(f"  Skipped labels: {skip_labels}")
    if dataset_name:
        print(f"  Dataset name (title): {dataset_name}")

    print("\n3. Generating scheduler heatmap...")
    heatmap_results = generate_scheduler_heatmap(
        results, primary_labels, skip_labels, output_dir,
        args.min_acc_threshold, args.max_acc_threshold, args.colormap,
        args.method_name, dataset_name,
        args.ignore_threshold_min, args.ignore_threshold_max, args.ignore_threshold_step,
        reference_point=config_reference,
    )

    print("\n4. Saving evaluation report...")
    save_report(results, heatmap_results, label_counts,
                output_dir / 'evaluation_report.txt',
                reference_point=config_reference)
    print(f"   Generated: evaluation_report.txt")

    print("\n5. Saving optimal thresholds...")
    with open(output_dir / 'best_thresholds.txt', 'w') as f:
        f.write(f"ignore_threshold: {heatmap_results['best_ignore']}\n")
        f.write(f"acc_threshold: {heatmap_results['best_acc']}\n")
        f.write(f"accuracy: {heatmap_results['best_accuracy']}\n")
        f.write(f"savings: {heatmap_results['best_savings']}\n")
    print(f"   Generated: best_thresholds.txt")

    print(f"\nOutputs saved to: {output_dir}/")
    print(f"Best configuration: accuracy={heatmap_results['best_accuracy']:.3f}, "
          f"savings={heatmap_results['best_savings']*100:.1f}%")


if __name__ == "__main__":
    main()
