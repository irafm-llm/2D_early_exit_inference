#### [Paper (PDF)](docs/2D%20early%20exit.pdf)

The project introduces a two-dimensional (2D) early exit strategy that coordinates layer-wise and sentence-wise exiting for classification tasks in large language models. By processing input
incrementally sentence-by-sentence while progressively activating deeper layers, our method achieves multiplicative computational savings that exceed those from optimizing either dimension
independently. Experimental evaluation across four state-of-the-art LLMs (Llama 3.1, Llama 3.2, Gemma, Qwen; 3B-8B parameters) on three sentiment classification datasets demonstrates additional speed-ups of 1.4–2.3× over optimal layer-wise early exit for simpler tasks with vanilla models, with graceful degradation on complex multi-class problems. The approach is model-agnostic, requires only lightweight classification adapters, and is orthogonal to complementary efficiency methods such as quantization and pruning.

---

## Datasets

Datasets used in the paper and experiments:

- [Amazon Reviews 5-class](https://huggingface.co/datasets/davidadamczyk/Amazon_reviews_5cl-v2) — 5-class sentiment classification
- [MMS Subset](https://huggingface.co/datasets/davidadamczyk/mms_subset) — 3-class sentiment classification
- [Two-class Reviews](https://huggingface.co/datasets/davidadamczyk/two_class_reviews) — binary sentiment classification

---

## Repository Structure

```
2D_early_exit_inference/
├── pipelines/
│   ├── adapters_plus_backbone_tuning/   # Adapter fine-tuning trained jointly with the backbone (optional LoRA via PEFT)
│   │   ├── config.py         # Model, dataset, training hyperparameters — update before running
│   │   ├── train.py          # Fine-tuning entry point
│   │   ├── model.py          # Adapter module definition
│   │   └── utils.py          # Data loading and preprocessing helpers
│   │
│   ├── adapter_tuning/       # Frozen LLM hidden state extraction + per-layer MLP classifiers
│   │   ├── extract_embeddings.py   # Extract per-sentence mean embeddings from all layers
│   │   ├── train_classifiers.py    # Train one MLP classifier per layer on extracted embeddings
│   │   ├── dataset_configs.py      # Dataset configurations and prompt templates
│   │   └── model_configs.py        # Per-model classifier hyperparameters
│   │
│   └── layerskip_tuning/     # LayerSkip SFT training + early-exit evaluation
│       ├── config.py         # Model, dataset, and eval hyperparameters — update before running
│       ├── train.py          # SFT training entry point (LayerSkip-aware trainer)
│       ├── eval.py           # Early-exit evaluation and accuracy reporting
│       └── custom_trainer.py # LayerSkipSFTTrainer with rotating early-exit loss
│
└── early_exit/               # 2D early exit algorithm (consumes output from any pipeline)
    └── prepare_data.py       # Normalize pkl format from any pipeline into standard hierarchical format
```

---

## Configuration

Before running any script, update the relevant `config.py`:

**`pipelines/adapters_plus_backbone_tuning/config.py`**
- `model_ckpt` — HuggingFace model ID (e.g. `meta-llama/Llama-3.1-8B-Instruct`)
- `dataset_name` — HuggingFace dataset ID
- `dataset_config` — set `text_column_name`, `label_column_name`, `num_labels`, and `label_names` to match your dataset

**`pipelines/layerskip_tuning/config.py`**
- `model_name` / `tokenizer_name` — HuggingFace model ID
- `dataset_name` — HuggingFace dataset ID
- `eval_skip_layer` — controls which intermediate layer is used to make predictions during evaluation (instead of the final layer); accuracy is computed from this layer's output
- `idx2label` / `label2idx` — update to match your dataset's class labels

---

## Running

### Pipeline 1: Adapters + Backbone Fine-tuning

```bash
cd pipelines/adapters_plus_backbone_tuning
python train.py
```

Trains adapters jointly with the backbone. LoRA (PEFT) can be enabled via `peft_training` in `TrainingConfig`. Outputs `test_preds.pkl` with per-layer predictions.

### Pipeline 2: Adapter Fine-tuning

```bash
cd pipelines/adapter_tuning
python extract_embeddings.py --config amazon_reviews --model meta-llama/Meta-Llama-3.1-8B-Instruct
python train_classifiers.py --model llama_3_1_8b --train-dir <train_embeddings_dir> --test-dir <test_embeddings_dir> --output-dir <output_dir>
```

Extracts hidden states from all layers of a frozen LLM, then trains one MLP classifier per layer. Outputs `training_results_v3.pkl` with per-layer predictions and probabilities.

### Pipeline 3: LayerSkip SFT

```bash
cd pipelines/layerskip_tuning
python train.py   # fine-tune with LayerSkip loss
python eval.py    # evaluate with early exit at eval_skip_layer
```

Fine-tunes the model with a combined loss across a rotating early-exit layer and the final layer. Evaluation runs greedy inference with early exit and writes accuracy to `metrics.txt`.

### Early Exit

```bash
cd early_exit
python prepare_data.py --input <pipeline_output.pkl> --output normalized.pkl
```

Normalizes the pkl output from any pipeline into the standard hierarchical format expected by the 2D early exit algorithm.
