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
├── fine_tuning/          # Adapter fine-tuning trained jointly with the backbone (optional LoRA via PEFT)
│   ├── config.py         # Model, dataset, training hyperparameters — update before running
│   ├── train.py          # Fine-tuning entry point
│   ├── model.py          # Adapter module definition
│   └── utils.py          # Data loading and preprocessing helpers
│
└── layerskip/            # LayerSkip-based SFT + early-exit evaluation
    ├── config.py         # Model, dataset, and eval hyperparameters — update before running
    ├── train.py          # SFT training entry point (LayerSkip-aware trainer)
    ├── eval.py           # Early-exit evaluation and accuracy reporting
    └── custom_trainer.py # LayerSkipSFTTrainer with loss
```

---

## Configuration

Before running any script, update the relevant `config.py`:

**`fine_tuning/config.py`**
- `model_ckpt` — HuggingFace model ID (e.g. `meta-llama/Llama-3.1-8B-Instruct`)
- `dataset_name` — HuggingFace dataset ID
- `dataset_config` — set `text_column_name`, `label_column_name`, `num_labels`, and `label_names` to match your dataset

**`layerskip/config.py`**
- `model_name` / `tokenizer_name` — HuggingFace model ID
- `dataset_name` — HuggingFace dataset ID
- `eval_skip_layer` — controls which intermediate layer is used to make predictions during evaluation (instead of the final layer); accuracy is computed from this layer's output
- `idx2label` / `label2idx` — update to match your dataset's class labels

---

## Running

### Fine-tuning (adapter-based)

```bash
cd fine_tuning
python train.py
```

Trains adapters jointly with the backbone. LoRA (PEFT) can be enabled via `peft_training` in `TrainingConfig`. Checkpoints are saved to the path defined by `TrainingConfig.output_dir` in `config.py`.

### LayerSkip SFT Training

```bash
cd layerskip
python train.py
```

Fine-tunes the full model using `LayerSkipSFTTrainer`, which applies a combined loss from a rotating early-exit layer and the final layer at each step. The trained model is saved locally using the model and dataset name.

### LayerSkip Evaluation

```bash
cd layerskip
python eval.py
```

Loads the fine-tuned checkpoint and runs greedy inference with early exit at `eval_skip_layer`. Reports accuracy and writes results to `metrics.txt`.
