"""
Model-specific configurations for training classifiers on embeddings.

Each model has its own preprocessing steps and optimized hyperparameters.

Refactored from: new_pipeline/model_configs.py
"""

MODEL_CONFIGS = {
    "llama_3_1_8b": {
        "display_name": "Llama 3.1-8B Instruct",
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "output_suffix": "_results",

        # Preprocessing
        "squeeze_extra_dim": False,
        "squeeze_axis": None,

        # Training hyperparameters
        "epochs": 60,
        "learning_rate": 1e-5,
        "batch_size": 128,
        "hidden_dim": 256,
        "dropout": 0.0,

        # Optimizer settings
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "scheduler": "reduce_plateau",
        "scheduler_params": {
            "mode": "min",
            "patience": 5,
            "factor": 0.5
        },

        # Expected data format
        "expected_layers": 32,
        "expected_embed_dim": 4096,

        "notes": "Standard transformer embeddings, no preprocessing needed"
    },

    "llama_3_2_3b": {
        "display_name": "Llama 3.2-3B Instruct",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "output_suffix": "_results_v2",

        "squeeze_extra_dim": False,
        "squeeze_axis": None,

        "epochs": 50,
        "learning_rate": 3e-4,
        "batch_size": 128,
        "hidden_dim": 256,
        "dropout": 0.15,

        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "scheduler": "reduce_plateau",
        "scheduler_params": {
            "mode": "min",
            "patience": 4,
            "factor": 0.6
        },

        "expected_layers": 28,
        "expected_embed_dim": 3072,

        "notes": "Smaller model, slightly more aggressive training"
    },

    "qwen": {
        "display_name": "Qwen 2.5-3B Instruct",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "output_suffix": "_results",

        "squeeze_extra_dim": False,
        "squeeze_axis": None,

        "epochs": 55,
        "learning_rate": 2.5e-4,
        "batch_size": 128,
        "hidden_dim": 256,
        "dropout": 0.12,

        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "scheduler": "reduce_plateau",
        "scheduler_params": {
            "mode": "min",
            "patience": 5,
            "factor": 0.5
        },

        "expected_layers": 36,
        "expected_embed_dim": 2048,

        "notes": "Qwen architecture, standard preprocessing"
    },

    "gemma": {
        "display_name": "Gemma 3N-E4B-it",
        "model_id": "google/gemma-3n-E4B-it",
        "output_suffix": "_results_v3",

        # Gemma specific preprocessing
        "squeeze_extra_dim": True,
        "squeeze_axis": 2,
        "ndim_check": True,
        "shape_check": (1,),

        "epochs": 50,
        "learning_rate": 1e-4,
        "batch_size": 64,
        "hidden_dim": 256,
        "dropout": 0.2,

        "optimizer": "adamw",
        "weight_decay": 2e-4,
        "scheduler": "reduce_plateau",
        "scheduler_params": {
            "mode": "min",
            "patience": 6,
            "factor": 0.4
        },

        "expected_layers": 36,
        "expected_embed_dim": 3072,

        "notes": "Experimental Gemma model, needs special preprocessing for extra dimension"
    }
}

DEFAULT_CONFIG = {
    "display_name": "Unknown Model",
    "model_id": None,
    "output_suffix": "_results",

    "squeeze_extra_dim": False,
    "squeeze_axis": None,

    "epochs": 40,
    "learning_rate": 1e-4,
    "batch_size": 128,
    "hidden_dim": 256,
    "dropout": 0.1,

    "optimizer": "adamw",
    "weight_decay": 1e-4,
    "scheduler": "reduce_plateau",
    "scheduler_params": {
        "mode": "min",
        "patience": 5,
        "factor": 0.5
    },

    "expected_layers": None,
    "expected_embed_dim": None,

    "notes": "Default configuration for unknown models"
}


def get_model_config(model_name):
    """Returns configuration for a given model name."""
    config = MODEL_CONFIGS.get(model_name, DEFAULT_CONFIG.copy())
    config["model_name"] = model_name
    return config


def list_available_models():
    """Prints available models and their configurations."""
    print("Available models:")
    print("=" * 50)
    for model_name, config in MODEL_CONFIGS.items():
        print(f"{model_name:15} - {config['display_name']}")
        print(f"{'':15}   Epochs: {config['epochs']}, LR: {config['learning_rate']}")
        print(f"{'':15}   {config['notes']}")
        print()


if __name__ == "__main__":
    list_available_models()
