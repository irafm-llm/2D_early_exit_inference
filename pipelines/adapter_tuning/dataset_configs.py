"""
Dataset configurations - prompt templates, label mappings, and optimal thresholds.

Refactored from: new_pipeline/main_config.py
"""

DATASET_CONFIGS = {
    "amazon_reviews": {
        "dataset_name": "davidadamczyk/Amazon_reviews_5cl-v2",
        "prompt_template": """
Classify the sentiment of the following text as Very Negative, Negative, Neutral or Positive based solely on its tone, context, and language.

Text:
{text}

Output:
""",
        "label_config": {
            "source_column": "class_index",
            "source_values": [1, 2, 3, 4, 5],
            "target_values": [0, 1, 2, 3, 4],
            "names": ["very_negative", "negative", "neutral", "positive", "very_positive"],
            "skip_labels": [],
            "primary_labels": [0, 1, 2, 3, 4]
        },
        "optimal_thresholds": {
            "ignore_threshold": 0.1,
            "acc_threshold": 1.8
        }
    },
    "two_class_reviews": {
        "dataset_name": "davidadamczyk/two_class_reviews",
        "prompt_template": """
Classify the sentiment of the following text as Negative or Positive, based solely on its tone, context, and language.

Text:
{text}

Output:
""",
        "label_config": {
            "source_column": "label",
            "source_values": [0, 1],
            "target_values": [0, 1],
            "names": ["negative", "positive"],
            "skip_labels": [],
            "primary_labels": [0, 1]
        },
        "optimal_thresholds": {
            "ignore_threshold": 0.1,
            "acc_threshold": 1.8
        }
    },
    "mms_subset": {
        "dataset_name": "davidadamczyk/mms_subset",
        "prompt_template": """
Classify the sentiment of the following text as Negative, Neutral, or Positive, based solely on its tone, context, and language.

Text:
{text}

Output:
""",
        "label_config": {
            "source_column": "label",
            "source_values": [0, 1, 2],
            "target_values": [0, 1, 2],
            "names": ["negative", "neutral", "positive"],
            "skip_labels": [1],
            "primary_labels": [0, 2]
        },
        "optimal_thresholds": {
            "ignore_threshold": 0.3,
            "acc_threshold": 2.1
        }
    },
}
