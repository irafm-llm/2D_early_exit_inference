import random
import torch
import torch.nn as nn
import datasets

random.seed(42)


def load_dataset(config):
    dataset = datasets.load_dataset(config.dataset_name, config.subset)

    if config.n_training_samples:
        random_indices = random.sample(range(len(dataset['train'])), config.n_training_samples)
        dataset['train'] = dataset['train'].select(random_indices)

    if config.label_column_name != 'label':
        for subset in dataset:
            dataset[subset] = dataset[subset].rename_column(config.label_column_name, "label")

    if config.label_names is not None:
        # if label idx starts with 1
        if min(dataset['train']['label']) == 1:
            dataset = dataset.map(lambda x: {'label': x['label'] - 1})

        # cast int column into ClassLabel column required during training
        num_classes = len(config.label_names)
        label_feature = datasets.ClassLabel(num_classes = num_classes, names=config.label_names)
        dataset = dataset.cast_column("label", label_feature)
  
    if config.shuffle:
        dataset['train'] = dataset['train'].shuffle(seed=42)

    if config.validation_split:
        # split training subset into train and validation
        training_subset = dataset['train']
        training_subset = training_subset.train_test_split(test_size=0.1, stratify_by_column="label", seed=42)
        dataset['train'] = training_subset['train']
        dataset['validation'] = training_subset['test']
        
    return dataset


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, device, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)).to(device)
        self.variance_epsilon = eps

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def calculate_gradient_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm