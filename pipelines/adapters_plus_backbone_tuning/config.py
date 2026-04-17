from pathlib import Path
from dataclasses import dataclass


seed            = 42
data_dir        = 'data'
n_epochs        = 2
#'google/gemma-3n-E4B-it' 
#'meta-llama/Llama-3.1-8B-Instruct'
#'meta-llama/Llama-3.2-3B-Instruct'
model_ckpt      = 'meta-llama/Llama-3.1-8B-Instruct'
logging_dir     = Path('runs', model_ckpt.replace('/', '_') + f'_epochs-{n_epochs}')
dataset_name    = 'davidadamczyk/mms_subset' 


@dataclass
class DatasetConfig:
    text_column_name: str
    label_column_name: str
    dataset_name: str
    subset: str=None
    shuffle: bool=False
    n_training_samples: int=None               # number of training samples
    validation_split: bool=False
    label_names: list=None                     # cast int label column into ClassLabel feature
    data_dir: str=None
    num_labels: int=3


@dataclass
class TrainingConfig:
    max_length: int = 768
    output_dir: str = f'{dataset_name.split("/")[-1]}-{model_ckpt}-r64'
    train_batch_size: int = 1
    test_batch_size: int = 4
    lr: float = 3e-5
    adapter_dropout: float = 0.20
    logging_steps: int = 200
    gradient_accumulation_steps: int = 16
    n_epochs: int = n_epochs
    train_steps: int = None 
    adapter_hidden_size: int = 200  # intermediate projection size inside each AdapterModule
    adapter_norm: str = None
    validation_steps: int  = None 
    peft_training: bool = True  # use LoRA for backbone; set False for full fine-tuning (requires FP32/BF16)



dataset_config = DatasetConfig('text', 'label', dataset_name, num_labels=3,
                               validation_split=False, label_names=['0', '1', '2'], shuffle=True)



training_config = TrainingConfig(max_length=768, output_dir=f'{dataset_config.dataset_name.split("/")[-1]}-{model_ckpt}-r64', train_batch_size=1, gradient_accumulation_steps=16,
                                 test_batch_size=1, lr=1e-4, adapter_dropout=0.2)




