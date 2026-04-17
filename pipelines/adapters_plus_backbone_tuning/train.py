import os
import gc
import json
import random

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from model import CascadeDecoderModule
from utils import load_dataset
from config import dataset_config, training_config, model_ckpt, seed

random.seed(seed)


def checkpoint_dir(dataset_config):
    checkpoint_path = os.path.join(f'{dataset_config.dataset_name.replace("/", "_")}', model_ckpt.replace('/', '_'))
    os.makedirs(checkpoint_path, exist_ok=True)
    return checkpoint_path


def train():
    logging_path = checkpoint_dir(dataset_config)
    print('logging path', logging_path)
    dataset = load_dataset(dataset_config)
    print(dataset)
    label2idx = {l: i for i, l in enumerate(dataset['train'].features['label'].names)}
    print(f'Dataset config: {dataset_config}')
    print(f'Training config: {training_config}')
    print(f'Labels: {label2idx}')
    
    decoder_model = CascadeDecoderModule(model_ckpt, label2idx, training_config,                                        
                                         n_labels=dataset_config.num_labels)
    
    if dataset_config.text_column_name != "text":
        dataset = dataset.rename_column(dataset_config.text_column_name, "text")

    dataset = dataset.map(lambda x: decoder_model.encode_sentences(x, training_config.max_length), batched=False)

    decoder_model.fit(dataset, logging_path)

    pred_df, metadata_dict = decoder_model.evaluate_dataset(dataset['test'], output_dir=logging_path, save_embeddings=True)
    pred_df.to_pickle(os.path.join(logging_path, 'test_preds.pkl'))
    metadata_dict = json.dumps(metadata_dict)
    with open(os.path.join(logging_path, 'metadata.json', 'w')) as f:
        f.write(metadata_dict)
    
    del decoder_model
    gc.collect()


if __name__ == '__main__':
    train()
