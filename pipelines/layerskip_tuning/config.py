from huggingface_hub import whoami

model_name = 'meta-llama/Llama-3.1-8B-Instruct' 
dataset_name = "davidadamczyk/Amazon_reviews_5cl-v2" 

output_root_dir = "./checkpoints/"
hub_model_id = f"{whoami()['name']}/layerskip-{model_name.split('/')[1]}-{dataset_name.split('/')[1]}"
output_dir = f"{output_root_dir}/{hub_model_id}"

per_device_train_batch_size = 1
gradient_accumulation_steps = 8
learning_rate    = 2e-5
num_train_epochs = 2.0
peft_training    = False
bf16_training    = True
max_length       = 768
eval_skip_layer  = 15
save_total_limit = 2

idx2label = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
label2idx = {v: k for k, v in idx2label.items()}