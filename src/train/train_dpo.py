import torch
import multiprocessing
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from transformers import set_seed

set_seed(42)

# Modelo base
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = "<|endoftext|>"  # Ajusta según tu tokenizer
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Dataset local
dataset = load_dataset("json", data_files="tictactoe_dpo.json", split="train")
dataset = dataset.train_test_split(test_size=0.01)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Añadir <eos> si hace falta
def process(example):
    example["prompt"] = example["prompt"]
    example["chosen"] = example["chosen"] + tokenizer.eos_token
    example["rejected"] = example["rejected"] + tokenizer.eos_token
    return example

train_dataset = train_dataset.map(process, num_proc=multiprocessing.cpu_count(), load_from_cache_file=False)
eval_dataset = eval_dataset.map(process, num_proc=multiprocessing.cpu_count(), load_from_cache_file=False)

# Configuración del entrenamiento
training_args = DPOConfig(
    output_dir="./qwen2.5-1.5b-dpo",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    eval_strategy="steps",
    eval_steps=25,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    learning_rate=1e-6,
    num_train_epochs=1,
    max_length=1024,
    max_prompt_length=1024,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="paged_adamw_8bit",
    bf16=True,
    log_level="debug"
)

# Entrenador DPO
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Entrenamiento
trainer.train()
