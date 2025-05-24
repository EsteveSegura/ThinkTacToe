# train_qwen_tictactoe.py

import os
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# Login a Hugging Face
login(token=os.environ["HF_TOKEN"])

# Modelo base
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Asegurar que pad_token esté definido
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

model.gradient_checkpointing_enable()
model = torch.compile(model)

dataset = load_dataset("json", data_files="tictactoe_hf.json", split="train")

training_args = TrainingArguments(
    output_dir="./qwen2.5-1.5b-tictactoe",
    per_device_train_batch_size=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=250,
    save_total_limit=1,
    report_to="none",
    fp16=True,
    gradient_checkpointing=True
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=256,
    dataset_text_field="text"
)

trainer.train()
