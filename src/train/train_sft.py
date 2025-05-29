import os
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Login a Hugging Face
login(token=os.environ["HF_TOKEN"])

# Modelo base
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

model.gradient_checkpointing_enable()

dataset = load_dataset("json", data_files="tictactoe_dataset_sft.jsonl", split="train")

def formatting_func(example):
    return example["text"]

training_args = SFTConfig(
    output_dir="./qwen2.5-0.5b-tictactoe-sft",
    per_device_train_batch_size=32,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=250,
    save_total_limit=1,
    report_to="none",
    fp16=True,
    gradient_checkpointing=True,
    max_seq_length=256,
    eos_token="<|im_end|>",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_func,
)

trainer.train()
