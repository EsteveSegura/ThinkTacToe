from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

path = "./qwen2.5-0.5b-tictactoe-sft/checkpoint-942"
model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

train_dataset = load_dataset("json", data_files="./tictactoe_dpo.jsonl", split="train")

training_args = DPOConfig(output_dir="qwen2.5-0.5b-tictactoe-dpo", logging_steps=10)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()