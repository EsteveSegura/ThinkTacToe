from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

path = "./models/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

with open('./tictactoe_dpo.json', 'r') as f:
    samples = json.load(f)

dataset = Dataset.from_list(samples)

training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO-TicTacToe", logging_steps=10)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()