from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
import json

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

from datasets import Dataset

# Cargar datos desde el archivo JSON
with open('./tictactoe_dpo.json', 'r') as f:
    samples = json.load(f)

dataset = Dataset.from_list(samples)

training_args = DPOConfig(output_dir="Qwen2.5-1.5B-DPO")
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

trainer.train()