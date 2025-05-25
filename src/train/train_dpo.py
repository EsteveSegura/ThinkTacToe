import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

# Modelo base
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Asegurar que el token de padding esté definido

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Dataset
dataset = load_dataset("json", data_files="tictactoe_dpo.json", split="train")

# Configuración
training_args = DPOConfig(
    output_dir="./qwen2.5-1.5b-dpo",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-6,
    fp16=True,
    logging_steps=10,
    save_steps=250,
    save_total_limit=1,
    report_to="none",
    beta=0.1,
    max_prompt_length=256,
    max_length=384
)

# Entrenador
trainer = DPOTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset
)

trainer.train()
