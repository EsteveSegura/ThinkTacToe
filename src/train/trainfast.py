from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset
from huggingface_hub import login
import os

# Autenticación
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Modelo base
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Dataset
dataset = load_dataset("json", data_files="tictactoe_hf.json", split="train")

# Entrenamiento rápido
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=512,
    dataset_text_field="text",
    args=dict(
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=5e-5,
        output_dir="./qwen-tictactoe-test",
        logging_steps=10,
        save_strategy="epoch",         # Solo guarda al final del epoch
        fp16=True,                     # Usa float16 si tu GPU lo soporta
        gradient_accumulation_steps=2, # Simula batch más grande
        remove_unused_columns=False
    ),
)

trainer.train()