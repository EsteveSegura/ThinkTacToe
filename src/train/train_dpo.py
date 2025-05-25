import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

# Cargar el dataset
dataset = load_dataset('json', data_files='tictactoe_dpo.json', split='train')

# Configuración del modelo
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# Configuración de DPO
config = DPOConfig(
    beta=0.01,
    learning_rate=1e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    max_prompt_length=512,
    max_length=1024,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    output_dir="./qwen2.5-1.5b-dpo",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True
)

# Inicializar el entrenador DPO
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Si no se proporciona, se utiliza una copia del modelo actual
    args=config,
    tokenizer=tokenizer,
    train_dataset=dataset
)

# Iniciar el entrenamiento
trainer.train()
