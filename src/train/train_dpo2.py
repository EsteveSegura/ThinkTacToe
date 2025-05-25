from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
import json

# Cargar modelo y tokenizer
model_name = "Qwen/Qwen2.5-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Cargar datos desde archivo JSON
with open('./tictactoe_dpo.json', 'r') as f:
    samples = json.load(f)

dataset = Dataset.from_list(samples)

# Configuración de entrenamiento
training_args = DPOConfig(
    output_dir="Qwen2.5-1.5B-DPO",
    per_device_train_batch_size=8,                 # Tamaño razonable para GPUs grandes
    gradient_accumulation_steps=2,                 # Aumenta batch efectivo sin usar más memoria
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    save_total_limit=1,
    fp16=True,                                     # Usa mixed precision si tienes GPU compatible
    bf16=False,                                    # Cambia a True si tu GPU lo soporta (como A100/H100)
    dataloader_num_workers=4,                      # Preprocesamiento más rápido
    remove_unused_columns=True,
    push_to_hub=False,
    max_length=1024,                               # Limita la secuencia si es razonable
    max_prompt_length=512,
    logging_dir="./logs",
    report_to="none",                              # Evita overhead innecesario
    gradient_checkpointing=True,                   # Menos memoria, pero más lento por paso
    truncation_mode="keep_end",                    # Útil si tus muestras son largas
    disable_dropout=True                           # Determinismo y rendimiento en inferencia
)

# Inicializar el entrenador
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

# Entrenamiento
trainer.train()
