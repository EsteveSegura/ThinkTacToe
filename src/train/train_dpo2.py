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
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,  # Activar entrenamiento en precisión mixta
    report_to="none"
)

# Inicializar el entrenador
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Entrenamiento
trainer.train()
