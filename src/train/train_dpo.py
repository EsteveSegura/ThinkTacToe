from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset

# Cargar modelo y tokenizer
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Cargar conjunto de datos
dataset = load_dataset("json", data_files="tictactoe_dpo.json", split="train")

# Configuración de entrenamiento (usa TrainingArguments, no DPOConfig)
training_args = TrainingArguments(
    output_dir="./qwen2.5-1.5b-dpo",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-6,
    fp16=True,
    logging_steps=10,
    save_steps=250,
    save_total_limit=1,
    report_to="none"
)

# Instanciar DPOTrainer sin DPOConfig
trainer = DPOTrainer(
    model=model,
    args=training_args,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_prompt_length=256,
    max_length=384
)

# Iniciar entrenamiento
trainer.train()
