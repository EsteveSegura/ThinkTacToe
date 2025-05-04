import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Autenticación con Hugging Face
login(token=os.environ["HF_TOKEN"])

# Modelo base
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Cargar dataset
dataset = load_dataset("json", data_files="tictactoe_hf.json", split="train")

# Configuración de entrenamiento rápido
training_args = TrainingArguments(
    output_dir="./qwen2.5-1.5b-tictactoe",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,
    report_to="none"
)

# Entrenador
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=512,
    dataset_text_field="text"
)

trainer.train()
