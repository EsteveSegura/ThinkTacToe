from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset

model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Tu dataset con campos: prompt, chosen, rejected
dataset = load_dataset("json", data_files="tictactoe_dpo.json", split="train")

# Tokenizar
def preprocess(example):
    return {
        "prompt_input_ids": tokenizer(example["prompt"], truncation=True, max_length=256)["input_ids"],
        "chosen_input_ids": tokenizer(example["chosen"], truncation=True, max_length=128)["input_ids"],
        "rejected_input_ids": tokenizer(example["rejected"], truncation=True, max_length=128)["input_ids"],
    }

tokenized_dataset = dataset.map(preprocess)

# Entrenamiento
training_args = TrainingArguments(
    output_dir="./qwen2.5-1.5b-dpo",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-6,
    fp16=True,
    logging_steps=10,
    save_steps=250,
    report_to="none"
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    beta=0.1
)

trainer.train()
