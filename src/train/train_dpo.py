import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer

model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

dataset = load_dataset("json", data_files="tictactoe_dpo.json", split="train")

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

trainer = DPOTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset,
    beta=0.1,
    dpo_kwargs={
        "max_prompt_length": 256,
        "max_response_length": 128,
    }
)

trainer.train()
