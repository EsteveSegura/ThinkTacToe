import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


login(token=os.environ["HF_TOKEN"])

model_name = "Qwen/Qwen2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

dataset = load_dataset("json", data_files="tictactoe_hf.json", split="train")

training_args = TrainingArguments(
    output_dir="./qwen2.5-1.5b-tictactoe",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    learning_rate=2e-5,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=512
)

trainer.train()
