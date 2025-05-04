from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset

model_name = "Qwen/Qwen-1_5B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

dataset = load_dataset("json", data_files="tictactoe_hf.json", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length=512,
    args=dict(
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=2e-5,
        output_dir="./qwen1.5b-tictactoe",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False
    ),
)

trainer.train()
