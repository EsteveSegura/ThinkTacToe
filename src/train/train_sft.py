import os
import argparse
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model using SFT')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name or path of the base model to use')
    parser.add_argument('--data_files', type=str, required=True,
                      help='Path to the SFT dataset JSONL file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the trained model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Login a Hugging Face
    login(token=os.environ["HF_TOKEN"])

    # Modelo base
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()

    dataset = load_dataset("json", data_files=args.data_files, split="train")

    def formatting_func(example):
        return example["text"]

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=250,
        save_total_limit=1,
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
        max_seq_length=256,
        eos_token="<|im_end|>",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
    )

    trainer.train()

if __name__ == "__main__":
    main()
