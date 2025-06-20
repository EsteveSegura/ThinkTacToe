import argparse
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model using DPO')
    parser.add_argument('--path', type=str, required=True,
                      help='Path to the SFT checkpoint to use as base model')
    parser.add_argument('--data_files', type=str, required=True,
                      help='Path to the DPO dataset JSONL file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the trained model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(args.path)
    tokenizer = AutoTokenizer.from_pretrained(args.path)

    train_dataset = load_dataset("json", data_files=args.data_files, split="train")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        logging_steps=10
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset
        num_train_epochs=3
    )

    trainer.train()

if __name__ == "__main__":
    main()