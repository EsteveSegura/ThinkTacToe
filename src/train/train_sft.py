import os
import argparse
import torch
import json
from datetime import datetime
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
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
    parser.add_argument('--logs_dir', type=str, default='./logs',
                      help='Directory to save training logs')
    return parser.parse_args()

class LoggingCallback(TrainerCallback):
    """Callback personalizado para guardar logs de entrenamiento"""
    
    def __init__(self, logs_dir, model_type="sft"):
        super().__init__()
        self.logs_dir = logs_dir
        self.model_type = model_type
        os.makedirs(logs_dir, exist_ok=True)
        
        # Crear archivo de logs con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(logs_dir, f"{model_type}_training_logs_{timestamp}.txt")
        
        # Escribir header del archivo
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {model_type.upper()} Training Logs ===\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Método llamado cuando se generan logs durante el entrenamiento"""
        if logs is not None:
            # Guardar logs en archivo
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"Step {state.global_step}: {json.dumps(logs, indent=2)}\n")
                f.write("-" * 30 + "\n")
            
            # También imprimir en consola
            print(f"Step {state.global_step}: {logs}")

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

    # Crear callback para logging
    logging_callback = LoggingCallback(args.logs_dir, "sft")

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
        callbacks=[logging_callback],
    )

    trainer.train()

if __name__ == "__main__":
    main()
