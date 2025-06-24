#!/usr/bin/env python3
"""
Script de entrenamiento SFT para Tic-Tac-Toe usando el formato minimax.
Optimizado con special tokens y configuración específica para Qwen.
"""

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
    parser = argparse.ArgumentParser(description='Train a model using SFT with minimax format')
    parser.add_argument('--model_name', type=str, 
                      default='Qwen/Qwen2.5-0.5B',
                      help='Name or path of the base model to use')
    parser.add_argument('--data_files', type=str, 
                      default='./datasets/tictactor_sft_nothink_minmax.jsonl',
                      help='Path to the SFT dataset JSONL file')
    parser.add_argument('--output_dir', type=str, 
                      default='./qwen2.5-0.5b-tictactoe-sft-minimax',
                      help='Directory to save the trained model')
    parser.add_argument('--logs_dir', type=str, default='./logs',
                      help='Directory to save training logs')
    return parser.parse_args()

class LoggingCallback(TrainerCallback):
    """Callback personalizado para guardar logs de entrenamiento"""
    
    def __init__(self, logs_dir, model_type="sft_minimax"):
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
    
    print(f"🤖 Iniciando entrenamiento SFT con formato minimax...")
    print(f"📁 Modelo base: {args.model_name}")
    print(f"📊 Dataset: {args.data_files}")
    print(f"💾 Output dir: {args.output_dir}")
    
    # Login a Hugging Face (opcional)
    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
        print("✅ Login a Hugging Face completado")

    # 1. Cargar modelo base Qwen
    print(f"📥 Cargando modelo y tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✅ Modelo y tokenizer cargados")

    # 2. Añadir special tokens del tablero
    print(f"🔧 Añadiendo special tokens para formato minimax...")
    SPECIAL_TOKENS = [
        "<|board_start|>", "<|board_end|>", 
        "<|turn|>", "<|symbol|>",  # Corregido: turn y symbol en lugar de player
        "<|move|>", "<|end|>", "<|blank|>",
        "<|X|>", "<|O|>"  # Añadido: símbolos de jugadores
    ] + [f"<|{r}-{c}|>" for r in range(3) for c in range(3)]
    
    print(f"   - Tokens añadidos: {len(SPECIAL_TOKENS)}")
    print(f"   - Tokens del tablero: {SPECIAL_TOKENS[:7]}")
    print(f"   - Tokens de símbolos: {SPECIAL_TOKENS[7:9]}")
    print(f"   - Tokens de coordenadas: {len(SPECIAL_TOKENS[9:])} posiciones")
    
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    
    # Redimensionar embeddings (sin resize_to_multiple_of para compatibilidad)
    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_vocab_size = model.get_input_embeddings().weight.shape[0]
    
    print(f"   - Vocabulario original: {original_vocab_size}")
    print(f"   - Vocabulario nuevo: {new_vocab_size}")
    print(f"   - Tokens añadidos: {new_vocab_size - original_vocab_size}")

    # Configurar pad_token si no existe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   - Pad token configurado: {tokenizer.pad_token}")

    # Habilitar gradient checkpointing para ahorrar memoria
    model.gradient_checkpointing_enable()
    print("✅ Gradient checkpointing habilitado")

    # Cargar dataset
    print(f"📊 Cargando dataset...")
    dataset = load_dataset("json", data_files=args.data_files, split="train")
    print(f"✅ Dataset cargado: {len(dataset)} ejemplos")

    def formatting_func(example):
        """Función de formateo para el dataset"""
        return example["text"]

    # Crear callback para logging
    logging_callback = LoggingCallback(args.logs_dir, "sft_minimax")

    # 3. Configuración SFT optimizada
    print(f"⚙️ Configurando parámetros de entrenamiento...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=16,  # Reducido por los nuevos embeddings
        gradient_accumulation_steps=2,   # Para mantener batch efectivo
        num_train_epochs=1,
        learning_rate=2e-5,
        eos_token="<|im_end|>",          # Clave: alinear con plantilla de Qwen
        logging_steps=20,
        save_steps=250,
        save_total_limit=2,
        report_to="none",
        bf16=True,                       # Cambiado de fp16 a bf16 para evitar conflictos
        gradient_checkpointing=True,
        max_seq_length=256,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        # Configuraciones adicionales para estabilidad
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    print(f"📋 Configuración:")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   - Learning rate: {training_args.learning_rate}")
    print(f"   - EOS token: {training_args.eos_token}")
    print(f"   - Max seq length: {training_args.max_seq_length}")

    # 4. Trainer con tokenizer explícito
    print(f"🚀 Inicializando trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        callbacks=[logging_callback],
    )

    print("🎯 Comenzando entrenamiento...")
    trainer.train()

    print("✅ Entrenamiento completado!")
    print(f"💾 Modelo guardado en: {args.output_dir}")

if __name__ == "__main__":
    main() 