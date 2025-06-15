# Training Scripts Documentation

This document explains how to use the training scripts for both SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) training.

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- TRL
- Datasets
- Hugging Face Hub token (set as environment variable `HF_TOKEN`)

## SFT Training

The SFT training script (`train_sft.py`) is used for supervised fine-tuning of the base model.

### Usage

```bash
python src/train/train_sft.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "tictactoe_sft_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-sft"
```

### Parameters

- `--model_name`: Name or path of the base model to use (e.g., "Qwen/Qwen2.5-0.5B")
- `--data_files`: Path to the SFT dataset JSONL file
- `--output_dir`: Directory to save the trained model

### Example

```bash
# Train using template-based thoughts
python src/train/train_sft.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "tictactoe_sft_template.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-sft-template"

# Train using LLM-generated thoughts
python src/train/train_sft.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "tictactoe_sft_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-sft-llm"
```

## DPO Training

The DPO training script (`train_dpo.py`) is used for direct preference optimization, which requires a previously SFT-trained model.

### Usage

```bash
python src/train/train_dpo.py \
    --path "qwen2.5-0.5b-tictactoe-sft/checkpoint-942" \
    --data_files "tictactoe_dpo_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-dpo"
```

### Parameters

- `--path`: Path to the SFT checkpoint to use as base model
- `--data_files`: Path to the DPO dataset JSONL file
- `--output_dir`: Directory to save the trained model

### Example

```bash
# Train using template-based thoughts
python src/train/train_dpo.py \
    --path "qwen2.5-0.5b-tictactoe-sft-template/checkpoint-942" \
    --data_files "tictactoe_dpo_template.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-dpo-template"

# Train using LLM-generated thoughts
python src/train/train_dpo.py \
    --path "qwen2.5-0.5b-tictactoe-sft-llm/checkpoint-942" \
    --data_files "tictactoe_dpo_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-dpo-llm"
```

## Training Pipeline

A typical training pipeline would be:

1. Generate SFT dataset with desired thought generation method (template/LLM)
2. Train SFT model
3. Generate DPO dataset with same thought generation method
4. Train DPO model using the SFT checkpoint

Example pipeline:

```bash
# 1. Generate datasets
python src/dataset/dataset_generator_sft.py  # Generates tictactoe_sft_llm.jsonl
python src/dataset/dataset_generator_dpo.py  # Generates tictactoe_dpo_llm.jsonl

# 2. Train SFT
python src/train/train_sft.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "tictactoe_sft_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-sft-llm"

# 3. Train DPO
python src/train/train_dpo.py \
    --path "qwen2.5-0.5b-tictactoe-sft-llm/checkpoint-942" \
    --data_files "tictactoe_dpo_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-dpo-llm"
``` 