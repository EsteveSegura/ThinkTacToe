# Training Scripts Documentation

This document explains how to use the training scripts for SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), and GRPO (Generative Reward-Powered Optimization) training.

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
    --output_dir "qwen2.5-0.5b-tictactoe-sft" \
    --logs_dir "./logs"
```

### Parameters

- `--model_name`: Name or path of the base model to use (e.g., "Qwen/Qwen2.5-0.5B")
- `--data_files`: Path to the SFT dataset JSONL file
- `--output_dir`: Directory to save the trained model
- `--logs_dir`: Directory to save training logs (default: "./logs")

### Example

```bash
# Train using template-based thoughts
python src/train/train_sft.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "tictactoe_sft_template.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-sft-template" \
    --logs_dir "./logs"

# Train using LLM-generated thoughts
python src/train/train_sft.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "tictactoe_sft_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-sft-llm" \
    --logs_dir "./logs"
```

## DPO Training

The DPO training script (`train_dpo.py`) is used for direct preference optimization, which requires a previously SFT-trained model.

### Usage

```bash
python src/train/train_dpo.py \
    --path "qwen2.5-0.5b-tictactoe-sft/checkpoint-942" \
    --data_files "tictactoe_dpo_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-dpo" \
    --logs_dir "./logs"
```

### Parameters

- `--path`: Path to the SFT checkpoint to use as base model
- `--data_files`: Path to the DPO dataset JSONL file
- `--output_dir`: Directory to save the trained model
- `--logs_dir`: Directory to save training logs (default: "./logs")

### Example

```bash
# Train using template-based thoughts
python src/train/train_dpo.py \
    --path "qwen2.5-0.5b-tictactoe-sft-template/checkpoint-942" \
    --data_files "tictactoe_dpo_template.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-dpo-template" \
    --logs_dir "./logs"

# Train using LLM-generated thoughts
python src/train/train_dpo.py \
    --path "qwen2.5-0.5b-tictactoe-sft-llm/checkpoint-942" \
    --data_files "tictactoe_dpo_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-dpo-llm" \
    --logs_dir "./logs"
```

## GRPO Training

The GRPO training script (`train_grpo.py`) is used for Generative Reward-Powered Optimization, which uses a reward function to evaluate and improve model performance directly.

### Usage

```bash
python src/train/train_grpo.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "tictactoe_grpo_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-grpo" \
    --logs_dir "./logs"
```

### Parameters

- `--model_name`: Name or path of the base model to use (e.g., "Qwen/Qwen2.5-0.5B")
- `--data_files`: Path to the GRPO dataset JSONL file
- `--output_dir`: Directory to save the trained model
- `--logs_dir`: Directory to save training logs (default: "./logs")

### Example

```bash
# Train using LLM-generated thoughts
python src/train/train_grpo.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "tictactoe_grpo_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-grpo-llm" \
    --logs_dir "./logs"
```

### Reward Function

The GRPO training uses a sophisticated reward function that evaluates move quality based on:

- **Winning moves**: +1.0 (immediate victory)
- **Blocking moves**: +0.8 (prevents opponent from winning)
- **Center moves**: +0.6 (strategic position)
- **Corner moves**: +0.4 (good strategic positions)
- **Edge moves**: +0.2 (neutral positions)
- **Invalid moves**: -1.0 (penalty for invalid moves)

The reward function parses the board state from the prompt and evaluates each generated move in context.

## Training Logs

All training scripts now include automatic logging functionality that saves training progress to text files.

### Log Files

Training logs are automatically saved to the specified `--logs_dir` directory with the following naming convention:
- SFT logs: `sft_training_logs_YYYYMMDD_HHMMSS.txt`
- DPO logs: `dpo_training_logs_YYYYMMDD_HHMMSS.txt`
- GRPO logs: `grpo_training_logs_YYYYMMDD_HHMMSS.txt`

### Log Content

Each log file contains:
- Training start timestamp
- Step-by-step training metrics including:
  - Loss values
  - Gradient norms
  - Learning rates
  - Token accuracy
  - Epoch progress
  - Number of tokens processed
  - Reward values (for GRPO)

### Example Log Output

```
=== GRPO Training Logs ===
Started at: 2024-01-15 14:30:25
==================================================

Step 10: {
  "loss": 0.0612,
  "grad_norm": 1.2164300680160522,
  "learning_rate": 1.4814814814814815e-06,
  "num_tokens": 733184.0,
  "mean_token_accuracy": 0.972096461057663,
  "epoch": 2.86,
  "reward_mean": 0.75,
  "reward_std": 0.25
}
------------------------------
```

## Training Pipeline

A typical training pipeline would be:

1. Generate SFT dataset with desired thought generation method (template/LLM)
2. Train SFT model
3. Generate DPO dataset with same thought generation method
4. Train DPO model using the SFT checkpoint
5. (Optional) Generate GRPO dataset and train GRPO model

Example pipeline:

```bash
# 1. Generate datasets
python src/dataset/dataset_generator_sft.py  # Generates tictactoe_sft_llm.jsonl
python src/dataset/dataset_generator_dpo.py  # Generates tictactoe_dpo_llm.jsonl
python src/dataset/dataset_generator_grpo.py # Generates tictactoe_grpo_llm.jsonl

# 2. Train SFT with logging
python src/train/train_sft.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "tictactoe_sft_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-sft-llm" \
    --logs_dir "./logs"

# 3. Train DPO with logging
python src/train/train_dpo.py \
    --path "qwen2.5-0.5b-tictactoe-sft-llm/checkpoint-942" \
    --data_files "tictactoe_dpo_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-dpo-llm" \
    --logs_dir "./logs"

# 4. Train GRPO with logging
python src/train/train_grpo.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "tictactoe_grpo_llm.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-grpo-llm" \
    --logs_dir "./logs"
```

## Monitoring Training Progress

During training, you can monitor progress in several ways:

1. **Console output**: Real-time logs are printed to the console
2. **Log files**: Detailed logs are saved to text files in the specified logs directory
3. **Model checkpoints**: Models are saved at regular intervals in the output directory

The logging system ensures that all training metrics are preserved for later analysis and debugging. 






accelerate launch src/train/train_sft.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "./datasets/tictactoe_sft_nothink.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-sft-nothink" \
    --logs_dir "./logs"


accelerate launch src/train/train_sft.py \
    --model_name "Qwen/Qwen2.5-3B" \
    --data_files "./datasets/tictactoe_minimax_20250629_022203.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-sft-nothink-minmax" \
    --logs_dir "./logs"



accelerate launch src/train/train_grpo_minimax.py \
    --model_name "qwen2.5-0.5b-tictactoe-sft-nothink-minmax/checkpoint-26" \
    --data_files "./datasets/tictactoe_grpo_from_minimax_20250624_175237.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-grpo-llm" \
    --logs_dir "./logs"



accelerate launch src/train/train_sft_minimax_fixed.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "./datasets/tictactoe_minimax_20250624_214328.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-sft-minimax-fixed" \
    --logs_dir "./logs" \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 5e-5