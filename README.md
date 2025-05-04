# Think Tac Toe

## Introduction

**TicTacLearn** is an AI project focused on training a language model (LLM) to **play Tic Tac Toe** through step-by-step reasoning using **supervised fine-tuning (SFT)**.

By leveraging tokenized board representations, structured reasoning, and systematic evaluation, the goal is not only to produce valid moves, but also to **explain strategic decisions in natural language**.

---

## What does this project do?

### ✅ Main objectives

* 🧱 **Design a structured input/output format** for the game:

  * Board representation using custom tokens.
  * Reasoning steps wrapped in `<think>...</think>` before each move.
  * Moves expressed precisely as `<|move|><|row-col|>`.

* 🧠 **Generate a high-quality SFT dataset**:

  * Complete games (wins, losses, draws).
  * Moves labeled as `win`, `block`, or `neutral`.
  * Include recoverable mistakes to improve robustness.

* 🧪 **Fine-tune a base model (Qwen-1.5B)** to:

  * Learn fundamental strategies.
  * Verbally express its reasoning before acting.
  * Maintain logical consistency across full games.

* 📊 **Evaluate using a custom benchmark**:

  * `TicTacBench`: a test suite with varying difficulty levels.
  * Metrics: win/draw/loss rate, reasoning quality, and move consistency.

---

## Board representation

Input example:

```
<|0-0|><|X|> <|0-1|><|O|> <|0-2|><|blank|>
<|1-0|><|O|> <|1-1|><|X|> <|1-2|><|blank|>
<|2-0|><|blank|> <|2-1|><|blank|> <|2-2|><|blank|>
<think>I detect a winning opportunity: I place at 0-2 and complete the line.</think>
```

Output example:

```
<|move|><|0-2|><|end|>
```

---

## Project structure

* `engine.py` → game engine for rule validation and turn tracking.
* `representation.py` → converts a board into token format like `<|0-0|><|X|>`.
* `reasoning.py` → generates `<think>...</think>` reasoning texts for each situation.
* `dataset_generator.py` → full generator for SFT training examples.
* `benchmark/` → evaluation scripts for post-training performance tests.

---

## Current status

✅ Game engine implemented
✅ Board tokenizer implemented
✅ Templates for `win`, `block`, `neutral` reasoning
🔜 Full dataset generator
🔜 Fine-tuning with Qwen-1.5B
🔜 Evaluation with TicTacBench benchmark

---

## Why this project?

This is an exploration of:

* Teaching *visual and strategic reasoning* to LLMs.
* Understanding how well an LLM can learn simple game logic.
* Laying the groundwork to scale toward games like **Connect 4**, **Reversi**, or even **complex board games**.
