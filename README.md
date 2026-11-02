# Tool-RL: Function Calling for Compact Language Models

## Overview
A pipeline for training small, open-source LLMs (1B–7B parameters) to reliably generate tool calls with correct function selection, argument validation, and JSON compliance using supervised fine-tuning and reinforcement learning.

## Key Features
- **Data Pipeline:** Expands APIGen-MT-5k (5k conversations) into one example per tool call, with the conversation history before it as the prompt
- **SFT Training:** Hugging Face Trainer support for Phi-2 (2.7B) and Mistral-7B with mixed-precision and gradient accumulation
- **RL Fine-Tuning:** GRPO (trl) with LoRA adapters, rewarding tool selection, schema validity, and argument correctness
- **Evaluation:** Greedy-decoded metrics for JSON validity, tool-selection accuracy, exact match, argument accuracy, and mean reward

## Quick Start
```bash
# Setup
pip install -r requirements.txt
python src/data.py --splits train --val-ratio 0.05

# Train
python src/train_sft.py --config experiments/config.yaml

# RL refinement (starts from the SFT checkpoint in experiments/runs/<run_name>/final)
python src/train_rl.py --config experiments/config.yaml

# Evaluate a model, SFT checkpoint, or RL adapter on the val split
python src/eval.py --config experiments/config.yaml --model experiments/runs/debug_rl/final
```

See [docs/TRAINING.md](docs/TRAINING.md) for the full training guide, configuration, and troubleshooting.

## Repo Layout
```
src/            data.py, prompts.py, rewards.py, train_sft.py, train_rl.py, eval.py
experiments/    configs (local Phi-2, Colab Mistral) and test_setup.py
notebooks/      colab_Phi2.ipynb (clones this repo and runs SFT), explore.ipynb
docs/           TRAINING.md
data/           raw + processed datasets (generated, gitignored)
```

## Technical Stack
- **Data:** Hugging Face Datasets, JSONL preprocessing
- **Training:** Transformers, TRL (GRPO), PEFT (LoRA), Torch, YAML config orchestration
- **Hardware:** Supports 8GB+ VRAM (local) or A100 (cloud)
- **Models:** Microsoft Phi-2, Mistral-7B-Instruct-v0.3 (easily extensible)

## Project Status
| Phase | Status | Output |
|-------|--------|--------|
| Data preprocessing | ✅ Complete | 19,699 train / 1,029 val examples (split by conversation) |
| SFT baseline | ✅ Code complete | Checkpoint pipeline; full training run still to do |
| RL fine-tuning | ✅ Code complete | GRPO + LoRA loop (smoke-tested); full run on SFT checkpoint still to do |
| Evaluation suite | ✅ Code complete | `src/eval.py` tool-call metrics; baseline vs SFT vs RL comparison still to do |
