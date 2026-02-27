# Tool-RL: Function Calling for Compact Language Models

## Overview
A production-ready pipeline for training small, open-source LLMs (1B–7B parameters) to reliably generate tool calls with correct function selection, argument validation, and JSON compliance using supervised fine-tuning and reinforcement learning.

## Key Features
- **Data Pipeline:** Preprocesses APIGen-MT-5k (~5k examples) into prompt/target pairs with chat templates
- **SFT Training:** Hugging Face Trainer support for Phi-2 (2.7B) and Mistral-7B with mixed-precision and gradient accumulation
- **RL Ready:** Modular reward shaping architecture for PPO-style fine-tuning on tool-call accuracy, schema validity, and argument correctness
- **Evaluation:** Metrics for JSON validity, schema compliance, and tool-selection accuracy

## Quick Start
```bash
# Setup
pip install -r requirements.txt
python src/data.py --splits train --val-ratio 0.05

# Train
python src/train_sft.py --config experiments/config.yaml

# (Upcoming) RL refinement
python src/train_rl.py --config experiments/config.yaml
```

## Technical Stack
- **Data:** Hugging Face Datasets, JSONL preprocessing
- **Training:** Transformers, Torch, YAML config orchestration
- **Hardware:** Supports 8GB+ VRAM (local) or A100 (cloud)
- **Models:** Microsoft Phi-2, Mistral-7B (easily extensible)

## Project Status
| Phase | Status | Output |
|-------|--------|--------|
| Data preprocessing | ✅ Complete | ~4.7k train / ~250 val examples |
| SFT baseline | ✅ Complete | Working model checkpoint pipeline |
| RL fine-tuning | 🔄 In Progress | Reward shaping + PPO loop |
| Evaluation suite | ⏳ Next | Tool-call accuracy metrics |
