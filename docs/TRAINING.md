# Training Guide

## Overview
The pipeline trains an LLM to emit tool calls in three stages: supervised fine-tuning (SFT), RL refinement with GRPO + LoRA, and evaluation. All commands run from the repo root.

## Prerequisites

### Local (Phi-2)
- GPU with at least 8GB VRAM (FP16)
- Python 3.11+, CUDA 11.8+

### Colab
- Phi-2: any GPU runtime (see `notebooks/colab_Phi2.ipynb`)
- Mistral-7B: A100. `mistralai/Mistral-7B-Instruct-v0.3` is gated, so accept its license on Hugging Face and set `HF_TOKEN`.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Process data
```bash
python src/data.py --splits train --val-ratio 0.05 --seed 42
```

This writes:
- `data/raw/apigen_mt5k/train.jsonl`: the raw dataset (5,000 conversations)
- `data/processed/train.jsonl`: 19,699 examples
- `data/processed/val.jsonl`: 1,029 examples

Each example is one tool call, stored as `system` (the tool list), `turns` (the conversation before the call) and `target` (the call as `{"name", "arguments"}` JSON). The train/val split is by conversation. Calls to the `think` pseudo-tool, and calls to tools missing from the tool list, are never used as targets; they stay in the history of later examples.

### 3. Validate setup
```bash
python experiments/test_setup.py
```

This checks the data format, checks that the target survives tokenization, and reports how much history gets dropped to fit `max_length`.

## Prompt fitting
Tools are listed one per line, e.g. `get_user_details(user_id: string) - Get the details of a user`, which takes about 400–520 tokens. When a prompt is longer than `max_length`, the oldest turns are dropped first. The target is never truncated, and an example whose target can't fit is dropped and logged. SFT, RL and eval all build prompts through `src/prompts.py`.

## Stage 1: SFT

Quick test:
```bash
python src/train_sft.py --config experiments/config.yaml --max-train-samples 32 --max-eval-samples 8 --max-steps 2
```

Full run:
```bash
python src/train_sft.py --config experiments/config.yaml
```

Output: `experiments/runs/<run_name>/final` (model + tokenizer).

## Stage 2: RL (GRPO + LoRA)

Check the reward code without training:
```bash
python src/train_rl.py --config experiments/config.yaml --dry-run
```

Train:
```bash
python src/train_rl.py --config experiments/config.yaml
```

- Starts from the SFT checkpoint (`experiments/runs/<run_name>/final`) if it exists, else the base model. Override with `rl.init_model`.
- Samples `rl.num_generations` completions per prompt and scores each with `src/rewards.py`. The maximum reward is 3.0: tool selection + schema validity + argument accuracy, with -0.2 when no tool call can be parsed.
- Output: a LoRA adapter in `experiments/runs/<run_name>_rl/final`.

## Stage 3: Evaluation
```bash
python src/eval.py --config experiments/config.yaml --model experiments/runs/debug/final --output experiments/runs/debug/preds.jsonl
```

Reports JSON validity, tool-selection accuracy, exact match, argument accuracy and mean reward on `data/processed/val.jsonl`. `--model` accepts a Hub ID, an SFT checkpoint, or an RL adapter. `--predictions <file>` re-scores saved predictions without running the model again.

## Key configuration

| Parameter | Local (Phi-2) | Colab (Mistral-7B) | Description |
|-----------|---------------|-------------------|-------------|
| `model.model_id` | microsoft/phi-2 | mistralai/Mistral-7B-Instruct-v0.3 | Hugging Face model ID |
| `train.per_device_batch_size` | 1 | 4 | SFT batch size per GPU |
| `train.gradient_accumulation_steps` | 8 | 4 | SFT steps per optimizer update |
| `train.max_length` | 2048 | 3200 | Prompt + target token budget |
| `train.fp16` / `train.bf16` | fp16 | bf16 | Mixed precision |
| `train.sft_epochs` | 1 | 3 | SFT epochs |
| `train.rl_steps` | 500 | 2000 | GRPO steps |
| `rl.num_generations` | 4 | 4 | GRPO group size |
| `rl.max_completion_length` | 128 | 128 | Tokens sampled per completion (covers ~95% of targets) |
| `rl.beta` | 0.04 | 0.04 | KL penalty against the base weights |
| `rl.lora_r` / `rl.lora_alpha` | 16 / 32 | 16 / 32 | LoRA rank / scale |

## Common issues

### Out of memory
- SFT: lower `per_device_batch_size` and raise `gradient_accumulation_steps`
- RL: lower `rl.per_device_batch_size` or `rl.max_completion_length`. The batch size times gradient accumulation must divide by `rl.num_generations`.

### PyTorch not found / CPU only
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
