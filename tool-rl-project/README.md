# Tool-RL Project

Train a compact open-source language model to follow tool-usage instructions using supervised fine-tuning (SFT) and reinforcement learning (RL) on the APIGen-MT-5k dataset.

## Repository Structure

```
tool-rl-project/
├── data/
│   ├── raw/        # downloaded APIGen-MT-5k dumps
│   └── processed/  # prompt/target pairs ready for training
├── src/
│   ├── data.py     # data loading + preprocessing helpers
│   ├── prompts.py  # chat + tool prompt templates
│   ├── rewards.py  # reward shaping utilities for RL
│   ├── train_sft.py# supervised fine-tuning loop
│   ├── train_rl.py # RLHF-style PPO loop
│   ├── eval.py     # evaluation + metrics
│   └── utils.py    # misc helpers (logging, configs)
├── notebooks/
│   └── explore.ipynb
├── experiments/
│   └── config.yaml # experiment defaults
├── requirements.txt
└── README.md
```

## Environment Setup

1. **Create a virtual environment** (pick one):
   - `python -m venv .venv && .venv\Scripts\activate`
   - `conda create -n tool-rl python=3.11 && conda activate tool-rl`
2. **Install dependencies:** `pip install -r requirements.txt`
3. (Optional) Install CUDA-enabled PyTorch following https://pytorch.org/get-started/locally/

## Data Workflow

1. Download APIGen-MT-5k into `data/raw/`.
2. Run `python -m src.data --prepare` to clean + split conversations into prompt/target records stored under `data/processed/`.
3. Inspect samples in `notebooks/explore.ipynb`.

## Training

- **SFT:** `python -m src.train_sft --config experiments/config.yaml`
- **RL:** `python -m src.train_rl --config experiments/config.yaml`

Both scripts expect Hugging Face style model identifiers and will write checkpoints/logs under `experiments/` (configurable).

## Evaluation

Use `python -m src.eval --split validation` to compute tool-call accuracy, JSON validity, and schema compliance metrics.

## Notes

- The project targets smaller models (e.g., 1B-3B parameters) to control compute.
- Reward shaping encourages accurate tool selection, argument validity, and restraint when no tool is suitable.
- Update `experiments/config.yaml` to point to your model, tokenizer, dataset shards, and logging destinations.
