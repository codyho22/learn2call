"""Reinforcement learning entrypoint (GRPO + LoRA via trl)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from prompts import build_prompt_ids
from rewards import RewardConfig, compute_reward, parse_prediction
from train_sft import build_tokenizer, get_cfg, load_config, resolve_path


def build_reward_config(cfg: Dict[str, Any]) -> RewardConfig:
    reward_cfg = get_cfg(cfg, "reward", default={})
    return RewardConfig(
        tool_selection=float(reward_cfg.get("tool_selection", 1.0)),
        schema_validity=float(reward_cfg.get("schema_validity", 1.0)),
        argument_accuracy=float(reward_cfg.get("argument_accuracy", 1.0)),
        no_call_penalty=float(reward_cfg.get("no_call_penalty", -0.2)),
    )


def parse_reference(target: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(target)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def stub_policy_prediction(reference: Dict[str, Any]) -> Dict[str, Any]:
    # Dry-run behavior: echo reference so reward path is verifiable.
    return {
        "name": reference.get("name"),
        "arguments": reference.get("arguments", {}),
    }


def resolve_init_model(cfg: Dict[str, Any]) -> str:
    """Start RL from the SFT checkpoint when it exists, else the base model."""
    init_model = get_cfg(cfg, "rl", "init_model", default=None)
    if init_model is None:
        output_root = get_cfg(cfg, "logging", "output_dir", default="experiments/runs")
        init_model = f"{output_root}/{get_cfg(cfg, 'run_name', default='sft')}/final"
    if resolve_path(init_model).exists():
        return str(resolve_path(init_model))
    base_model = get_cfg(cfg, "model", "model_id", default="gpt2")
    print(f"RL init model {init_model} not found; starting from base model {base_model}")
    return base_model


def run_grpo(cfg: Dict[str, Any], ds: Dataset, reward_config: RewardConfig, output_dir: Path, rl_steps: int) -> None:
    model_name = resolve_init_model(cfg)
    tokenizer = build_tokenizer(model_name)

    max_length = get_cfg(cfg, "train", "max_length", default=1024)
    max_completion_length = get_cfg(cfg, "rl", "max_completion_length", default=128)
    prompt_budget = max_length - max_completion_length

    # GRPO takes text prompts; fit them to the budget the same way SFT does.
    ds = ds.map(
        lambda row: {"prompt": tokenizer.decode(build_prompt_ids(row["system"], row["turns"], tokenizer, prompt_budget))},
        remove_columns=["system", "turns"],
    )

    def tool_call_reward(completions: List[str], target: List[str], **kwargs: Any) -> List[float]:
        return [
            compute_reward(parse_prediction(completion), parse_reference(ref), reward_config)
            for completion, ref in zip(completions, target)
        ]

    use_fp16 = get_cfg(cfg, "train", "fp16", default=False)
    use_bf16 = get_cfg(cfg, "train", "bf16", default=False)
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=get_cfg(cfg, "model", "trust_remote_code", default=False),
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    log_interval = get_cfg(cfg, "logging", "log_interval", default=10)
    args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=rl_steps,
        learning_rate=float(get_cfg(cfg, "rl", "learning_rate", default=1e-5)),
        per_device_train_batch_size=get_cfg(cfg, "rl", "per_device_batch_size", default=4),
        gradient_accumulation_steps=get_cfg(cfg, "rl", "gradient_accumulation_steps", default=2),
        num_generations=get_cfg(cfg, "rl", "num_generations", default=4),
        max_completion_length=max_completion_length,
        temperature=float(get_cfg(cfg, "rl", "temperature", default=1.0)),
        beta=float(get_cfg(cfg, "rl", "beta", default=0.04)),
        logging_steps=log_interval,
        save_steps=log_interval * 5,
        save_total_limit=3,
        fp16=use_fp16 and not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=get_cfg(cfg, "train", "gradient_checkpointing", default=False),
        report_to=[],
    )
    peft_config = LoraConfig(
        r=get_cfg(cfg, "rl", "lora_r", default=16),
        lora_alpha=get_cfg(cfg, "rl", "lora_alpha", default=32),
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=tool_call_reward,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    print(f"Starting GRPO from {model_name}: {rl_steps} steps, {len(ds)} prompts")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved LoRA adapter to {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agent for tool use")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Override train.rl_steps")
    parser.add_argument("--dry-run", action="store_true", help="Only check the reward path; no training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    train_path = resolve_path(get_cfg(cfg, "data", "processed_path", default="data/processed/train.jsonl"))
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    reward_config = build_reward_config(cfg)
    run_name = get_cfg(cfg, "run_name", default="rl")
    rl_steps = args.max_steps or int(get_cfg(cfg, "train", "rl_steps", default=500))
    output_dir = resolve_path(get_cfg(cfg, "logging", "output_dir", default="experiments/runs")) / f"{run_name}_rl"

    ds = load_dataset("json", data_files={"train": str(train_path)})["train"]
    if args.max_train_samples and args.max_train_samples < len(ds):
        ds = ds.select(range(args.max_train_samples))

    if args.dry_run:
        rewards = [
            compute_reward(stub_policy_prediction(ref), ref, reward_config)
            for ref in (parse_reference(target) for target in ds["target"])
        ]
        print(f"Dry run on {len(rewards)} examples (reference echoed as prediction)")
        print(f"Reward stats -> mean: {sum(rewards) / len(rewards):.4f}, min: {min(rewards):.4f}, max: {max(rewards):.4f}")
        return

    run_grpo(cfg, ds, reward_config, output_dir, rl_steps)


if __name__ == "__main__":
    main()
