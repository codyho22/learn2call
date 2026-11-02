"""Evaluation script for tool-call accuracy and schema compliance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM

from prompts import build_prompt_ids
from rewards import RewardConfig, _argument_match_score, compute_reward, parse_prediction
from train_sft import build_tokenizer, get_cfg, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tool-usage models")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--model", default=None, help="Model/checkpoint to evaluate (default: config model)")
    parser.add_argument("--split", default="data/processed/val.jsonl", help="Processed JSONL to evaluate")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None, help="Where to write predictions JSONL")
    parser.add_argument("--predictions", type=Path, default=None, help="Score an existing predictions JSONL instead of generating")
    return parser.parse_args()


def score(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate tool-call metrics over rows with `completion` and `target` text."""

    reward_config = RewardConfig()
    totals = {"json_valid": 0.0, "tool_selection": 0.0, "exact_match": 0.0, "argument_accuracy": 0.0, "reward": 0.0}
    for row in rows:
        pred = parse_prediction(row["completion"])
        ref = json.loads(row["target"])
        totals["reward"] += compute_reward(pred, ref, reward_config)
        if pred is None:
            continue
        totals["json_valid"] += 1
        name_ok = pred.get("name") == ref["name"]
        totals["tool_selection"] += name_ok
        totals["exact_match"] += name_ok and pred.get("arguments") == ref["arguments"]
        totals["argument_accuracy"] += name_ok * _argument_match_score(pred.get("arguments"), ref["arguments"])
    return {key: value / max(len(rows), 1) for key, value in totals.items()}


@torch.no_grad()
def generate(rows: List[Dict[str, Any]], model, tokenizer, max_length: int, max_new_tokens: int, batch_size: int) -> List[str]:
    """Greedy-decode a completion for each processed example."""

    completions: List[str] = []
    prompt_budget = max_length - max_new_tokens
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompt_ids = [build_prompt_ids(r["system"], r["turns"], tokenizer, prompt_budget) for r in batch]
        width = max(len(ids) for ids in prompt_ids)
        # Left-pad so every prompt ends right where generation starts.
        input_ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(ids)) + ids for ids in prompt_ids])
        attention_mask = torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in prompt_ids])
        output = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        completions.extend(tokenizer.batch_decode(output[:, width:], skip_special_tokens=True))
        print(f"Generated {len(completions)}/{len(rows)}")
    return completions


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.predictions:
        with args.predictions.open("r", encoding="utf-8") as fp:
            rows = [json.loads(line) for line in fp]
    else:
        with resolve_path(args.split).open("r", encoding="utf-8") as fp:
            rows = [json.loads(line) for line in fp]
        if args.max_samples:
            rows = rows[:args.max_samples]

        model_name = args.model or get_cfg(config, "model", "model_id", default="gpt2")
        tokenizer = build_tokenizer(model_name)
        use_bf16 = get_cfg(config, "train", "bf16", default=False)
        use_fp16 = get_cfg(config, "train", "fp16", default=False)
        dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 and torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=get_cfg(config, "model", "trust_remote_code", default=False),
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        max_length = get_cfg(config, "train", "max_length", default=1024)
        completions = generate(rows, model, tokenizer, max_length, args.max_new_tokens, args.batch_size)
        rows = [{"completion": c, "target": r["target"]} for c, r in zip(completions, rows)]

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as fp:
                for row in rows:
                    fp.write(json.dumps(row, ensure_ascii=True) + "\n")

    metrics = score(rows)
    print(f"\nEvaluated {len(rows)} examples")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
