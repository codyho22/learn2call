"""Supervised fine-tuning entrypoint."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def get_cfg(cfg: Dict[str, Any], *keys: str, default: Any) -> Any:
    cursor: Any = cfg
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_tokenizer(name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def tokenize_examples(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dict[str, List[List[int]]]:
    input_ids_list: List[List[int]] = []
    attention_masks: List[List[int]] = []
    labels_list: List[List[int]] = []

    eos_id = tokenizer.eos_token_id
    for prompt, target in zip(examples["prompt"], examples["target"]):
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        if eos_id is not None:
            target_ids = target_ids + [eos_id]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        if max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        attention_mask = [1] * len(input_ids)
        input_ids_list.append(input_ids)
        attention_masks.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_masks,
        "labels": labels_list,
    }


def collate_batch(batch: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    pad_id = tokenizer.pad_token_id
    input_ids: List[List[int]] = []
    attention_masks: List[List[int]] = []
    labels: List[List[int]] = []

    for item in batch:
        length = len(item["input_ids"])
        pad_size = max_len - length
        input_ids.append(item["input_ids"] + [pad_id] * pad_size)
        attention_masks.append(item["attention_mask"] + [0] * pad_size)
        labels.append(item["labels"] + [-100] * pad_size)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SFT model for tool use")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--model-name", default=None, help="Override model name")
    parser.add_argument("--tokenizer-name", default=None, help="Override tokenizer name")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = get_cfg(config, "seed", default=42)
    set_seed(seed)

    # New config structure: reads from model.model_id / model.tokenizer_id
    # Falls back to legacy model_name / tokenizer_name for backward compatibility
    model_name = args.model_name or get_cfg(config, "model", "model_id", default=None)
    if model_name is None:
        model_name = get_cfg(config, "model_name", default="gpt2")
    
    tokenizer_name = args.tokenizer_name or get_cfg(config, "model", "tokenizer_id", default=None)
    if tokenizer_name is None:
        tokenizer_name = get_cfg(config, "tokenizer_name", default=model_name)
    output_root = resolve_path(get_cfg(config, "logging", "output_dir", default="experiments/runs"))
    run_name = get_cfg(config, "run_name", default="sft")
    output_dir = output_root / run_name

    train_path = resolve_path(get_cfg(config, "data", "processed_path", default="data/processed/train.jsonl"))
    eval_path = resolve_path(get_cfg(config, "data", "eval_path", default="data/processed/val.jsonl"))
    data_files: Dict[str, str] = {"train": str(train_path)}
    if eval_path.exists():
        data_files["validation"] = str(eval_path)

    dataset: DatasetDict = load_dataset("json", data_files=data_files)
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))
    if args.max_eval_samples and "validation" in dataset:
        dataset["validation"] = dataset["validation"].select(range(args.max_eval_samples))

    tokenizer = build_tokenizer(tokenizer_name)
    max_length = get_cfg(config, "train", "max_length", default=1024)

    tokenized = dataset.map(
        lambda batch: tokenize_examples(batch, tokenizer, max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    per_device_batch_size = get_cfg(config, "train", "per_device_batch_size", default=2)
    learning_rate = get_cfg(config, "train", "learning_rate", default=5e-5)
    num_epochs = get_cfg(config, "train", "sft_epochs", default=1)
    log_interval = get_cfg(config, "logging", "log_interval", default=50)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        logging_steps=log_interval,
        evaluation_strategy="steps" if "validation" in tokenized else "no",
        save_strategy="steps",
        save_steps=log_interval,
        max_steps=args.max_steps if args.max_steps else -1,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=lambda batch: collate_batch(batch, tokenizer),
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))


if __name__ == "__main__":
    main()
