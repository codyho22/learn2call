import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from datasets import Dataset, load_dataset

from prompts import render_system, render_tool_call

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "apigen_mt5k"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"


def load_apigen(split: str = "train") -> Dataset:
    """Load the requested APIGen-MT-5k split from Hugging Face."""

    return load_dataset("Salesforce/APIGen-MT-5k", split=split)


def dump_raw_split(dataset: Dataset, split: str) -> Path:
    """Persist the raw split as JSONL inside data/raw for reproducibility."""

    RAW_PATH.mkdir(parents=True, exist_ok=True)
    out_path = RAW_PATH / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as fp:
        for row in dataset:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")
    return out_path


SKIPPED_TARGET_TOOLS = {"think"}
ROLE_MAP = {"human": "user", "gpt": "assistant", "function_call": "assistant", "observation": "tool"}


def parse_tool_call(value: str) -> Dict | None:
    """Parse a function_call message value into a {"name", "arguments"} dict."""

    try:
        call_dict = json.loads(value or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(call_dict, dict) or not call_dict.get("name"):
        return None
    arguments = call_dict.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return {"name": call_dict["name"], "arguments": arguments}


def format_conversation(example: Dict) -> List[Dict]:
    """Expand one conversation into a training example per tool call.

    Each example's prompt is the conversation history before that call; tool
    calls in the history are rendered in the same JSON format as targets.
    """

    tools = example.get("tools", [])
    if isinstance(tools, str):
        tools = json.loads(tools)
    system = render_system(tools)
    # Calls to tools missing from the tool list are dataset noise: keep them in
    # the history (they happened) but never train on them as targets.
    tool_names = {tool.get("name") for tool in tools}

    records: List[Dict] = []
    turns: List[Dict[str, str]] = []
    for message in example.get("conversations", []):
        role = ROLE_MAP.get(message.get("from", "").lower())
        if role is None:
            continue
        content = message.get("value", "")

        if message.get("from", "").lower() == "function_call":
            call = parse_tool_call(content)
            if call is None:
                continue
            content = render_tool_call(call["name"], call["arguments"])
            has_user_turn = any(turn["role"] == "user" for turn in turns)
            is_target_tool = call["name"] in tool_names and call["name"] not in SKIPPED_TARGET_TOOLS
            if is_target_tool and has_user_turn:
                records.append({"system": system, "turns": list(turns), "target": content})

        turns.append({"role": role, "content": content})
    return records


def preprocess_dataset(dataset: Iterable[Dict]) -> List[Dict]:
    """Expand every conversation into per-tool-call examples."""

    processed: List[Dict] = []
    for record in dataset:
        processed.extend(format_conversation(record))
    return processed


def split_train_val(
    examples: List[Dict],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Shuffle and split examples (conversations) into train/val subsets."""

    if not examples or val_ratio <= 0:
        return examples, []

    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_examples = shuffled[:val_count]
    train_examples = shuffled[val_count:]
    return train_examples, val_examples


def write_jsonl(records: Sequence[Dict], path: Path) -> None:
    """Write list of dicts as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download + preprocess APIGen-MT-5k")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="HF dataset splits to download (default: train)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation ratio when splitting the train split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for the train/val split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        dataset = load_apigen(split)
        raw_path = dump_raw_split(dataset, split)

        if split == "train":
            # Split by conversation so no conversation appears in both sets.
            train_convos, val_convos = split_train_val(
                list(dataset), args.val_ratio, args.seed
            )
            train_records = preprocess_dataset(train_convos)
            write_jsonl(train_records, PROCESSED_PATH / "train.jsonl")
            print(f"Wrote {len(train_records)} train examples from {len(train_convos)} conversations")
            if val_convos:
                val_records = preprocess_dataset(val_convos)
                write_jsonl(val_records, PROCESSED_PATH / "val.jsonl")
                print(f"Wrote {len(val_records)} val examples from {len(val_convos)} conversations")
        else:
            write_jsonl(preprocess_dataset(dataset), PROCESSED_PATH / f"{split}.jsonl")

        print(
            f"Saved raw {split} split to {raw_path.relative_to(PROJECT_ROOT)} "
            f"({len(dataset)} rows)"
        )


if __name__ == "__main__":
    main()
