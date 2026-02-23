import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from datasets import Dataset, load_dataset

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


def extract_tool_call(messages: Sequence[Dict]) -> Dict | None:
    """Return the first tool call (function_call from="" message value) if present."""

    for message in messages:
        if message.get("from", "").lower() == "function_call":
            try:
                call_dict = json.loads(message.get("value", "{}"))
                if call_dict.get("name"):
                    return call_dict
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def extract_latest_user_message(messages: Sequence[Dict]) -> str | None:
    """Return the most recent user (human) message content, if available."""

    user_messages = [m for m in messages if m.get("from", "").lower() == "human"]
    if not user_messages:
        return None
    return user_messages[-1].get("value", "").strip()


def format_prompt(tools: Sequence[Dict], user_message: str) -> str:
    """Construct the textual prompt shown to the model."""

    tools_json = json.dumps(tools, indent=2)
    return (
        "SYSTEM: You have access to these tools:\n"
        f"{tools_json}\n\n"
        f"USER: {user_message}\n"
        "ASSISTANT:"
    )


def format_example(example: Dict) -> Dict | None:
    """Convert a raw dataset record into a prompt/target pair."""

    tools = example.get("tools", [])
    # APIGen-MT-5k uses "conversations" instead of "messages"
    conversations = example.get("conversations", [])
    tool_call = extract_tool_call(conversations)
    user_message = extract_latest_user_message(conversations)
    if not tool_call or not user_message:
        return None

    prompt = format_prompt(tools, user_message)
    target = json.dumps(
        {
            "name": tool_call.get("name"),
            "arguments": tool_call.get("arguments"),
        },
        ensure_ascii=True,
    )

    return {"prompt": prompt, "target": target}


def preprocess_dataset(dataset: Iterable[Dict]) -> List[Dict]:
    """Apply formatting and drop records that lack tool calls."""

    processed: List[Dict] = []
    for record in dataset:
        formatted = format_example(record)
        if formatted:
            processed.append(formatted)
    return processed


def split_train_val(
    examples: List[Dict],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Shuffle and split processed examples into train/val subsets."""

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
        processed_records = preprocess_dataset(dataset)

        if split == "train":
            train_records, val_records = split_train_val(
                processed_records, args.val_ratio, args.seed
            )
            write_jsonl(train_records, PROCESSED_PATH / "train.jsonl")
            if val_records:
                write_jsonl(val_records, PROCESSED_PATH / "val.jsonl")
        else:
            write_jsonl(processed_records, PROCESSED_PATH / f"{split}.jsonl")

        print(
            f"Saved raw {split} split to {raw_path.relative_to(PROJECT_ROOT)} "
            f"({len(dataset)} rows)"
        )


if __name__ == "__main__":
    main()
