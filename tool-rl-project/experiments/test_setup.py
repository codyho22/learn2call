"""Test script to validate data and tokenization before training."""
from pathlib import Path
import json
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_sft import build_tokenizer, tokenize_examples, load_config, get_cfg


def test_data_format():
    """Check if processed data files exist and are correctly formatted."""
    print("=" * 60)
    print("Testing data format...")
    print("=" * 60)
    
    train_path = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
    val_path = PROJECT_ROOT / "data" / "processed" / "val.jsonl"
    
    if not train_path.exists():
        print(f"[FAIL] Train data not found at {train_path}")
        return False
    
    if not val_path.exists():
        print(f"[FAIL] Val data not found at {val_path}")
        return False
    
    # Check a few examples
    with train_path.open("r") as f:
        examples = [json.loads(line) for line in f.readlines()[:5]]
    
    for i, ex in enumerate(examples):
        if "prompt" not in ex or "target" not in ex:
            print(f"[FAIL] Example {i} missing 'prompt' or 'target' key")
            return False
            
        if not isinstance(ex["prompt"], str) or not isinstance(ex["target"], str):
            print(f"[FAIL] Example {i} has non-string values")
            return False
    
    print(f"[OK] Found {train_path.stat().st_size // 1024 // 1024}MB train data")
    print(f"[OK] Found {val_path.stat().st_size // 1024 // 1024}MB val data")
    print(f"[OK] Data format looks good")
    return True


def test_tokenization():
    """Test tokenization with a sample."""
    print("\n" + "=" * 60)
    print("Testing tokenization...")
    print("=" * 60)
    
    config_path = PROJECT_ROOT / "experiments" / "config.yaml"
    config = load_config(config_path)
    
    model_id = get_cfg(config, "model", "model_id", default="microsoft/phi-2")
    tokenizer_id = get_cfg(config, "model", "tokenizer_id", default=model_id)
    max_length = get_cfg(config, "train", "max_length", default=512)
    
    print(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = build_tokenizer(tokenizer_id)
    
    # Create a small test example
    test_examples = {
        "prompt": ["SYSTEM: Test\nUSER: Hello\nASSISTANT:"],
        "target": ['{"name": "test_func", "arguments": {"arg": "value"}}'],
    }
    
    try:
        result = tokenize_examples(test_examples, tokenizer, max_length)
        
        print(f"[OK] Tokenization successful")
        print(f"   Input IDs length: {len(result['input_ids'][0])}")
        print(f"   Attention mask length: {len(result['attention_mask'][0])}")
        print(f"   Labels length: {len(result['labels'][0])}")
        
        # Check lengths match
        if len(result['input_ids'][0]) != len(result['labels'][0]):
            print(f"[FAIL] Length mismatch: input_ids != labels")
            return False
        
        return True
    except Exception as e:
        print(f"[FAIL] Tokenization failed: {e}")
        return False


def test_sequence_lengths():
    """Check sequence length distribution."""
    print("\n" + "=" * 60)
    print("Checking sequence lengths...")
    print("=" * 60)
    
    train_path = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
    config_path = PROJECT_ROOT / "experiments" / "config.yaml"
    config = load_config(config_path)
    
    model_id = get_cfg(config, "model", "model_id", default="microsoft/phi-2")
    tokenizer_id = get_cfg(config, "model", "tokenizer_id", default=model_id)
    max_length = get_cfg(config, "train", "max_length", default=512)
    
    tokenizer = build_tokenizer(tokenizer_id)
    
    # Sample 100 examples
    with train_path.open("r") as f:
        examples = [json.loads(line) for line in f.readlines()[:100]]
    
    lengths = []
    truncated = 0
    
    for ex in examples:
        prompt_len = len(tokenizer(ex["prompt"], add_special_tokens=False).input_ids)
        target_len = len(tokenizer(ex["target"], add_special_tokens=False).input_ids)
        total_len = prompt_len + target_len + 1  # +1 for EOS
        lengths.append(total_len)
        
        if total_len > max_length:
            truncated += 1
    
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    min_len = min(lengths)
    
    print(f"Sequence length statistics (n=100):")
    print(f"  Average: {avg_len:.0f}")
    print(f"  Min: {min_len}")
    print(f"  Max: {max_len}")
    print(f"  Max length config: {max_length}")
    print(f"  Truncated: {truncated}/{len(lengths)} ({100*truncated/len(lengths):.1f}%)")
    
    if truncated > 50:
        print(f"[WARN]  Warning: {truncated}% of sequences will be truncated")
        print(f"   Consider increasing max_length in config.yaml")
    else:
        print(f"[OK] Truncation rate looks acceptable")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VALIDATING TRAINING SETUP")
    print("=" * 60 + "\n")
    
    tests = [
        ("Data Format", test_data_format),
        ("Tokenization", test_tokenization),
        ("Sequence Lengths", test_sequence_lengths),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n🎉 All tests passed! Ready to train.")
        return 0
    else:
        print("\n[WARN]  Some tests failed. Please fix before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
