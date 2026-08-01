import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample key-value pairs from a source JSON mapping."
    )
    parser.add_argument(
        "--input",
        default="vit_b_16_1k.json",
        help="Path to source JSON mapping (e.g., vit_b_16_1k.json)",
    )
    parser.add_argument(
        "--output",
        default="vit_b_16_100.json",
        help="Path to output JSON mapping (e.g., vit_b_16_100.json)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random pairs to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--sort-keys",
        action="store_true",
        help="Sort sampled keys alphabetically before writing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object/dict mapping.")

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")

    items = list(data.items())
    if args.num_samples > len(items):
        raise ValueError(
            f"--num-samples ({args.num_samples}) is larger than available items ({len(items)})."
        )

    rng = random.Random(args.seed)
    sampled_items = rng.sample(items, args.num_samples)

    if args.sort_keys:
        sampled_items.sort(key=lambda kv: kv[0])

    sampled_dict = dict(sampled_items)

    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sampled_dict, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(sampled_dict)} samples to {output_path}")


if __name__ == "__main__":
    main()
