import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample from either a plain JSON mapping or a classification report "
            "containing correct_samples."
        )
    )
    parser.add_argument(
        "--input",
        default="vit_b_16_1k.json",
        help=(
            "Path to source JSON. Supported formats: "
            "(1) mapping {class_folder: image_path} or "
            "(2) report with correct_samples list."
        ),
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
        help=(
            "Number of random pairs to sample. "
            "For correct_samples input, this is number of classes (keys) sampled."
        ),
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


def _key_from_sample(sample):
    image_path = sample.get("image_path")
    if not image_path:
        return None, None

    key = sample.get("wnid")
    if key is None:
        key = Path(image_path).parent.name

    key = str(key).strip()
    if not key:
        return None, None
    return key, str(image_path)


def build_mapping_candidates(data):
    if isinstance(data, dict) and isinstance(data.get("correct_samples"), list):
        correct_samples = data["correct_samples"]
        candidates = {}
        for sample in correct_samples:
            if not isinstance(sample, dict):
                continue
            key, image_path = _key_from_sample(sample)
            if key is None:
                continue
            candidates.setdefault(key, []).append(image_path)

        if not candidates:
            raise ValueError("correct_samples exists but no valid (key, image_path) entries were found.")

        return candidates, "correct_samples"

    if isinstance(data, dict):
        mapping = {str(k): str(v) for k, v in data.items()}
        if not mapping:
            raise ValueError("Input mapping is empty.")
        return {k: [v] for k, v in mapping.items()}, "mapping"

    raise ValueError("Input JSON must be a mapping or a dict containing correct_samples.")


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    candidates, source_type = build_mapping_candidates(data)

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")

    keys = list(candidates.keys())
    if args.num_samples > len(keys):
        raise ValueError(
            f"--num-samples ({args.num_samples}) is larger than available keys ({len(keys)})."
        )

    rng = random.Random(args.seed)
    sampled_keys = rng.sample(keys, args.num_samples)
    sampled_items = [(key, rng.choice(candidates[key])) for key in sampled_keys]

    if args.sort_keys:
        sampled_items.sort(key=lambda kv: kv[0])

    sampled_dict = dict(sampled_items)

    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sampled_dict, f, ensure_ascii=False, indent=2)

    print(f"Input type: {source_type}")
    print(f"Saved {len(sampled_dict)} samples to {output_path}")


if __name__ == "__main__":
    main()
