import argparse
import json
import random
from pathlib import Path


def sample_json(data, n: int):
	"""Return a random subset from list or dict JSON data."""
	if isinstance(data, list):
		if n > len(data):
			raise ValueError(f"n={n} is larger than list size={len(data)}")
		return random.sample(data, n)

	if isinstance(data, dict):
		if n > len(data):
			raise ValueError(f"n={n} is larger than dict size={len(data)}")
		keys = random.sample(list(data.keys()), n)
		return {k: data[k] for k in keys}

	raise TypeError(
		"Input JSON must be either a list or dict. "
		f"Found type: {type(data).__name__}"
	)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Randomly sample items from a JSON file and write to output file."
	)
	parser.add_argument(
		"--input_json",
		type=str,
		help="Path to input JSON file (e.g. classification_reuslt/imagenet/vit_b_16_100.json)",
	)
	parser.add_argument(
		"-n",
		"--num-samples",
		type=int,
		required=True,
		help="Number of items to sample",
	)
	parser.add_argument(
		"-o",
		"--output-json",
		type=str,
		default=None,
		help="Path to output JSON file. Default: <input_stem>_random_<n>.json",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Random seed for reproducible sampling",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	if args.num_samples <= 0:
		raise ValueError("--num-samples must be > 0")

	if args.seed is not None:
		random.seed(args.seed)

	input_path = Path(args.input_json)
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	with input_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	sampled = sample_json(data, args.num_samples)

	if args.output_json is None:
		output_path = input_path.with_name(
			f"{input_path.stem}_random_{args.num_samples}.json"
		)
	else:
		output_path = Path(args.output_json)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(sampled, f, indent=2, ensure_ascii=False)

	print(f"Input size: {len(data)}")
	print(f"Sampled: {args.num_samples}")
	print(f"Saved to: {output_path}")


if __name__ == "__main__":
	main()
