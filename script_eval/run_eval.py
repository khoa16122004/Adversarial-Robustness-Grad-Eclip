import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run eval_curve_information_fair_auc.py for each explain method folder "
			"under a parent root."
		)
	)
	parser.add_argument(
		"--parent-root",
		type=str,
		default="D:/Adversarial-Robustness-Grad-Eclip/result/ImageNet/CLIP_ViTB32/del_optimize_output",
		help=(
			"Parent folder containing explain-method subfolders "
			"(e.g., del_optimize_output/eclip, del_optimize_output/game, ...)."
		),
	)
	parser.add_argument(
		"--methods",
		nargs="*",
		default=None,
		help="Optional explain methods to run. Default: auto-detect all subfolders.",
	)
	parser.add_argument(
		"--pattern",
		type=str,
		default="*information.json",
		help="Filename pattern to pass through to evaluation script.",
	)
	parser.add_argument(
		"--metrics",
		nargs="*",
		default=None,
		help="Optional metrics to pass through (e.g., --metrics del ins).",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=300,
		help="Figure DPI passed to evaluation script.",
	)
	parser.add_argument(
		"--save-per-folder-summary",
		action="store_true",
		help="Pass --save-per-folder-summary to evaluation script.",
	)
	return parser.parse_args()


def discover_methods(parent_root: Path, selected_methods: List[str] | None) -> List[str]:
	if selected_methods:
		return selected_methods

	methods = sorted([p.name for p in parent_root.iterdir() if p.is_dir()])
	return methods


def main() -> None:
	args = parse_args()

	parent_root = Path(args.parent_root).expanduser().resolve()
	if not parent_root.exists() or not parent_root.is_dir():
		raise FileNotFoundError(f"Parent root does not exist or is not a directory: {parent_root}")

	eval_script = Path(__file__).with_name("eval_curve_information_fair_auc.py")
	if not eval_script.exists():
		raise FileNotFoundError(f"Cannot find script: {eval_script}")

	methods = discover_methods(parent_root, args.methods)
	if not methods:
		raise RuntimeError(f"No explain-method subfolders found under: {parent_root}")

	print(f"Parent root: {parent_root}")
	print(f"Methods: {', '.join(methods)}")

	failures: List[str] = []

	for method in methods:
		input_root = parent_root / method
		if not input_root.exists() or not input_root.is_dir():
			print(f"[SKIP] {method}: folder not found ({input_root})")
			failures.append(method)
			continue

		cmd = [
			sys.executable,
			str(eval_script),
			"--input-root",
			str(input_root),
			"--pattern",
			args.pattern,
			"--dpi",
			str(args.dpi),
		]

		if args.metrics:
			cmd.extend(["--metrics", *args.metrics])
		if args.save_per_folder_summary:
			cmd.append("--save-per-folder-summary")

		print("=" * 80)
		print(f"[RUN] method={method}")
		print("Command:", " ".join(cmd))

		result = subprocess.run(cmd, check=False)
		if result.returncode != 0:
			failures.append(method)
			print(f"[FAIL] method={method}, returncode={result.returncode}")
		else:
			print(f"[OK] method={method}")

	print("=" * 80)
	if failures:
		print(f"Finished with failures: {', '.join(failures)}")
		raise SystemExit(1)

	print("All methods finished successfully.")


if __name__ == "__main__":
	main()
