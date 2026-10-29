"""Generate a classify.py label mapping from immediate class subfolders."""

import argparse
import json
import re
from pathlib import Path


def build_label_index(data_path, strip_prefix_pattern=r"^n\d+-"):
    prefix_pattern = re.compile(strip_prefix_pattern)
    if not data_path.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {data_path}")

    folders = sorted(
        folder for folder in data_path.iterdir()
        if folder.is_dir() and not folder.name.startswith(".")
    )
    if not folders:
        raise ValueError(f"No class subfolders found in: {data_path}")

    label_index = {}
    for label, folder in enumerate(folders):
        class_name = prefix_pattern.sub("", folder.name, count=1)
        class_name = class_name.replace("_", " ").replace("-", " ").strip()
        if not class_name:
            raise ValueError(f"Cannot infer class name from folder: {folder.name}")
        label_index[folder.name] = [label, class_name]
    return label_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", required=True, type=Path,
        help="Dataset root containing class subfolders, e.g. n02090379-redbone",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("standfordpet_label.json"),
        help="Output label mapping JSON (default: standfordpet_label.json)",
    )
    parser.add_argument(
        "--strip-prefix-pattern", default=r"^n\d+-.",
        help=(
            r"Regex removed once from each folder name before normalizing spaces "
            r"(default: ^n\d+-). For CUB use ^[^.]*\. to remove through the first dot. "
            "Unmatched folder names are unchanged."
        ),
    )
    args = parser.parse_args()

    label_index = build_label_index(args.data_path, args.strip_prefix_pattern)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(label_index, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Saved {len(label_index)} classes to: {args.output}")


if __name__ == "__main__":
    main()
