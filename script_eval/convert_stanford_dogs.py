import argparse
import json
import re
from pathlib import Path
from typing import List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Stanford Dogs-style folder dataset into label-map/classnames/sample files."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Path to dataset image root containing class subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save generated files",
    )
    parser.add_argument(
        "--label-map-name",
        type=str,
        default="stanford_dogs_label_map.json",
        help="Output filename for label map json",
    )
    parser.add_argument(
        "--classnames-name",
        type=str,
        default="stanford_dogs_classnames.txt",
        help="Output filename for classnames txt",
    )
    parser.add_argument(
        "--sample-json-name",
        type=str,
        default="stanford_dogs_sample.json",
        help="Output filename for sample mapping json",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=1,
        help="How many sample images per class to include in sample json",
    )
    return parser.parse_args()


def infer_class_name(folder_name: str) -> str:
    """Infer a readable class name from folder name.

    Example:
    - n02085620-Chihuahua -> Chihuahua
    - n02089867-Walker_hound -> Walker hound
    """
    name = re.sub(r"^n\d{8}-", "", folder_name)
    name = name.replace("_", " ").strip()
    return name


def list_images(folder: Path) -> List[Path]:
    images = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"images-dir does not exist or is not a folder: {images_dir}")

    if args.sample_per_class < 1:
        raise ValueError("--sample-per-class must be >= 1")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_folders = sorted([p for p in images_dir.iterdir() if p.is_dir()])
    if not class_folders:
        raise ValueError(f"No class folders found in: {images_dir}")

    items = []
    classnames = []
    sample_mapping = {}

    for class_index, folder_path in enumerate(class_folders):
        folder_name = folder_path.name
        class_name = infer_class_name(folder_name)
        images = list_images(folder_path)

        if not images:
            # Skip empty class folders to avoid bad mappings.
            continue

        items.append(
            {
                "folder": folder_name,
                "class_index": class_index,
                "class_name": class_name,
            }
        )
        classnames.append(class_name)

        chosen = images[: args.sample_per_class]
        if args.sample_per_class == 1:
            rel_path = chosen[0].relative_to(images_dir).as_posix()
            sample_mapping[folder_name] = rel_path
        else:
            rel_paths = [img.relative_to(images_dir).as_posix() for img in chosen]
            sample_mapping[folder_name] = rel_paths

    if not items:
        raise ValueError(f"No valid classes with images found in: {images_dir}")

    label_map = {"items": items}

    label_map_path = output_dir / args.label_map_name
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    classnames_path = output_dir / args.classnames_name
    with classnames_path.open("w", encoding="utf-8") as f:
        for class_name in classnames:
            f.write(class_name + "\n")

    sample_json_path = output_dir / args.sample_json_name
    with sample_json_path.open("w", encoding="utf-8") as f:
        json.dump(sample_mapping, f, ensure_ascii=False, indent=2)

    print(f"Classes converted: {len(items)}")
    print(f"Saved label map: {label_map_path}")
    print(f"Saved classnames: {classnames_path}")
    print(f"Saved sample json: {sample_json_path}")


if __name__ == "__main__":
    main()
