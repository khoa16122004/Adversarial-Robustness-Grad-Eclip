import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert CUB folder-class list (e.g. 001.Black_footed_Albatross) "
            "to classification metadata files."
        )
    )
    parser.add_argument(
        "--classnames-folder-path",
        type=str,
        default="data_metadata/cub_classnames_folder.txt",
        help="Path to file containing one class folder name per line",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_metadata",
        help="Directory to save generated files",
    )
    parser.add_argument(
        "--label-map-name",
        type=str,
        default="cub_label_map.json",
        help="Output filename for label map json",
    )
    parser.add_argument(
        "--classnames-name",
        type=str,
        default="cub_classnames.txt",
        help="Output filename for classnames txt",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional image root folder to build sample mapping json",
    )
    parser.add_argument(
        "--sample-json-name",
        type=str,
        default="cub_sample.json",
        help="Output filename for sample mapping json (when --images-dir is provided)",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=1,
        help="How many sample images per class to include when --images-dir is provided",
    )
    return parser.parse_args()


def parse_cub_line(line: str) -> Tuple[int, str, str]:
    """Parse one line like '001.Black_footed_Albatross'.

    Returns:
        (class_index_zero_based, folder_name, class_name)
    """
    text = line.strip()
    parts = text.split()
    if len(parts) >= 2 and parts[0].isdigit():
        # Support lines like: "1 001.Black_footed_Albatross"
        # by taking component 1 as the folder token.
        text = parts[1]

    match = re.match(r"^(\d+)\.(.+)$", text)
    if match is None:
        raise ValueError(f"Invalid CUB class line: {line!r}")

    class_id = int(match.group(1))
    folder_name = text
    class_name = match.group(2).replace("_", " ").strip()
    class_index = class_id - 1
    if class_index < 0:
        raise ValueError(f"Class id must be >= 1, got: {class_id}")
    return class_index, folder_name, class_name


def read_cub_items(classnames_folder_path: Path) -> List[Dict[str, object]]:
    if not classnames_folder_path.exists() or not classnames_folder_path.is_file():
        raise FileNotFoundError(f"File not found: {classnames_folder_path}")

    items: List[Dict[str, object]] = []
    seen_folders = set()
    seen_labels = set()

    with classnames_folder_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            class_index, folder_name, class_name = parse_cub_line(line)
            if folder_name in seen_folders:
                raise ValueError(f"Duplicate folder entry: {folder_name}")
            if class_index in seen_labels:
                raise ValueError(f"Duplicate class index: {class_index}")

            seen_folders.add(folder_name)
            seen_labels.add(class_index)
            items.append(
                {
                    "folder": folder_name,
                    "class_index": class_index,
                    "class_name": class_name,
                }
            )

    if not items:
        raise ValueError(f"No valid classes found in: {classnames_folder_path}")

    items.sort(key=lambda x: int(x["class_index"]))
    return items


def list_images(folder: Path) -> List[Path]:
    return [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def build_sample_mapping(images_dir: Path, items: List[Dict[str, object]], sample_per_class: int) -> Dict[str, object]:
    if sample_per_class < 1:
        raise ValueError("--sample-per-class must be >= 1")

    mapping: Dict[str, object] = {}
    for item in items:
        folder_name = str(item["folder"])
        folder_path = images_dir / folder_name
        if not folder_path.exists() or not folder_path.is_dir():
            continue

        images = list_images(folder_path)
        if not images:
            continue

        chosen = images[:sample_per_class]
        if sample_per_class == 1:
            mapping[folder_name] = chosen[0].relative_to(images_dir).as_posix()
        else:
            mapping[folder_name] = [img.relative_to(images_dir).as_posix() for img in chosen]
    return mapping


def main() -> None:
    args = parse_args()

    classnames_folder_path = Path(args.classnames_folder_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = read_cub_items(classnames_folder_path)
    classnames = [str(item["class_name"]) for item in items]

    label_map = {"items": items}
    label_map_path = output_dir / args.label_map_name
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    classnames_path = output_dir / args.classnames_name
    with classnames_path.open("w", encoding="utf-8") as f:
        for class_name in classnames:
            f.write(class_name + "\n")

    print(f"Classes converted: {len(items)}")
    print(f"Saved label map: {label_map_path}")
    print(f"Saved classnames: {classnames_path}")

    if args.images_dir is not None:
        images_dir = Path(args.images_dir).expanduser().resolve()
        if not images_dir.exists() or not images_dir.is_dir():
            raise FileNotFoundError(f"images-dir does not exist or is not a folder: {images_dir}")

        sample_mapping = build_sample_mapping(images_dir, items, args.sample_per_class)
        sample_json_path = output_dir / args.sample_json_name
        with sample_json_path.open("w", encoding="utf-8") as f:
            json.dump(sample_mapping, f, ensure_ascii=False, indent=2)
        print(f"Saved sample json: {sample_json_path}")


if __name__ == "__main__":
    main()
