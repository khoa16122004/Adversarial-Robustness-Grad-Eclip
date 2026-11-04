import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

try:
    from CLIP import clip
except ImportError:
    import clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a clean-image subset listed in JSON into a separate output folder, "
            "preserving relative structure and resizing images."
        )
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help=(
            "Path to JSON file. Supports: "
            "(1) mapping {class_key: relative_image_path}, or "
            "(2) report with correct_samples containing image_path."
        ),
    )
    parser.add_argument(
        "--images-root",
        required=True,
        help="Root folder containing clean images referred by relative paths in JSON.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output folder to store resized subset images.",
    )
    parser.add_argument(
        "--size-source",
        default="clip_preprocess",
        choices=["clip_preprocess", "manual"],
        help=(
            "How to determine resize size. "
            "clip_preprocess: infer from CLIP preprocess transforms; manual: use --resize-width/--resize-height."
        ),
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Output image width after resize (used only when --size-source manual).",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=None,
        help="Output image height after resize (used only when --size-source manual).",
    )
    parser.add_argument("--clip-model", default="ViT-B/16", help="CLIP model name used for preprocess")
    parser.add_argument(
        "--clip-checkpoint",
        default=None,
        help="Optional local checkpoint path passed to clip.load instead of --clip-model",
    )
    parser.add_argument("--device", default="cpu", help="Device for clip.load (cpu or cuda)")
    parser.add_argument(
        "--resample",
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        help="Resize interpolation method.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output image if it already exists.",
    )
    return parser.parse_args()


def load_relative_paths(json_path: Path) -> List[str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON root must be a dict/object")

    paths: List[str] = []

    if "correct_samples" in data:
        samples = data.get("correct_samples")
        if not isinstance(samples, list):
            raise ValueError("correct_samples must be a list")
        for item in samples:
            if not isinstance(item, dict):
                continue
            rel = item.get("image_path")
            if isinstance(rel, str) and rel.strip():
                paths.append(rel.strip().replace("\\", "/"))
        return paths

    for value in data.values():
        if isinstance(value, str) and value.strip():
            paths.append(value.strip().replace("\\", "/"))

    return paths


def get_resample_method(name: str) -> int:
    mapping: Dict[str, int] = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return mapping[name]


def _parse_hw_from_size(size_obj: object) -> Optional[Tuple[int, int]]:
    if isinstance(size_obj, int):
        return int(size_obj), int(size_obj)

    if isinstance(size_obj, (tuple, list)):
        if len(size_obj) == 1:
            return int(size_obj[0]), int(size_obj[0])
        if len(size_obj) >= 2:
            # torchvision convention is (height, width)
            return int(size_obj[0]), int(size_obj[1])

    return None


def infer_size_from_clip_preprocess(args: argparse.Namespace) -> Tuple[int, int]:
    clip_source = args.clip_checkpoint or args.clip_model
    if args.clip_checkpoint:
        ckpt = Path(args.clip_checkpoint).expanduser().resolve()
        if not ckpt.exists() or not ckpt.is_file():
            raise FileNotFoundError(f"clip checkpoint not found: {ckpt}")

    _, preprocess = clip.load(clip_source, device=args.device)
    transforms = getattr(preprocess, "transforms", [])

    inferred_hw: Optional[Tuple[int, int]] = None
    for transform in transforms:
        name = transform.__class__.__name__.lower()
        if "resize" in name or "crop" in name:
            size_obj = getattr(transform, "size", None)
            parsed = _parse_hw_from_size(size_obj)
            if parsed is not None:
                inferred_hw = parsed

    if inferred_hw is None:
        raise ValueError("Could not infer target size from CLIP preprocess transforms")

    return inferred_hw


def resolve_resize_hw(args: argparse.Namespace) -> Tuple[int, int]:
    if args.size_source == "clip_preprocess":
        return infer_size_from_clip_preprocess(args)

    if args.resize_width is None or args.resize_height is None:
        raise ValueError("--resize-width and --resize-height are required when --size-source manual")
    if args.resize_width < 1 or args.resize_height < 1:
        raise ValueError("resize width/height must be >= 1")
    return int(args.resize_height), int(args.resize_width)


def main() -> None:
    args = parse_args()

    input_json = Path(args.input_json).expanduser().resolve()
    images_root = Path(args.images_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not input_json.exists() or not input_json.is_file():
        raise FileNotFoundError(f"input json not found: {input_json}")
    if not images_root.exists() or not images_root.is_dir():
        raise FileNotFoundError(f"images root not found or not a directory: {images_root}")

    resize_h, resize_w = resolve_resize_hw(args)

    rel_paths = load_relative_paths(input_json)
    if not rel_paths:
        raise ValueError("No image paths found in input JSON")

    # Keep only unique paths while preserving order.
    seen: Set[str] = set()
    unique_rel_paths: List[str] = []
    for rel in rel_paths:
        if rel not in seen:
            seen.add(rel)
            unique_rel_paths.append(rel)

    output_root.mkdir(parents=True, exist_ok=True)
    resample = get_resample_method(args.resample)

    total = len(unique_rel_paths)
    saved = 0
    skipped_existing = 0
    missing = 0
    errors = 0

    for rel in unique_rel_paths:
        src = images_root / rel
        dst = output_root / rel

        if not src.exists() or not src.is_file():
            missing += 1
            print(f"[MISSING] {src}")
            continue

        if dst.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            image = Image.open(src).convert("RGB")
            resized = image.resize((resize_w, resize_h), resample)
            resized.save(dst)
            saved += 1
        except Exception as exc:  # pylint: disable=broad-except
            errors += 1
            print(f"[ERROR] {src}: {exc}")

    print("=" * 80)
    print(f"Input JSON: {input_json}")
    print(f"Images root: {images_root}")
    print(f"Output root: {output_root}")
    print(f"Resize source: {args.size_source}")
    print(f"Resize: {resize_w}x{resize_h} ({args.resample})")
    print(f"Unique JSON paths: {total}")
    print(f"Saved: {saved}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Missing source: {missing}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
