import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export progressive saliency masks and corresponding insertion/deletion images "
            "for one sample folder."
        )
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        required=True,
        help="Path to one sample folder (contains saliency map and clean image)",
    )
    parser.add_argument(
        "--saliency-path",
        type=str,
        default=None,
        help="Optional explicit saliency file (.npy/.png/.jpg). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--clean-image-path",
        type=str,
        default=None,
        help="Optional explicit clean image path. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of progressive masks from 0%% to 100%% (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <sample-dir>/ins_del_masks",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="step",
        help="Filename prefix for saved masks/images",
    )
    return parser.parse_args()


def _pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def _find_saliency_path(sample_dir: Path) -> Path:
    candidates = [
        sample_dir / "saliency" / "eclip_saliency.npy",
        sample_dir / "saliency" / "gradcam_saliency.npy",
        sample_dir / "saliency" / "maskclip_saliency.npy",
        sample_dir / "saliency" / "game_saliency.npy",
        sample_dir / "saliency" / "rise_saliency.npy",
    ]
    direct = _pick_first_existing(candidates)
    if direct is not None:
        return direct

    npy_hits = sorted(sample_dir.glob("**/*saliency*.npy"))
    if npy_hits:
        return npy_hits[0]

    img_hits = sorted(sample_dir.glob("**/*saliency*.png"))
    if img_hits:
        return img_hits[0]

    img_hits = sorted(sample_dir.glob("**/*saliency*.jpg"))
    if img_hits:
        return img_hits[0]

    raise FileNotFoundError(
        f"Could not find saliency file under: {sample_dir}. "
        "Expected something like saliency/*_saliency.npy"
    )


def _find_clean_image_path(sample_dir: Path) -> Path:
    candidates = [
        sample_dir / "clean_image_resized.png",
        sample_dir / "clean_image.png",
        sample_dir / "original_image.png",
    ]
    direct = _pick_first_existing(candidates)
    if direct is not None:
        return direct

    png_hits = sorted(sample_dir.glob("clean_image*.png"))
    if png_hits:
        return png_hits[0]

    png_hits = sorted(sample_dir.glob("original*.png"))
    if png_hits:
        return png_hits[0]

    raise FileNotFoundError(
        f"Could not find clean image under: {sample_dir}. "
        "Expected clean_image_resized.png or clean_image*.png"
    )


def _load_saliency(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    else:
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Saliency must be 2D or RGB-like 3D. Got shape: {arr.shape}")

    arr = arr - float(arr.min())
    max_val = float(arr.max())
    if max_val > 0:
        arr = arr / max_val
    return arr


def _load_clean_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _resize_saliency_to_image(saliency: np.ndarray, image_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    if saliency.shape == (h, w):
        return saliency
    resized = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def _build_mask(order: np.ndarray, num_pixels: int, k: int, h: int, w: int) -> np.ndarray:
    k = int(max(0, min(num_pixels, k)))
    mask = np.zeros(num_pixels, dtype=np.float32)
    if k > 0:
        mask[order[:k]] = 1.0
    return mask.reshape(h, w)


def main() -> None:
    args = parse_args()

    sample_dir = Path(args.sample_dir).expanduser().resolve()
    if not sample_dir.exists() or not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    if args.steps < 2:
        raise ValueError("--steps must be >= 2 to include start and end states")

    saliency_path = (
        Path(args.saliency_path).expanduser().resolve()
        if args.saliency_path
        else _find_saliency_path(sample_dir)
    )
    clean_image_path = (
        Path(args.clean_image_path).expanduser().resolve()
        if args.clean_image_path
        else _find_clean_image_path(sample_dir)
    )

    if not saliency_path.exists():
        raise FileNotFoundError(f"Saliency path not found: {saliency_path}")
    if not clean_image_path.exists():
        raise FileNotFoundError(f"Clean image path not found: {clean_image_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else sample_dir / "ins_del_masks"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = output_dir / "masks"
    insertion_dir = output_dir / "insertion_images"
    deletion_dir = output_dir / "deletion_images"
    masks_dir.mkdir(parents=True, exist_ok=True)
    insertion_dir.mkdir(parents=True, exist_ok=True)
    deletion_dir.mkdir(parents=True, exist_ok=True)

    clean_image = _load_clean_image(clean_image_path)
    saliency = _load_saliency(saliency_path)
    saliency = _resize_saliency_to_image(saliency, clean_image.shape)

    h, w, _ = clean_image.shape
    num_pixels = h * w
    order = np.argsort(-saliency.reshape(-1))

    ratios = np.linspace(0.0, 1.0, args.steps, dtype=np.float64)
    steps_manifest = []

    for idx, ratio in enumerate(ratios):
        k = int(round(float(ratio) * num_pixels))
        mask = _build_mask(order, num_pixels, k, h, w)

        mask_u8 = (mask * 255.0).astype(np.uint8)

        insertion = (clean_image.astype(np.float32) * mask[:, :, None]).astype(np.uint8)
        deletion = (clean_image.astype(np.float32) * (1.0 - mask[:, :, None])).astype(np.uint8)

        stem = f"{args.prefix}_{idx:02d}"
        mask_png = masks_dir / f"{stem}_mask.png"
        mask_npy = masks_dir / f"{stem}_mask.npy"
        ins_png = insertion_dir / f"{stem}_ins.png"
        del_png = deletion_dir / f"{stem}_del.png"

        cv2.imwrite(str(mask_png), mask_u8)
        np.save(mask_npy, mask.astype(np.float32))
        cv2.imwrite(str(ins_png), cv2.cvtColor(insertion, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(del_png), cv2.cvtColor(deletion, cv2.COLOR_RGB2BGR))

        steps_manifest.append(
            {
                "step_index": idx,
                "ratio": float(ratio),
                "num_revealed_pixels": int(k),
                "num_total_pixels": int(num_pixels),
                "mask_png": str(mask_png),
                "mask_npy": str(mask_npy),
                "insertion_image": str(ins_png),
                "deletion_image": str(del_png),
            }
        )

    manifest = {
        "sample_dir": str(sample_dir),
        "clean_image_path": str(clean_image_path),
        "saliency_path": str(saliency_path),
        "output_dir": str(output_dir),
        "steps": int(args.steps),
        "mask_definition": "M_t is 1 for top-saliency pixels and 0 elsewhere.",
        "insertion_definition": "M_t * clean_image",
        "deletion_definition": "(1 - M_t) * clean_image",
        "items": steps_manifest,
    }

    manifest_path = output_dir / "mask_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "sample_dir": str(sample_dir),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "steps": int(args.steps),
        "clean_image": str(clean_image_path),
        "saliency": str(saliency_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
