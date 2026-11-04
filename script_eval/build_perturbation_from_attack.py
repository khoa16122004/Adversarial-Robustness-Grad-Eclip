import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import cv2


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SKIP_NAME_TOKENS = (
    "overlay",
    "summary",
    "saliency",
    "perturbation",
    "clean",
    "original",
)


@dataclass
class AdvSample:
    attack_root: Path
    sample_dir: Path
    adv_image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build perturbation visualization images for each adversarial sample folder. "
            "For each sample: adv image - clean image (from classification result JSON)."
        )
    )
    parser.add_argument(
        "--classification-json",
        required=True,
        help=(
            "Path to classification JSON. Supports either: "
            "(1) plain mapping {class_key: relative_image_path}, or "
            "(2) report JSON containing correct_samples."
        ),
    )
    parser.add_argument(
        "--clean-images-root",
        required=True,
        help="Root folder of clean/original images used by the classification JSON relative paths.",
    )
    parser.add_argument(
        "--model-root",
        default=None,
        help=(
            "Model result folder containing attack method subfolders. "
            "Example: D:/neurips_result/result/ImageNet/CLIP_ViTB16"
        ),
    )
    parser.add_argument(
        "--attack-folders",
        nargs="*",
        default=["Xshift_score", "ins_optimize_output", "del_optimize_output"],
        help="Attack folder names under --model-root.",
    )
    parser.add_argument(
        "--attack-roots",
        nargs="*",
        default=None,
        help="Optional explicit attack root folders. If provided, overrides --model-root + --attack-folders.",
    )
    parser.add_argument(
        "--adv-filename-contains",
        default="",
        help="Optional substring filter for adversarial image filename (case-insensitive).",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Save raw perturbation array to perturbation.npy inside each sample folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing perturbation outputs.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="Blend factor for perturbation overlay in [0, 1].",
    )
    return parser.parse_args()


def load_class_to_relpath(json_path: Path) -> Dict[str, str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("classification JSON must be an object/dict")

    if "correct_samples" in data:
        samples = data.get("correct_samples")
        if not isinstance(samples, list):
            raise ValueError("correct_samples must be a list")

        mapping: Dict[str, str] = {}
        for item in samples:
            if not isinstance(item, dict):
                continue
            wnid = item.get("wnid")
            rel = item.get("image_path")
            if isinstance(wnid, str) and wnid.strip() and isinstance(rel, str) and rel.strip():
                mapping[wnid.strip()] = rel.strip().replace("\\", "/")
        return mapping

    mapping = {}
    for k, v in data.items():
        if isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip():
            mapping[k.strip()] = v.strip().replace("\\", "/")
    return mapping


def resolve_attack_roots(args: argparse.Namespace) -> List[Path]:
    if args.attack_roots:
        roots = [Path(p).expanduser().resolve() for p in args.attack_roots]
        return roots

    if not args.model_root:
        raise ValueError("Provide either --attack-roots or --model-root")

    model_root = Path(args.model_root).expanduser().resolve()
    roots = [model_root / folder for folder in args.attack_folders]
    return roots


def score_candidate_image(path: Path, needle: str) -> int:
    name = path.name.lower()
    score = 0

    if "adversarial_image" in name:
        score += 100
    if "adv" in name:
        score += 60

    if needle and needle in name:
        score += 30

    for token in SKIP_NAME_TOKENS:
        if token in name:
            score -= 80

    return score


def sample_dir_from_image(image_path: Path) -> Path:
    sample_dir = image_path.parent
    if sample_dir.name.lower() == "saliency":
        sample_dir = sample_dir.parent
    return sample_dir


def collect_adv_samples(attack_root: Path, needle: str) -> List[AdvSample]:
    if not attack_root.exists() or not attack_root.is_dir():
        return []

    best_by_sample_dir: Dict[Path, Tuple[int, Path]] = {}

    for path in sorted(attack_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        score = score_candidate_image(path, needle)
        if score <= 0:
            continue

        sample_dir = sample_dir_from_image(path)
        prev = best_by_sample_dir.get(sample_dir)
        if prev is None or score > prev[0] or (score == prev[0] and str(path) < str(prev[1])):
            best_by_sample_dir[sample_dir] = (score, path)

    items = [
        AdvSample(attack_root=attack_root, sample_dir=sample_dir, adv_image_path=best_path)
        for sample_dir, (_, best_path) in sorted(best_by_sample_dir.items(), key=lambda x: str(x[0]))
    ]
    return items


def infer_class_key(sample_dir: Path, attack_root: Path, valid_keys: Iterable[str]) -> Optional[str]:
    valid_set = set(valid_keys)

    # Prefer nearest path parts first: sample folder name, then ancestors.
    for part in reversed(sample_dir.parts):
        if part in valid_set:
            return part

    # Also inspect relative path parts from attack root.
    try:
        rel_parts = sample_dir.relative_to(attack_root).parts
    except Exception:
        rel_parts = sample_dir.parts

    for part in rel_parts:
        if part in valid_set:
            return part

    return None


def load_rgb_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32)


def _normalize_saliency_map(saliency_map: torch.Tensor) -> torch.Tensor:
    map_2d = saliency_map.detach().float().cpu().squeeze()
    map_min = map_2d.min()
    map_max = map_2d.max()
    den = (map_max - map_min).item()
    if den > 1e-12:
        map_2d = (map_2d - map_min) / (map_max - map_min)
    else:
        map_2d = torch.zeros_like(map_2d)
    return map_2d


def _save_saliency_map(saliency_map: torch.Tensor, output_path: Path) -> torch.Tensor:
    # Save a single saliency map with standard normalization and OpenCV JET colormap.
    norm_map = _normalize_saliency_map(saliency_map)
    heatmap_uint8 = (norm_map * 255.0).to(dtype=torch.uint8).cpu().numpy()
    heat_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heat_u8 = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(heat_u8).save(output_path)
    return norm_map


def perturbation_to_saliency(delta: torch.Tensor) -> torch.Tensor:
    # Match requested formulation with batch dimension:
    # saliency = grad.abs().sum(dim=1)
    # saliency = (H*W) * saliency / saliency.sum()
    grad = delta.unsqueeze(0)  # 1,C,H,W
    saliency = grad.abs().sum(dim=1)  # 1,H,W
    h = int(saliency.shape[-2])
    w = int(saliency.shape[-1])
    denom = saliency.view(saliency.size(0), -1).sum(dim=1).view(-1, 1, 1)
    saliency = (h * w) * saliency / (denom + 1e-8)
    return saliency[0]


def perturbation_to_output(
    delta: torch.Tensor,
    base_image: torch.Tensor,
    perturbation_path: Path,
    overlay_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    saliency = perturbation_to_saliency(delta)
    norm = _save_saliency_map(saliency, perturbation_path)
    heatmap_uint8 = (norm * 255.0).to(dtype=torch.uint8).cpu().numpy()
    heat_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    base = base_image.permute(1, 2, 0).clamp(0.0, 255.0).cpu().numpy().astype(np.float32)
    alpha = float(overlay_alpha)
    overlay_u8 = np.clip(base * (1.0 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)

    return overlay_u8, saliency.cpu().numpy()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.overlay_alpha <= 1.0):
        raise ValueError("--overlay-alpha must be in [0, 1]")

    class_json = Path(args.classification_json).expanduser().resolve()
    clean_root = Path(args.clean_images_root).expanduser().resolve()
    attack_roots = resolve_attack_roots(args)

    if not class_json.exists():
        raise FileNotFoundError(f"classification json not found: {class_json}")
    if not clean_root.exists() or not clean_root.is_dir():
        raise FileNotFoundError(f"clean image root not found or not a directory: {clean_root}")

    class_to_relpath = load_class_to_relpath(class_json)
    if not class_to_relpath:
        raise ValueError("No class->image mapping could be loaded from classification JSON")

    needle = args.adv_filename_contains.lower().strip()

    total_attacks = 0
    total_samples = 0
    total_saved = 0
    total_skipped_missing_class = 0
    total_skipped_missing_clean = 0
    total_skipped_size_mismatch = 0
    total_skipped_existing = 0
    total_errors = 0

    for attack_root in attack_roots:
        if not attack_root.exists() or not attack_root.is_dir():
            print(f"[SKIP] Attack root not found: {attack_root}")
            continue

        total_attacks += 1
        adv_samples = collect_adv_samples(attack_root, needle)
        if not adv_samples:
            print(f"[WARN] No adversarial-like images found under: {attack_root}")
            continue

        saved_this_attack = 0
        skipped_this_attack = 0

        for item in adv_samples:
            total_samples += 1

            class_key = infer_class_key(item.sample_dir, attack_root, class_to_relpath.keys())
            if class_key is None:
                total_skipped_missing_class += 1
                skipped_this_attack += 1
                continue

            clean_rel = class_to_relpath.get(class_key)
            if not clean_rel:
                total_skipped_missing_class += 1
                skipped_this_attack += 1
                continue

            clean_path = clean_root / clean_rel
            if not clean_path.exists() or not clean_path.is_file():
                total_skipped_missing_clean += 1
                skipped_this_attack += 1
                continue

            perturbation_path = item.sample_dir / "perturbation.png"
            overlay_path = item.sample_dir / "perturbation_overlay.png"
            npy_path = item.sample_dir / "perturbation.npy"

            if (
                not args.overwrite
                and perturbation_path.exists()
                and overlay_path.exists()
                and npy_path.exists()
            ):
                total_skipped_existing += 1
                skipped_this_attack += 1
                continue

            try:
                adv = load_rgb_tensor(item.adv_image_path)
                clean = load_rgb_tensor(clean_path)
                if adv.shape != clean.shape:
                    total_skipped_size_mismatch += 1
                    skipped_this_attack += 1
                    continue

                delta = adv - clean
                perturbation_overlay, saliency_npy = perturbation_to_output(
                    delta=delta,
                    base_image=clean,
                    perturbation_path=perturbation_path,
                    overlay_alpha=args.overlay_alpha,
                )

                ensure_parent(perturbation_path)
                Image.fromarray(perturbation_overlay).save(overlay_path)
                np.save(str(npy_path), saliency_npy)

                saved_this_attack += 1
                total_saved += 1
            except Exception as exc:  # pylint: disable=broad-except
                total_errors += 1
                skipped_this_attack += 1
                print(f"[ERROR] {item.adv_image_path}: {exc}")

        print(
            f"[DONE] {attack_root.name}: "
            f"found={len(adv_samples)} saved={saved_this_attack} skipped={skipped_this_attack}"
        )

    print("=" * 80)
    print(f"Attack roots processed: {total_attacks}")
    print(f"Sample folders scanned: {total_samples}")
    print(f"Perturbation outputs saved: {total_saved}")
    print(f"Skipped (class not resolved): {total_skipped_missing_class}")
    print(f"Skipped (clean image missing): {total_skipped_missing_clean}")
    print(f"Skipped (size mismatch): {total_skipped_size_mismatch}")
    print(f"Skipped (already exists): {total_skipped_existing}")
    print(f"Errors: {total_errors}")


if __name__ == "__main__":
    main()
