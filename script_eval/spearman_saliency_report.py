import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from CLIP import clip
except ImportError:
    import clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Spearman correlation between clean saliency maps and attack saliency maps "
            "for each dataset, attack method, and explainer."
        )
    )
    parser.add_argument(
        "--result-root",
        type=str,
        required=True,
        help="Root directory containing dataset folders (e.g., D:/neurips_result/result)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="CLIP_ViTB16",
        help="Model folder name under each dataset (default: CLIP_ViTB16)",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default=None,
        help="Optional CLIP model name for clip.load. Default: resolved from --model-name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for CLIP inference (cuda/cpu). Default: auto.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names. Default: all dataset folders under result root.",
    )
    parser.add_argument(
        "--clean-folder",
        type=str,
        default="clean_causual_score",
        help="Folder containing clean saliency maps.",
    )
    parser.add_argument(
        "--attack-folders",
        nargs="*",
        default=["Xshift_score", "ins_optimize_output", "del_optimize_output"],
        help="Attack folders to compare against clean (PGD excluded by default).",
    )
    parser.add_argument(
        "--explainers",
        nargs="*",
        default=["eclip", "game", "gradcam", "maskclip", "rise"],
        help="Explainer folders to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <result-root>/saliency_spearman_report",
    )
    parser.add_argument(
        "--text-template",
        type=str,
        default="{}",
        help="Template for class text before CLIP tokenization (default: '{}').",
    )
    return parser.parse_args()


def resolve_clip_model_name(model_name: str) -> str:
    if model_name in clip.available_models():
        return model_name

    raw = model_name.strip()
    raw = re.sub(r"^clip[_-]?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("_", "")

    m = re.match(r"(?i)^vit([bhl])([0-9]{1,2})$", raw)
    if m:
        size = m.group(1).upper()
        patch = m.group(2)
        candidate = f"ViT-{size}/{patch}"
        if candidate in clip.available_models():
            return candidate

    candidates = [
        "ViT-B/16",
        "ViT-B/32",
        "ViT-L/14",
    ]
    for candidate in candidates:
        key = candidate.lower().replace("-", "").replace("/", "")
        if key in raw.lower():
            return candidate

    return "ViT-B/16"


def get_preprocess_normalization_stats(preprocess) -> Tuple[np.ndarray, np.ndarray]:
    transforms = getattr(preprocess, "transforms", [])
    for transform in transforms:
        if hasattr(transform, "mean") and hasattr(transform, "std"):
            mean = np.asarray(transform.mean, dtype=np.float32)
            std = np.asarray(transform.std, dtype=np.float32)
            return mean, std
    raise ValueError("Could not infer normalization mean/std from preprocess")


def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)

    sorted_x = x[order]
    i = 0
    while i < sorted_x.size:
        j = i + 1
        while j < sorted_x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata_average(a)
    rb = rankdata_average(b)

    ra -= ra.mean()
    rb -= rb.mean()

    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom <= 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def find_saliency_npy(sample_dir: Path, explainer: str) -> Optional[Path]:
    direct = sample_dir / "saliency" / f"{explainer}_saliency.npy"
    if direct.exists():
        return direct

    candidates = sorted(sample_dir.rglob("*saliency.npy"))
    if not candidates:
        return None

    for p in candidates:
        name = p.name.lower()
        if f"{explainer.lower()}_saliency.npy" in name:
            return p

    for p in candidates:
        if explainer.lower() in p.name.lower():
            return p

    return candidates[0]


def find_curve_information_json(sample_dir: Path) -> Optional[Path]:
    direct = sample_dir / "curve_information.json"
    if direct.exists():
        return direct

    candidates = sorted(sample_dir.glob("*information.json"))
    if candidates:
        return candidates[0]
    return None


def find_adversarial_image_path(sample_dir: Path) -> Optional[Path]:
    for p in sorted(sample_dir.glob("adversarial_image*.png")):
        return p
    for p in sorted(sample_dir.glob("*adv*.png")):
        return p
    for p in sorted(sample_dir.glob("*.png")):
        name = p.name.lower()
        if "overlay" in name or "summary" in name or "saliency" in name:
            continue
        return p
    return None


def collect_clean_maps(clean_explainer_root: Path, explainer: str) -> Dict[str, Path]:
    clean_maps: Dict[str, Path] = {}
    for class_dir in sorted(clean_explainer_root.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name.startswith("eval_curve_summary"):
            continue

        npy_path = find_saliency_npy(class_dir, explainer)
        if npy_path is not None:
            clean_maps[class_dir.name] = npy_path
    return clean_maps


def _score_attack_map_name(path: Path, explainer: str) -> Tuple[int, str]:
    name = path.name.lower()
    score = 0
    if explainer.lower() in name:
        score += 10
    if "del_" in name:
        score += 2
    if "ins_" in name:
        score += 1
    return -score, name


def collect_attack_maps(attack_explainer_root: Path, explainer: str) -> Dict[str, Tuple[Path, Path]]:
    per_sample: Dict[str, List[Path]] = {}
    sample_dirs: Dict[str, Path] = {}
    for npy_path in sorted(attack_explainer_root.rglob("*saliency.npy")):
        # Use nearest non-saliency parent folder as sample ID.
        sample_dir = npy_path.parent
        if sample_dir.name == "saliency":
            sample_dir = sample_dir.parent

        sample_id = sample_dir.name
        if sample_id.startswith("eval_curve_summary"):
            continue

        per_sample.setdefault(sample_id, []).append(npy_path)
        sample_dirs[sample_id] = sample_dir

    attack_maps: Dict[str, Tuple[Path, Path]] = {}
    for sample_id, candidates in per_sample.items():
        # Del/ins attack maps are equivalent in this setup, so choose one deterministic representative.
        chosen = sorted(candidates, key=lambda p: _score_attack_map_name(p, explainer))[0]
        attack_maps[sample_id] = (chosen, sample_dirs[sample_id])

    return attack_maps


def read_flat_array(path: Path) -> Optional[np.ndarray]:
    try:
        arr = np.load(path)
    except Exception:
        return None

    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return None
    return arr.reshape(-1)


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def discover_datasets(result_root: Path, selected: Optional[List[str]]) -> List[str]:
    if selected:
        return selected
    return sorted([p.name for p in result_root.iterdir() if p.is_dir()])


def extract_class_text(sample_dir: Path, text_template: str) -> Optional[str]:
    info_path = find_curve_information_json(sample_dir)
    data = None
    if info_path is not None:
        try:
            with info_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None

    if isinstance(data, dict):
        for key in ["target_classname", "pred_classname_adv", "pred_classname_clean", "class_name", "target_text"]:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return text_template.format(value.strip())

    sample_id = sample_dir.name
    if sample_id and sample_id[0].isalpha() and sample_id[1:].isdigit():
        return None
    return text_template.format(sample_id)


def preprocess_image_no_spatial(image: Image.Image, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=device).view(1, 3, 1, 1)
    return (tensor - mean_t) / std_t


def compute_adv_text_cosine(
    clip_model,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    image_path: Path,
    text: str,
    text_feature_cache: Dict[str, torch.Tensor],
) -> Optional[float]:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    image_tensor = preprocess_image_no_spatial(image, mean, std, device)
    with torch.no_grad():
        image_feat = clip_model.encode_image(image_tensor)
        image_feat = F.normalize(image_feat, dim=-1)

        if text not in text_feature_cache:
            tokens = clip.tokenize([text]).to(device)
            text_feat = clip_model.encode_text(tokens)
            text_feature_cache[text] = F.normalize(text_feat, dim=-1)
        text_feat = text_feature_cache[text]

        cosine = torch.sum(image_feat * text_feat, dim=-1).item()
    return float(cosine)


def main() -> None:
    args = parse_args()

    result_root = Path(args.result_root).expanduser().resolve()
    if not result_root.exists() or not result_root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {result_root}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_root / "saliency_spearman_report"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(result_root, args.datasets)
    if not datasets:
        raise RuntimeError("No dataset folders found.")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    resolved_clip_model = args.clip_model if args.clip_model else resolve_clip_model_name(args.model_name)
    clip_model, preprocess = clip.load(resolved_clip_model, device=device)
    clip_model.eval()
    norm_mean, norm_std = get_preprocess_normalization_stats(preprocess)
    text_feature_cache: Dict[str, torch.Tensor] = {}

    report: Dict[str, object] = {
        "result_root": str(result_root),
        "model_name": args.model_name,
        "resolved_clip_model": resolved_clip_model,
        "device": str(device),
        "text_template": args.text_template,
        "clean_folder": args.clean_folder,
        "attack_folders": args.attack_folders,
        "excluded_attacks": ["PGD_score"],
        "datasets": {},
    }

    csv_rows: List[Dict[str, object]] = []

    for dataset in datasets:
        dataset_root = result_root / dataset / args.model_name
        if not dataset_root.exists() or not dataset_root.is_dir():
            print(f"[SKIP] dataset={dataset}: missing model folder")
            continue

        clean_root = dataset_root / args.clean_folder
        if not clean_root.exists() or not clean_root.is_dir():
            print(f"[SKIP] dataset={dataset}: missing clean folder {args.clean_folder}")
            continue

        dataset_entry: Dict[str, object] = {}

        for attack in args.attack_folders:
            if attack.lower() == "pgd_score":
                continue

            attack_root = dataset_root / attack
            if not attack_root.exists() or not attack_root.is_dir():
                print(f"[SKIP] dataset={dataset}, attack={attack}: not found")
                continue

            attack_entry: Dict[str, object] = {}

            for explainer in args.explainers:
                clean_explainer_root = clean_root / explainer
                attack_explainer_root = attack_root / explainer

                if not clean_explainer_root.exists() or not clean_explainer_root.is_dir():
                    print(f"[SKIP] dataset={dataset}, attack={attack}, explain={explainer}: clean explainer missing")
                    continue
                if not attack_explainer_root.exists() or not attack_explainer_root.is_dir():
                    print(
                        f"[SKIP] dataset={dataset}, attack={attack}, explain={explainer}: attack explainer missing"
                    )
                    continue

                clean_maps = collect_clean_maps(clean_explainer_root, explainer)
                attack_maps = collect_attack_maps(attack_explainer_root, explainer)

                common_ids = sorted(set(clean_maps.keys()).intersection(attack_maps.keys()))
                if not common_ids:
                    print(f"[SKIP] dataset={dataset}, attack={attack}, explain={explainer}: no matched samples")
                    continue

                corrs: List[float] = []
                cosines: List[float] = []
                for sample_id in common_ids:
                    clean_arr = read_flat_array(clean_maps[sample_id])
                    attack_map_path, attack_sample_dir = attack_maps[sample_id]
                    attack_arr = read_flat_array(attack_map_path)
                    if clean_arr is None or attack_arr is None:
                        continue

                    n = min(clean_arr.size, attack_arr.size)
                    if n <= 1:
                        continue

                    rho = spearman_corr(clean_arr[:n], attack_arr[:n])
                    if np.isnan(rho):
                        continue
                    corrs.append(rho)

                    class_text = extract_class_text(attack_sample_dir, args.text_template)
                    adv_image_path = find_adversarial_image_path(attack_sample_dir)
                    if class_text is None or adv_image_path is None:
                        continue

                    cosine = compute_adv_text_cosine(
                        clip_model=clip_model,
                        device=device,
                        mean=norm_mean,
                        std=norm_std,
                        image_path=adv_image_path,
                        text=class_text,
                        text_feature_cache=text_feature_cache,
                    )
                    if cosine is not None and not np.isnan(cosine):
                        cosines.append(cosine)

                if not corrs:
                    print(f"[SKIP] dataset={dataset}, attack={attack}, explain={explainer}: no valid correlations")
                    continue

                stats = summarize(corrs)
                cosine_stats = summarize(cosines) if cosines else None
                attack_entry[explainer] = {
                    "num_clean_samples": int(len(clean_maps)),
                    "num_attack_samples": int(len(attack_maps)),
                    "num_matched_samples": int(len(common_ids)),
                    "num_valid_correlations": int(len(corrs)),
                    "spearman": stats,
                    "num_valid_adv_text_cosine": int(len(cosines)),
                    "adv_text_cosine": cosine_stats,
                }

                csv_rows.append(
                    {
                        "dataset": dataset,
                        "attack": attack,
                        "explainer": explainer,
                        "num_clean_samples": int(len(clean_maps)),
                        "num_attack_samples": int(len(attack_maps)),
                        "num_matched_samples": int(len(common_ids)),
                        "num_valid_correlations": int(len(corrs)),
                        "num_valid_adv_text_cosine": int(len(cosines)),
                        "spearman_mean": stats["mean"],
                        "spearman_std": stats["std"],
                        "spearman_median": stats["median"],
                        "spearman_min": stats["min"],
                        "spearman_max": stats["max"],
                        "adv_text_cosine_mean": cosine_stats["mean"] if cosine_stats else float("nan"),
                        "adv_text_cosine_std": cosine_stats["std"] if cosine_stats else float("nan"),
                        "adv_text_cosine_median": cosine_stats["median"] if cosine_stats else float("nan"),
                        "adv_text_cosine_min": cosine_stats["min"] if cosine_stats else float("nan"),
                        "adv_text_cosine_max": cosine_stats["max"] if cosine_stats else float("nan"),
                    }
                )

                print(
                    f"[OK] dataset={dataset}, attack={attack}, explain={explainer}, "
                    f"matched={len(common_ids)}, spearman_n={len(corrs)}, cosine_n={len(cosines)}, "
                    f"spearman_mean={stats['mean']:.4f}"
                )

            if attack_entry:
                dataset_entry[attack] = attack_entry

        if dataset_entry:
            report["datasets"][dataset] = dataset_entry

    json_path = output_dir / "saliency_spearman_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "saliency_spearman_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "attack",
                "explainer",
                "num_clean_samples",
                "num_attack_samples",
                "num_matched_samples",
                "num_valid_correlations",
                "num_valid_adv_text_cosine",
                "spearman_mean",
                "spearman_std",
                "spearman_median",
                "spearman_min",
                "spearman_max",
                "adv_text_cosine_mean",
                "adv_text_cosine_std",
                "adv_text_cosine_median",
                "adv_text_cosine_min",
                "adv_text_cosine_max",
            ],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print("=" * 80)
    print(f"Saved JSON summary: {json_path}")
    print(f"Saved CSV summary : {csv_path}")


if __name__ == "__main__":
    main()
