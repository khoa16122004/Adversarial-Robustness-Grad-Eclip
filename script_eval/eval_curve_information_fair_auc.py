import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate curve files with fair AUC: recompute AUC from curve normalized by its own max value."
        )
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Root folder to recursively search",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs. Default: <input-root>/eval_curve_summary_fair_auc",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="curve_information.json",
        help="Filename pattern for recursive search (default: curve_information.json)",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric names to include (e.g., --metrics del ins)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)",
    )
    parser.add_argument(
        "--save-per-folder-summary",
        action="store_true",
        help=(
            "Save normalized summary plot and normalized information json next to each source file"
        ),
    )
    parser.add_argument(
        "--per-folder-image-name",
        type=str,
        default="{metric}_summary_normalized.png",
        help="Output image filename template for each folder/file. Supports {metric}",
    )
    parser.add_argument(
        "--per-folder-json-name",
        type=str,
        default="{metric}_information_normalized.json",
        help="Output json filename template for each folder/file. Supports {metric}",
    )
    return parser.parse_args()


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_auc_from_curve(curve: np.ndarray) -> float:
    if curve.size == 0:
        return float("nan")
    x = np.linspace(0.0, 1.0, curve.size, dtype=np.float64)
    return float(np.trapezoid(curve.astype(np.float64), x))


def summarize_values(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def extract_metrics(
    data: dict, selected_metrics: Optional[set]
) -> Dict[str, Optional[np.ndarray]]:
    extracted: Dict[str, Optional[np.ndarray]] = {}

    rerun = data.get("rerun")
    if isinstance(rerun, dict):
        for metric_name, metric_obj in rerun.items():
            if selected_metrics is not None and metric_name not in selected_metrics:
                continue
            if not isinstance(metric_obj, dict):
                continue

            curve = metric_obj.get("curve")
            if isinstance(curve, list) and len(curve) > 0:
                extracted[metric_name] = np.asarray(curve, dtype=np.float32)

    clean_score_keys = {
        "del": "deletion_curve",
        "ins": "insertion_curve",
    }
    for metric_name, curve_key in clean_score_keys.items():
        if metric_name in extracted:
            continue
        if selected_metrics is not None and metric_name not in selected_metrics:
            continue

        curve = data.get(curve_key)
        if isinstance(curve, list) and len(curve) > 0:
            extracted[metric_name] = np.asarray(curve, dtype=np.float32)

    return extracted


def align_curves(curves: List[np.ndarray]) -> Tuple[np.ndarray, int, int, int]:
    lengths = [c.size for c in curves]
    min_len = min(lengths)
    max_len = max(lengths)
    truncated_count = sum(1 for ln in lengths if ln != min_len)

    aligned = np.stack([c[:min_len] for c in curves], axis=0)
    return aligned, min_len, max_len, truncated_count


def normalize_curve_by_max(curve: np.ndarray) -> Optional[np.ndarray]:
    if curve.size == 0:
        return None
    max_val = float(np.max(curve))
    if max_val <= 0.0:
        return None
    return curve.astype(np.float64) / max_val


def save_normalized_curve_summary_plot(
    normalized_curve: np.ndarray,
    fair_auc: float,
    metric_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    x = np.linspace(0.0, 1.0, normalized_curve.size, dtype=np.float64)

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, normalized_curve, linewidth=2.0)
    plt.fill_between(x, 0.0, normalized_curve, alpha=0.25)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_normalized_curve_information(
    normalized_curve: np.ndarray,
    fair_auc: float,
    metric_name: str,
    source_file: Path,
    output_path: Path,
) -> None:
    payload = {
        "source_file": str(source_file),
        "metric": metric_name,
        "normalization": "curve / max(curve)",
        "fair_auc": float(fair_auc),
        "normalized_curve": normalized_curve.tolist(),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist or is not a directory: {input_root}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_root / "eval_curve_summary_fair_auc"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_metrics = set(args.metrics) if args.metrics else None

    json_files = sorted(input_root.rglob(args.pattern))
    if not json_files:
        raise FileNotFoundError(
            f"No files matched pattern '{args.pattern}' under: {input_root}"
        )

    metric_store: Dict[str, Dict[str, object]] = {}
    parse_errors: List[str] = []
    clean_probs: List[float] = []
    adv_probs: List[float] = []
    per_folder_saved = 0

    for path in json_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # pylint: disable=broad-except
            parse_errors.append(f"{path}: {exc}")
            continue

        clean_val = _safe_float(data.get("clean_prob"))
        if clean_val is not None:
            clean_probs.append(clean_val)

        adv_val = _safe_float(data.get("adv_prob"))
        if adv_val is not None:
            adv_probs.append(adv_val)

        metric_items = extract_metrics(data, selected_metrics)
        for metric_name, curve_arr in metric_items.items():
            if metric_name not in metric_store:
                metric_store[metric_name] = {
                    "normalized_curves": [],
                    "fair_aucs": [],
                    "num_invalid_max": 0,
                }

            if curve_arr is None or curve_arr.size == 0:
                continue

            norm_curve = normalize_curve_by_max(curve_arr)
            if norm_curve is None:
                metric_store[metric_name]["num_invalid_max"] += 1
                continue

            fair_auc = compute_auc_from_curve(norm_curve)
            if np.isnan(fair_auc):
                continue

            metric_store[metric_name]["normalized_curves"].append(norm_curve.astype(np.float32))
            metric_store[metric_name]["fair_aucs"].append(fair_auc)

            if args.save_per_folder_summary:
                image_name = args.per_folder_image_name.format(metric=metric_name)
                json_name = args.per_folder_json_name.format(metric=metric_name)
                image_path = path.parent / image_name
                json_path = path.parent / json_name

                save_normalized_curve_summary_plot(
                    normalized_curve=norm_curve,
                    fair_auc=fair_auc,
                    metric_name=metric_name,
                    output_path=image_path,
                    dpi=args.dpi,
                )
                save_normalized_curve_information(
                    normalized_curve=norm_curve,
                    fair_auc=fair_auc,
                    metric_name=metric_name,
                    source_file=path,
                    output_path=json_path,
                )
                per_folder_saved += 1

    if not metric_store:
        raise RuntimeError("No valid metric data found. Check input folder and JSON format.")

    summary = {
        "input_root": str(input_root),
        "pattern": args.pattern,
        "num_files_scanned": len(json_files),
        "num_parse_errors": len(parse_errors),
        "parse_errors": parse_errors,
        "normalization": "curve / max(curve)",
        "clean_score": {
            "clean_prob": summarize_values(clean_probs),
            "adv_prob": summarize_values(adv_probs),
        },
        "metrics": {},
    }

    print(f"Scanned files: {len(json_files)}")
    if parse_errors:
        print(f"Parse errors: {len(parse_errors)} (saved in summary)")
    if args.save_per_folder_summary:
        print(f"Per-folder normalized summaries saved: {per_folder_saved}")

    for metric_name in sorted(metric_store.keys()):
        curves = metric_store[metric_name]["normalized_curves"]
        fair_aucs = metric_store[metric_name]["fair_aucs"]
        invalid_max = int(metric_store[metric_name]["num_invalid_max"])

        if len(curves) == 0:
            print(f"[{metric_name}] skipped: no valid normalized curves")
            continue

        aligned, min_len, max_len, truncated_count = align_curves(curves)

        mean_curve = aligned.mean(axis=0)
        std_curve = aligned.std(axis=0)

        mean_fair_auc = float(np.mean(fair_aucs))
        std_fair_auc = float(np.std(fair_aucs))

        x = np.linspace(0.0, 1.0, min_len, dtype=np.float64)

        plt.figure(figsize=(7, 4.5))
        plt.plot(x, mean_curve, linewidth=2.0)
        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.25,
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = output_dir / f"mean_curve_normalized_{metric_name}.png"
        plt.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
        plt.close()

        npy_path = output_dir / f"mean_curve_normalized_{metric_name}.npy"
        np.save(npy_path, mean_curve)

        summary["metrics"][metric_name] = {
            "num_normalized_curves": int(len(curves)),
            "num_fair_aucs": int(len(fair_aucs)),
            "num_invalid_max": invalid_max,
            "curve_length_min": int(min_len),
            "curve_length_max": int(max_len),
            "num_truncated_to_min_length": int(truncated_count),
            "mean_fair_auc": mean_fair_auc,
            "std_fair_auc": std_fair_auc,
            "plot_path": str(fig_path),
            "mean_curve_path": str(npy_path),
        }

        print(
            f"[{metric_name}] curves={len(curves)} fair_aucs={len(fair_aucs)} "
            f"mean_fair_auc={mean_fair_auc:.6f} std_fair_auc={std_fair_auc:.6f} "
            f"len(min/max)={min_len}/{max_len} invalid_max={invalid_max}"
        )

    summary_path = output_dir / "summary_fair_auc.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
