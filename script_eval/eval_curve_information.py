import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate curve_information.json files recursively: "
            "mean/std AUC and mean/std curve per metric."
        )
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Root folder to recursively search (e.g., del_optimize_output)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save outputs. "
            "Default: <input-root>/eval_curve_summary"
        ),
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
        help=(
            "Optional metric names to include (e.g., --metrics del ins). "
            "By default, all metrics found under rerun are used."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)",
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
) -> Dict[str, Tuple[Optional[float], Optional[np.ndarray]]]:
    extracted: Dict[str, Tuple[Optional[float], Optional[np.ndarray]]] = {}

    rerun = data.get("rerun")
    if isinstance(rerun, dict):
        for metric_name, metric_obj in rerun.items():
            if selected_metrics is not None and metric_name not in selected_metrics:
                continue
            if not isinstance(metric_obj, dict):
                continue

            curve = metric_obj.get("curve")
            auc = metric_obj.get("auc")

            curve_arr: Optional[np.ndarray] = None
            if isinstance(curve, list) and len(curve) > 0:
                curve_arr = np.asarray(curve, dtype=np.float32)

            auc_val = _safe_float(auc)
            if auc_val is None and curve_arr is not None:
                auc_val = compute_auc_from_curve(curve_arr)

            extracted[metric_name] = (auc_val, curve_arr)

    return extracted


def align_curves(curves: List[np.ndarray]) -> Tuple[np.ndarray, int, int, int]:
    lengths = [c.size for c in curves]
    min_len = min(lengths)
    max_len = max(lengths)
    truncated_count = sum(1 for ln in lengths if ln != min_len)

    aligned = np.stack([c[:min_len] for c in curves], axis=0)
    return aligned, min_len, max_len, truncated_count


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist or is not a directory: {input_root}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_root / "eval_curve_summary"
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
        for metric_name, (auc_val, curve_arr) in metric_items.items():
            if metric_name not in metric_store:
                metric_store[metric_name] = {
                    "aucs": [],
                    "curves": [],
                    "files": [],
                }

            if auc_val is not None and not np.isnan(auc_val):
                metric_store[metric_name]["aucs"].append(auc_val)

            if curve_arr is not None and curve_arr.size > 0:
                metric_store[metric_name]["curves"].append(curve_arr)
                metric_store[metric_name]["files"].append(str(path))

    if not metric_store:
        raise RuntimeError(
            "No valid metric data found. Check your input folder and JSON format."
        )

    summary = {
        "input_root": str(input_root),
        "pattern": args.pattern,
        "num_files_scanned": len(json_files),
        "num_parse_errors": len(parse_errors),
        "parse_errors": parse_errors,
        "clean_score": {
            "clean_prob": summarize_values(clean_probs),
            "adv_prob": summarize_values(adv_probs),
        },
        "metrics": {},
    }

    print(f"Scanned files: {len(json_files)}")
    if parse_errors:
        print(f"Parse errors: {len(parse_errors)} (saved in summary)")
    if clean_probs:
        clean_stats = summarize_values(clean_probs)
        print(
            "[clean_prob] "
            f"count={clean_stats['count']} "
            f"mean={clean_stats['mean']:.6f} "
            f"std={clean_stats['std']:.6f}"
        )
    if adv_probs:
        adv_stats = summarize_values(adv_probs)
        print(
            "[adv_prob] "
            f"count={adv_stats['count']} "
            f"mean={adv_stats['mean']:.6f} "
            f"std={adv_stats['std']:.6f}"
        )

    for metric_name in sorted(metric_store.keys()):
        aucs = metric_store[metric_name]["aucs"]
        curves = metric_store[metric_name]["curves"]

        if len(curves) == 0:
            print(f"[{metric_name}] skipped: no valid curves")
            continue

        aligned, min_len, max_len, truncated_count = align_curves(curves)

        mean_curve = aligned.mean(axis=0)
        std_curve = aligned.std(axis=0)

        mean_auc = float(np.mean(aucs)) if aucs else float("nan")
        std_auc = float(np.std(aucs)) if aucs else float("nan")

        x = np.linspace(0.0, 1.0, min_len, dtype=np.float64)

        plt.figure(figsize=(7, 4.5))
        plt.plot(x, mean_curve, linewidth=2.0, label="mean curve")
        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.25,
            label="mean +/- std",
        )
        plt.xlabel("Normalized step")
        plt.ylabel("Confidence")
        plt.title(f"Mean curve - {metric_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fig_path = output_dir / f"mean_curve_{metric_name}.png"
        plt.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
        plt.close()

        npy_path = output_dir / f"mean_curve_{metric_name}.npy"
        np.save(npy_path, mean_curve)

        summary["metrics"][metric_name] = {
            "num_curves": int(len(curves)),
            "num_aucs": int(len(aucs)),
            "curve_length_min": int(min_len),
            "curve_length_max": int(max_len),
            "num_truncated_to_min_length": int(truncated_count),
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "plot_path": str(fig_path),
            "mean_curve_path": str(npy_path),
        }

        print(
            f"[{metric_name}] curves={len(curves)} aucs={len(aucs)} "
            f"mean_auc={mean_auc:.6f} std_auc={std_auc:.6f} "
            f"len(min/max)={min_len}/{max_len}"
        )
        print(f"[{metric_name}] saved plot: {fig_path}")

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
