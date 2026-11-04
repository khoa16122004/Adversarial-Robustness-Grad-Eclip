import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


MODE_ALIASES = {
    "del": "del",
    "deletion": "del",
    "ins": "ins",
    "insertion": "ins",
}


def normalize_mode(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return MODE_ALIASES.get(value.strip().lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot optimization traces (PGD loss and mean p_t) averaged across datasets, "
            "comparing del vs ins for each explain method."
        )
    )
    parser.add_argument(
        "--result-root",
        type=str,
        required=True,
        help="Root containing dataset folders (e.g., D:/neurips_result/result)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model folder under each dataset (e.g., CLIP_ViTB16)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names to average over (e.g., StandfordPet CUB ImageNet)",
    )
    parser.add_argument(
        "--attack-folders",
        nargs="+",
        default=["del_optimize_output", "ins_optimize_output"],
        help="Attack output folders under each dataset/model",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional explain methods to include. Default: auto-detect all found methods",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="curve_information.json",
        help="File pattern to search recursively",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <result-root>/optimize_trace_compare",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.2,
        help="Visual scaling factor for fonts and line widths",
    )
    return parser.parse_args()


def apply_plot_style(zoom: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 12 * zoom,
            "axes.labelsize": 13 * zoom,
            "axes.titlesize": 14 * zoom,
            "xtick.labelsize": 11 * zoom,
            "ytick.labelsize": 11 * zoom,
            "legend.fontsize": 10 * zoom,
            "axes.linewidth": 1.1,
        }
    )


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def align_to_min_length(curves: List[np.ndarray]) -> Tuple[np.ndarray, int, int, int]:
    lengths = [c.size for c in curves]
    min_len = min(lengths)
    max_len = max(lengths)
    truncated_count = sum(1 for x in lengths if x != min_len)
    stacked = np.stack([c[:min_len] for c in curves], axis=0)
    return stacked, min_len, max_len, truncated_count


def build_trace_from_data(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    raw_losses = data.get("pgd_loss", [])
    trace = data.get("pgd_trace", [])

    losses: List[float] = []
    if isinstance(raw_losses, list):
        for v in raw_losses:
            fv = _safe_float(v)
            if fv is not None:
                losses.append(fv)

    if not losses and isinstance(trace, list):
        for item in trace:
            if not isinstance(item, dict):
                continue
            fv = _safe_float(item.get("loss"))
            if fv is not None:
                losses.append(fv)

    pt_means: List[float] = []
    if isinstance(trace, list):
        for item in trace:
            if not isinstance(item, dict):
                continue
            p_t = item.get("p_t", [])
            if isinstance(p_t, list) and p_t:
                vals = [x for x in (_safe_float(v) for v in p_t) if x is not None]
                if vals:
                    pt_means.append(float(np.mean(vals)))
                    continue
            pt_means.append(float("nan"))

    loss_arr = np.asarray(losses, dtype=np.float64)
    pt_arr = np.asarray(pt_means, dtype=np.float64)

    if pt_arr.size == 0 and loss_arr.size > 0:
        pt_arr = np.full(loss_arr.size, np.nan, dtype=np.float64)
    if loss_arr.size == 0 and pt_arr.size > 0:
        loss_arr = np.full(pt_arr.size, np.nan, dtype=np.float64)

    if loss_arr.size != pt_arr.size:
        min_len = min(loss_arr.size, pt_arr.size)
        if min_len <= 0:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        loss_arr = loss_arr[:min_len]
        pt_arr = pt_arr[:min_len]

    return loss_arr, pt_arr


def discover_method_from_relpath(rel_parts: Tuple[str, ...], attack_folder: str) -> str:
    attack_key = attack_folder.lower()
    parts_lower = [p.lower() for p in rel_parts]
    if attack_key in parts_lower:
        idx = parts_lower.index(attack_key)
        if idx + 1 < len(rel_parts):
            return rel_parts[idx + 1]
    return "unknown"


def infer_mode(data: dict, rel_parts: Tuple[str, ...]) -> Optional[str]:
    mode = normalize_mode(data.get("attack_mode"))
    if mode is not None:
        return mode

    for part in rel_parts:
        mode = normalize_mode(part)
        if mode is not None:
            return mode
    return None


def main() -> None:
    args = parse_args()
    if args.zoom <= 0:
        raise ValueError("--zoom must be positive")

    apply_plot_style(args.zoom)

    result_root = Path(args.result_root).expanduser().resolve()
    if not result_root.exists() or not result_root.is_dir():
        raise FileNotFoundError(f"Result root does not exist or is not a directory: {result_root}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_root / "optimize_trace_compare"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_methods = set(args.methods) if args.methods else None

    # dataset -> method -> mode -> metric(list of curves)
    dataset_store: Dict[str, Dict[str, Dict[str, Dict[str, List[np.ndarray]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"loss": [], "pt": []}))
    )

    parse_errors: List[str] = []
    skipped_no_mode = 0
    scanned_files = 0

    for dataset in args.datasets:
        model_root = result_root / dataset / args.model_name
        if not model_root.exists() or not model_root.is_dir():
            print(f"[SKIP] dataset={dataset}: missing model folder {model_root}")
            continue

        for attack_folder in args.attack_folders:
            attack_root = model_root / attack_folder
            if not attack_root.exists() or not attack_root.is_dir():
                print(f"[SKIP] dataset={dataset}, attack_folder={attack_folder}: missing")
                continue

            files = sorted(attack_root.rglob(args.pattern))
            print(f"[INFO] dataset={dataset}, attack_folder={attack_folder}, files={len(files)}")

            for path in files:
                scanned_files += 1
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as exc:  # pylint: disable=broad-except
                    parse_errors.append(f"{path}: {exc}")
                    continue

                rel = path.relative_to(model_root)
                rel_parts = rel.parts

                method = discover_method_from_relpath(rel_parts, attack_folder)
                if selected_methods is not None and method not in selected_methods:
                    continue

                mode = infer_mode(data, rel_parts)
                if mode is None:
                    skipped_no_mode += 1
                    continue

                loss_arr, pt_arr = build_trace_from_data(data)
                if loss_arr.size == 0:
                    continue

                dataset_store[dataset][method][mode]["loss"].append(loss_arr)
                dataset_store[dataset][method][mode]["pt"].append(pt_arr)

    # method -> mode -> metric -> list of dataset-level mean curves
    method_dataset_means: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: {"loss": [], "pt": []})
    )

    dataset_level_summary: Dict[str, object] = {}

    for dataset in sorted(dataset_store.keys()):
        dataset_level_summary[dataset] = {}
        for method in sorted(dataset_store[dataset].keys()):
            dataset_level_summary[dataset][method] = {}
            for mode in ["del", "ins"]:
                dataset_level_summary[dataset][method][mode] = {}
                for metric in ["loss", "pt"]:
                    curves = dataset_store[dataset][method][mode][metric]
                    if not curves:
                        continue
                    aligned, min_len, max_len, num_truncated = align_to_min_length(curves)
                    mean_curve = np.nanmean(aligned, axis=0)
                    method_dataset_means[method][mode][metric].append(mean_curve.astype(np.float64))

                    dataset_level_summary[dataset][method][mode][metric] = {
                        "num_samples": int(len(curves)),
                        "curve_length_min": int(min_len),
                        "curve_length_max": int(max_len),
                        "num_truncated_to_min_length": int(num_truncated),
                    }

    global_summary: Dict[str, object] = {
        "result_root": str(result_root),
        "model_name": args.model_name,
        "datasets": args.datasets,
        "attack_folders": args.attack_folders,
        "pattern": args.pattern,
        "num_files_scanned": int(scanned_files),
        "num_parse_errors": int(len(parse_errors)),
        "parse_errors": parse_errors,
        "num_skipped_without_mode": int(skipped_no_mode),
        "dataset_level": dataset_level_summary,
        "methods": {},
    }

    plotted_methods = 0
    for method in sorted(method_dataset_means.keys()):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharex=False)
        mode_styles = {
            "del": {"label": "DEL attack", "color": "#D55E00"},
            "ins": {"label": "INS attack", "color": "#0072B2"},
        }

        method_summary: Dict[str, object] = {}
        has_any_line = False

        for ax, metric in zip(axes, ["loss", "pt"]):
            title = "PGD Loss" if metric == "loss" else "Mean p_t"
            ax.set_title(f"{method} - {title}")
            ax.set_xlabel("Normalized Step")
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)

            if metric == "loss":
                ax.set_ylabel("Loss")
            else:
                ax.set_ylabel("Mean p_t")

            for mode in ["del", "ins"]:
                dataset_mean_curves = method_dataset_means[method][mode][metric]
                if not dataset_mean_curves:
                    continue

                aligned, min_len, max_len, num_truncated = align_to_min_length(dataset_mean_curves)
                mean_curve = np.nanmean(aligned, axis=0)
                std_curve = np.nanstd(aligned, axis=0)
                x = np.linspace(0.0, 1.0, min_len, dtype=np.float64)

                style = mode_styles[mode]
                ax.plot(x, mean_curve, linewidth=2.2, color=style["color"], label=style["label"])
                ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=style["color"], alpha=0.18)
                has_any_line = True

                method_summary.setdefault(mode, {})[metric] = {
                    "num_datasets_used": int(len(dataset_mean_curves)),
                    "curve_length_min": int(min_len),
                    "curve_length_max": int(max_len),
                    "num_truncated_to_min_length": int(num_truncated),
                    "mean_curve": mean_curve.tolist(),
                    "std_curve": std_curve.tolist(),
                }

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, loc="best")

        if not has_any_line:
            plt.close(fig)
            continue

        fig.tight_layout()
        out_fig = output_dir / f"optimize_compare_{method}.png"
        fig.savefig(out_fig, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        global_summary["methods"][method] = {
            "plot_path": str(out_fig),
            "modes": method_summary,
        }

        plotted_methods += 1
        print(f"[OK] method={method} -> {out_fig}")

    summary_path = output_dir / "optimize_compare_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    print(f"Scanned files: {scanned_files}")
    print(f"Parse errors: {len(parse_errors)}")
    print(f"Methods plotted: {plotted_methods}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
