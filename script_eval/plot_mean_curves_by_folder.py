import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


FOLDER_LABEL_ALIASES = {
    "clean_causal_score": "Without Attack",
    "clean_causual_score": "Without Attack",
    "xshift_score": "X-Shift",
    "x_shift_score": "X-Shift",
    "ins_optimize_output": "IOA",
    "del_optimize_output": "DOA",
    "ins_optimized_output": "IOA",
    "del_optimized_output": "DOA",
    "ins_del_optimize_output": "FOA",
    "ins_del_optimized_output": "FOA",
    "final_output": "FOA",
}

FOLDER_PLOT_ORDER = {
    "clean_causal_score": 0,
    "clean_causual_score": 0,
    "xshift_score": 1,
    "x_shift_score": 1,
    "ins_optimize_output": 2,
    "del_optimize_output": 3,
    "ins_optimized_output": 2,
    "del_optimized_output": 3,
    "ins_del_optimize_output": 4,
    "ins_del_optimized_output": 4,
    "final_output": 4,
}

FOLDER_COLOR_ALIASES = {
    "clean_causal_score": "#2E8B57",   # Without Attack: green
    "clean_causual_score": "#2E8B57",  # Without Attack: green
    "xshift_score": "#1F77B4",         # X-Shift: blue
    "x_shift_score": "#1F77B4",        # X-Shift: blue
    "ins_optimize_output": "#E75480",  # IOA: pink
    "del_optimize_output": "#FF8C00",  # DOA: orange
    "ins_optimized_output": "#E75480",  # IOA: pink
    "del_optimized_output": "#FF8C00",  # DOA: orange
    "ins_del_optimize_output": "#D62728",  # FOA: red
    "ins_del_optimized_output": "#D62728",  # FOA: red
    "final_output": "#D62728",  # FOA: red
}


def _normalize_label_key(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def folder_display_name(folder_name: str) -> str:
    return FOLDER_LABEL_ALIASES.get(_normalize_label_key(folder_name), folder_name)


def folder_plot_order(folder_name: str) -> int:
    return FOLDER_PLOT_ORDER.get(_normalize_label_key(folder_name), 1000)


def folder_plot_color(folder_name: str, fallback_palette: List[str], folder_idx: int) -> str:
    mapped = FOLDER_COLOR_ALIASES.get(_normalize_label_key(folder_name))
    if mapped is not None:
        return mapped
    return fallback_palette[folder_idx % len(fallback_palette)]


def metric_title(metric: str) -> str:
    return "Deletion" if metric == "del" else "Insertion"


def normalize_curve_by_max(curve: np.ndarray) -> Optional[np.ndarray]:
    if curve.size == 0:
        return None
    max_val = float(np.max(curve))
    if max_val <= 0.0:
        return None
    return (curve.astype(np.float64) / max_val).astype(np.float32)


def apply_paper_style(zoom: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 14 * zoom,
            "axes.labelsize": 16 * zoom,
            "axes.titlesize": 20 * zoom,
            "axes.linewidth": 1.2 * zoom,
            "xtick.labelsize": 13 * zoom,
            "ytick.labelsize": 13 * zoom,
            "legend.fontsize": 13 * zoom,
            "figure.dpi": 100,
            "savefig.dpi": 300,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot average curves for selected folders per dataset. "
            "Each folder is expected under each dataset root, for example: "
            "<result-root>/<dataset>/<model-name>/<folder>/<sample>/*.json"
        )
    )
    parser.add_argument(
        "--result-root",
        type=str,
        required=True,
        help="Root containing dataset directories (e.g., D:/neurips_result/result)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Optional model folder appended after dataset. "
            "Per-dataset root becomes <result-root>/<dataset>/<model-name>."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names. Default: auto-detect all subfolders under result-root.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        required=True,
        help="Folders to compare (e.g., clean_causual_score PGD_score Xshift_score SomethingElse)",
    )
    parser.add_argument(
        "--metric",
        choices=["del", "ins", "both"],
        default="del",
        help="Curve metric to analyze: del, ins, or both (side-by-side)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <result-root>/curve_compare_by_dataset",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI",
    )
    parser.add_argument(
        "--marker-every",
        type=int,
        default=10,
        help="Place a marker every N points on each mean curve (default: 12)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.5,
        help=(
            "Scale visual elements (font/line/marker). "
            "Increase this if the figure will be shrunk in paper."
        ),
    )
    parser.add_argument(
        "--figure-scale",
        type=float,
        default=0.7,
        help="Scale the overall figure canvas size (smaller value makes image dimensions smaller).",
    )
    parser.add_argument(
        "--normalize-curve-max",
        action="store_true",
        help="Normalize each sample curve by its own max value before averaging (fair-AUC style).",
    )
    parser.add_argument(
        "--average-datasets",
        action="store_true",
        help="Also generate a plot that averages per-folder mean curves across selected datasets.",
    )
    return parser.parse_args()


def _extract_curve(data: dict, metric: str) -> Optional[np.ndarray]:
    # clean_score format
    clean_keys = {
        "del": "deletion_curve",
        "ins": "insertion_curve",
    }

    curve_key = clean_keys[metric]
    curve = data.get(curve_key)
    if isinstance(curve, list) and len(curve) > 0:
        return np.asarray(curve, dtype=np.float32)

    # rerun format
    rerun = data.get("rerun")
    if isinstance(rerun, dict):
        metric_obj = rerun.get(metric)
        if isinstance(metric_obj, dict):
            curve = metric_obj.get("curve")
            if isinstance(curve, list) and len(curve) > 0:
                return np.asarray(curve, dtype=np.float32)

    return None


def collect_curves(folder_root: Path, metric: str) -> List[np.ndarray]:
    # Prioritize explicit files to reduce accidental reads.
    preferred_file = "deletion_information.json" if metric == "del" else "insertion_information.json"
    files = sorted(folder_root.rglob(preferred_file))
    if not files:
        files = sorted(folder_root.rglob("*information.json"))

    curves: List[np.ndarray] = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        curve = _extract_curve(data, metric)
        if curve is not None and curve.size > 0:
            curves.append(curve)

    return curves


def align_curves(curves: List[np.ndarray]) -> Tuple[np.ndarray, int, int, int]:
    lengths = [c.size for c in curves]
    min_len = min(lengths)
    max_len = max(lengths)
    num_truncated = sum(1 for ln in lengths if ln != min_len)

    aligned = np.stack([c[:min_len] for c in curves], axis=0)
    return aligned, min_len, max_len, num_truncated


def discover_datasets(result_root: Path, selected_datasets: Optional[List[str]]) -> List[str]:
    if selected_datasets:
        return selected_datasets

    return sorted([p.name for p in result_root.iterdir() if p.is_dir()])


def main() -> None:
    args = parse_args()
    if args.zoom <= 0:
        raise ValueError("--zoom must be positive")
    if args.figure_scale <= 0:
        raise ValueError("--figure-scale must be positive")

    apply_paper_style(args.zoom)

    if args.marker_every <= 0:
        raise ValueError("--marker-every must be a positive integer")

    result_root = Path(args.result_root).expanduser().resolve()
    if not result_root.exists() or not result_root.is_dir():
        raise FileNotFoundError(f"Result root does not exist or is not a directory: {result_root}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_root / "curve_compare_by_dataset"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(result_root, args.datasets)
    if not datasets:
        raise RuntimeError(f"No dataset folders found under: {result_root}")

    folders_raw = []
    seen = set()
    for name in args.folders:
        n = name.strip()
        key = _normalize_label_key(n)
        if not n or key in seen:
            continue
        folders_raw.append(n)
        seen.add(key)

    folders = sorted(
        folders_raw,
        key=lambda n: (folder_plot_order(n), _normalize_label_key(n)),
    )

    if len(folders) < 2:
        raise ValueError("Please provide at least 2 folders to compare.")

    summary: Dict[str, object] = {
        "result_root": str(result_root),
        "metric": args.metric,
        "normalize_curve_max": bool(args.normalize_curve_max),
        "average_datasets": bool(args.average_datasets),
        "folders": folders,
        "datasets": {},
    }

    metrics_to_plot = ["del", "ins"] if args.metric == "both" else [args.metric]
    metric_tag = "del_ins" if args.metric == "both" else args.metric

    print(f"Result root: {result_root}")
    print(f"Model name : {args.model_name if args.model_name else '(none)'}")
    print(f"Datasets   : {', '.join(datasets)}")
    print(f"Folders    : {', '.join(folders)}")
    print(f"Metric     : {', '.join(metrics_to_plot)}")
    print(f"Marker step: every {args.marker_every} points")
    print(f"Zoom       : {args.zoom}")
    print(f"FigureScale: {args.figure_scale}")
    print(f"NormByMax  : {args.normalize_curve_max}")
    print(f"AvgDatasets: {args.average_datasets}")

    # Stores one mean curve per (dataset, folder, metric) so each dataset contributes equally.
    dataset_mean_bank: Dict[str, Dict[str, List[np.ndarray]]] = {
        metric: {folder: [] for folder in folders} for metric in metrics_to_plot
    }

    for dataset in datasets:
        dataset_root = result_root / dataset
        if args.model_name:
            dataset_root = dataset_root / args.model_name
        if not dataset_root.exists() or not dataset_root.is_dir():
            print(f"[SKIP] dataset={dataset}: missing folder")
            continue

        per_metric_stats: Dict[str, Dict[str, object]] = {}

        fig, axes = plt.subplots(
            1,
            len(metrics_to_plot),
            figsize=(11.2 * len(metrics_to_plot) * args.figure_scale, 7.0 * args.figure_scale),
            sharey=True,
        )
        if len(metrics_to_plot) == 1:
            axes = [axes]
        plotted_total = 0
        legend_handles = None
        legend_labels = None
        # High-contrast, color-blind-friendly palette (Okabe-Ito + strong accents)
        palette = [
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#009E73",  # bluish green
            "#CC79A7",  # reddish purple
            "#F0E442",  # yellow
            "#000000",  # black
            "#56B4E9",  # sky blue
            "#E69F00",  # orange
        ]
        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

        for ax, metric in zip(axes, metrics_to_plot):
            per_folder_stats: Dict[str, object] = {}
            plotted_metric = 0

            for folder_idx, folder_name in enumerate(folders):
                folder_root = dataset_root / folder_name
                if not folder_root.exists() or not folder_root.is_dir():
                    print(f"[SKIP] dataset={dataset}, folder={folder_name}, metric={metric}: not found")
                    continue

                curves = collect_curves(folder_root, metric)
                if not curves:
                    print(f"[SKIP] dataset={dataset}, folder={folder_name}, metric={metric}: no curves")
                    continue

                if args.normalize_curve_max:
                    normalized_curves: List[np.ndarray] = []
                    dropped_non_positive = 0
                    for curve in curves:
                        norm_curve = normalize_curve_by_max(curve)
                        if norm_curve is None:
                            dropped_non_positive += 1
                            continue
                        normalized_curves.append(norm_curve)
                    curves = normalized_curves

                    if not curves:
                        print(
                            f"[SKIP] dataset={dataset}, folder={folder_name}, metric={metric}: "
                            "all curves have non-positive max"
                        )
                        continue
                    if dropped_non_positive > 0:
                        print(
                            f"[INFO] dataset={dataset}, folder={folder_name}, metric={metric}: "
                            f"dropped_non_positive_max={dropped_non_positive}"
                        )

                aligned, min_len, max_len, num_truncated = align_curves(curves)
                mean_curve = aligned.mean(axis=0)
                std_curve = aligned.std(axis=0)
                x = np.linspace(0.0, 1.0, min_len, dtype=np.float64)

                color = folder_plot_color(folder_name, palette, folder_idx)
                marker = markers[folder_idx % len(markers)]
                markevery = max(1, min_len // args.marker_every)

                ax.plot(
                    x,
                    mean_curve,
                    linewidth=3.1 * args.zoom,
                    color=color,
                    marker=marker,
                    markevery=markevery,
                    markersize=6.0 * args.zoom,
                    markerfacecolor="white",
                    markeredgewidth=1.4 * args.zoom,
                    label=folder_display_name(folder_name),
                )
                plotted_metric += 1
                plotted_total += 1

                per_folder_stats[folder_name] = {
                    "num_curves": int(len(curves)),
                    "curve_length_min": int(min_len),
                    "curve_length_max": int(max_len),
                    "num_truncated_to_min_length": int(num_truncated),
                    "mean_curve": mean_curve.tolist(),
                    "std_curve": std_curve.tolist(),
                }
                dataset_mean_bank[metric][folder_name].append(mean_curve.astype(np.float64))

                print(
                    f"[OK] dataset={dataset}, folder={folder_name}, metric={metric}, "
                    f"curves={len(curves)}, len(min/max)={min_len}/{max_len}"
                )

            if plotted_metric == 0:
                ax.set_visible(False)
                print(f"[SKIP] dataset={dataset}, metric={metric}: no valid curves for selected folders")
                continue

            per_metric_stats[metric] = {
                "folders": per_folder_stats,
            }
            ax.set_xlabel("Normalized step")
            ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        if plotted_total == 0:
            plt.close(fig)
            print(f"[SKIP] dataset={dataset}: no valid curves for selected folders")
            continue

        axes[0].set_ylabel("Confidence" if args.normalize_curve_max else "Confidence")
        if len(metrics_to_plot) > 1 and legend_handles and legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                frameon=False,
                ncol=max(1, len(legend_labels)),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.98),
            )
            fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
        else:
            if legend_handles and legend_labels:
                fig.legend(
                    legend_handles,
                    legend_labels,
                    frameon=False,
                    ncol=max(1, len(legend_labels)),
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.98),
                )
                fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
            else:
                fig.tight_layout()

        fig_path = output_dir / f"mean_curve_{metric_tag}_{dataset}.png"
        fig_pdf_path = output_dir / f"mean_curve_{metric_tag}_{dataset}.pdf"
        fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(fig_pdf_path, bbox_inches="tight")
        plt.close(fig)

        dataset_entry: Dict[str, object] = {
            "plot_path": str(fig_path),
            "plot_pdf_path": str(fig_pdf_path),
            "metrics": per_metric_stats,
        }
        if len(metrics_to_plot) == 1:
            dataset_entry["folders"] = per_metric_stats[metrics_to_plot[0]]["folders"]

        summary["datasets"][dataset] = dataset_entry

    if args.average_datasets:
        agg_fig, agg_axes = plt.subplots(
            1,
            len(metrics_to_plot),
            figsize=(11.2 * len(metrics_to_plot) * args.figure_scale, 7.0 * args.figure_scale),
            sharey=True,
        )
        if len(metrics_to_plot) == 1:
            agg_axes = [agg_axes]

        agg_legend_handles = None
        agg_legend_labels = None
        agg_plotted_total = 0
        aggregate_summary: Dict[str, object] = {
            "metrics": {},
        }

        palette = [
            "#0072B2",
            "#D55E00",
            "#009E73",
            "#CC79A7",
            "#F0E442",
            "#000000",
            "#56B4E9",
            "#E69F00",
        ]
        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

        for ax, metric in zip(agg_axes, metrics_to_plot):
            metric_summary: Dict[str, object] = {"folders": {}}
            plotted_metric = 0

            for folder_idx, folder_name in enumerate(folders):
                dataset_means = dataset_mean_bank[metric][folder_name]
                if not dataset_means:
                    print(f"[SKIP] aggregate, folder={folder_name}, metric={metric}: no dataset mean curves")
                    continue

                aligned, min_len, max_len, num_truncated = align_curves(dataset_means)
                mean_curve = aligned.mean(axis=0)
                std_curve = aligned.std(axis=0)
                x = np.linspace(0.0, 1.0, min_len, dtype=np.float64)

                color = folder_plot_color(folder_name, palette, folder_idx)
                marker = markers[folder_idx % len(markers)]
                markevery = max(1, min_len // args.marker_every)

                ax.plot(
                    x,
                    mean_curve,
                    linewidth=3.1 * args.zoom,
                    color=color,
                    marker=marker,
                    markevery=markevery,
                    markersize=6.0 * args.zoom,
                    markerfacecolor="white",
                    markeredgewidth=1.4 * args.zoom,
                    label=folder_display_name(folder_name),
                )
                plotted_metric += 1
                agg_plotted_total += 1

                metric_summary["folders"][folder_name] = {
                    "num_datasets": int(len(dataset_means)),
                    "curve_length_min": int(min_len),
                    "curve_length_max": int(max_len),
                    "num_truncated_to_min_length": int(num_truncated),
                    "mean_curve": mean_curve.tolist(),
                    "std_curve": std_curve.tolist(),
                }

                print(
                    f"[OK] aggregate, folder={folder_name}, metric={metric}, "
                    f"datasets={len(dataset_means)}, len(min/max)={min_len}/{max_len}"
                )

            if plotted_metric == 0:
                ax.set_visible(False)
                print(f"[SKIP] aggregate, metric={metric}: no valid curves")
                continue

            aggregate_summary["metrics"][metric] = metric_summary
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if agg_legend_handles is None:
                agg_legend_handles, agg_legend_labels = ax.get_legend_handles_labels()

        if agg_plotted_total > 0:
            agg_axes[0].set_ylabel("Normalized Confidence (curve/max)" if args.normalize_curve_max else "Confidence")
            if len(metrics_to_plot) > 1 and agg_legend_handles and agg_legend_labels:
                agg_fig.legend(
                    agg_legend_handles,
                    agg_legend_labels,
                    frameon=False,
                    ncol=max(1, len(agg_legend_labels)),
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.98),
                )
                agg_fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
            else:
                if agg_legend_handles and agg_legend_labels:
                    agg_fig.legend(
                        agg_legend_handles,
                        agg_legend_labels,
                        frameon=False,
                        ncol=max(1, len(agg_legend_labels)),
                        loc="upper center",
                        bbox_to_anchor=(0.5, 0.98),
                    )
                    agg_fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
                else:
                    agg_fig.tight_layout()

            agg_fig_path = output_dir / f"mean_curve_{metric_tag}_avg_datasets.png"
            agg_fig_pdf_path = output_dir / f"mean_curve_{metric_tag}_avg_datasets.pdf"
            agg_fig.savefig(agg_fig_path, dpi=args.dpi, bbox_inches="tight")
            agg_fig.savefig(agg_fig_pdf_path, bbox_inches="tight")

            aggregate_summary["plot_path"] = str(agg_fig_path)
            aggregate_summary["plot_pdf_path"] = str(agg_fig_pdf_path)
            summary["aggregate_datasets"] = aggregate_summary
        else:
            print("[SKIP] aggregate: no valid curves across datasets")

        plt.close(agg_fig)

    summary_path = output_dir / f"summary_mean_curves_{metric_tag}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Saved summary: {summary_path}")
    print(f"Saved plots under: {output_dir}")


if __name__ == "__main__":
    main()
