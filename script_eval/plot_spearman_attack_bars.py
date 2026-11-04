import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


ATTACK_LABELS = {
    # "Xshift_score": "X-Shift",
    "ins_optimize_output": "Ours (Ins-Oriented)",
    "del_optimize_output": "Ours (Del-Oriented)",
}

DATASET_LABELS = {
    "ImageNet": "ImageNet",
    "StandfordPet": "StanfordPets",
    "CUB": "CUB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped bar chart of Spearman correlation across 3 attacks and 3 datasets "
            "from saliency_spearman_summary.json."
        )
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=r"D:/neurips_result/result/saliency_spearman_report/saliency_spearman_summary.json",
        help="Path to saliency_spearman_summary.json",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["ImageNet", "StandfordPet", "CUB"],
        help="Dataset keys in report JSON",
    )
    parser.add_argument(
        "--attacks",
        nargs="*",
        default=["Xshift_score", "ins_optimize_output", "del_optimize_output"],
        help="Attack keys in report JSON",
    )
    parser.add_argument(
        "--explainers",
        nargs="*",
        default=None,
        help="Optional explainer subset. Default: auto from report.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output folder. Default: same folder as report-json",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.2,
        help="Scale text/line/annotation size for better readability.",
    )
    parser.add_argument(
        "--figure-scale",
        type=float,
        default=0.7,
        help="Scale overall canvas size. Smaller values produce a more compact figure.",
    )
    parser.add_argument("--dpi", type=int, default=400, help="Figure DPI")
    return parser.parse_args()


def apply_style(zoom: float) -> None:
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
            "axes.linewidth": 1.1 * zoom,
        }
    )


def discover_explainers(report: dict, datasets: List[str], attacks: List[str]) -> List[str]:
    exp_set = set()
    data_root = report.get("datasets", {})
    for dataset in datasets:
        d = data_root.get(dataset, {})
        for attack in attacks:
            a = d.get(attack, {})
            exp_set.update(a.keys())
    return sorted(exp_set)


def collect_stats(
    report: dict,
    datasets: List[str],
    attacks: List[str],
    explainers: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    data_root = report.get("datasets", {})
    stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    for explainer in explainers:
        stats[explainer] = {}
        for attack in attacks:
            vals: List[float] = []
            for dataset in datasets:
                d = data_root.get(dataset, {})
                a = d.get(attack, {})
                e = a.get(explainer, {})
                spearman_mean = e.get("spearman", {}).get("mean")
                if spearman_mean is None:
                    continue
                vals.append(float(spearman_mean))

            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                stats[explainer][attack] = {
                    "n_datasets": int(arr.size),
                    "mean": float(arr.mean()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
            else:
                stats[explainer][attack] = {
                    "n_datasets": 0,
                    "mean": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }

    return stats


def main() -> None:
    args = parse_args()
    if args.zoom <= 0:
        raise ValueError("--zoom must be positive")
    if args.figure_scale <= 0:
        raise ValueError("--figure-scale must be positive")

    apply_style(args.zoom)

    report_path = Path(args.report_json).expanduser().resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Report JSON not found: {report_path}")

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else report_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    explainers = args.explainers if args.explainers else discover_explainers(report, args.datasets, args.attacks)
    if not explainers:
        raise RuntimeError("No explainers found in report for selected datasets/attacks.")

    stats = collect_stats(report, args.datasets, args.attacks, explainers)

    print(f"Report: {report_path}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Attacks: {', '.join(args.attacks)}")
    print(f"Explainers: {', '.join(explainers)}")
    print(f"Dataset averaging count (target): {len(args.datasets)}")
    print(f"Zoom: {args.zoom}")
    print(f"Figure scale: {args.figure_scale}")

    # Grouped bars: x-axis is explainers, each group contains bars for attack methods.
    x = np.arange(len(explainers), dtype=np.float64)
    width = 0.78 / max(1, len(args.attacks))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(11.0 * args.figure_scale, 5.6 * args.figure_scale))

    csv_rows: List[Dict[str, object]] = []

    for attack_idx, attack in enumerate(args.attacks):
        means = [stats[explainer][attack]["mean"] for explainer in explainers]
        ns = [stats[explainer][attack]["n_datasets"] for explainer in explainers]

        offset = (attack_idx - (len(args.attacks) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            label=ATTACK_LABELS.get(attack, attack),
            color=colors[attack_idx % len(colors)],
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )

        for bar, mean_val in zip(bars, means):
            if np.isfinite(mean_val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    mean_val + 0.008,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9 * args.zoom,
                    rotation=0,
                )

        for explainer, mean_val, n_ds in zip(explainers, means, ns):
            csv_rows.append(
                {
                    "explainer": explainer,
                    "attack": attack,
                    "attack_label": ATTACK_LABELS.get(attack, attack),
                    "mean_spearman": mean_val,
                    "n_datasets": n_ds,
                }
            )

    ax.set_xticks(x)
    ax.set_xticklabels([name.upper() for name in explainers])
    ax.set_ylabel("Spearman Correlation")
    ax.set_xlabel("Explainer")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=len(args.attacks), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    fig_png = output_dir / "spearman_attack_dataset_grouped_bar.png"
    fig_pdf = output_dir / "spearman_attack_dataset_grouped_bar.pdf"
    fig.savefig(fig_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(fig_pdf, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / "spearman_attack_dataset_grouped_bar.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "explainer",
                "attack",
                "attack_label",
                "mean_spearman",
                "n_datasets",
            ],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print("=" * 80)
    print(f"Saved figure PNG: {fig_png}")
    print(f"Saved figure PDF: {fig_pdf}")
    print(f"Saved table CSV: {csv_path}")


if __name__ == "__main__":
    main()
