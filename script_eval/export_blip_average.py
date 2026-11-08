"""Export the BLIP four-metric table with equal weight per available explainer."""

import argparse
import csv
import json
import logging
from pathlib import Path
import statistics

try:
    from .export_blip_latex import ATTACKS, DATASETS, EXPLAINERS, collect_metrics
except ImportError:
    from export_blip_latex import ATTACKS, DATASETS, EXPLAINERS, collect_metrics


METRICS = ("IMD", "SRC", "DEL_mean", "ROAD")


def aggregate(means):
    result, counts = {}, {}
    for dataset, _ in DATASETS:
        for attack in ATTACKS:
            for metric in METRICS:
                key = (dataset, attack, metric)
                entries = {
                    explainer: means[(dataset, explainer, attack, metric)]
                    for explainer, _ in EXPLAINERS
                    if (dataset, explainer, attack, metric) in means
                }
                if attack == "Without Attack" and metric == "SRC":
                    entries = {}
                if entries:
                    result[key] = statistics.mean(entries.values())
                elif not (attack == "Without Attack" and metric == "SRC"):
                    logging.warning("No results: %s / %s / %s", *key)
                counts[key] = entries
    return result, counts


def build_table(averages, epsilon):
    lines = [
        r"\begin{table*}[h]", r"\centering", r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.1}", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{BLIP attack comparison "
        + rf"($\epsilon={epsilon}/255$). "
        + r"Results are sample means within each explainer, then equally averaged "
        r"across available explainers per cell. IMD is insertion minus deletion "
        r"fair AUC; SRC is clean/adversarial saliency Spearman correlation. "
        r"Lower IMD and SRC, higher Del$_{\text{mean}}$ and ROAD indicate more "
        r"effective attacks. Del$_{\text{mean}}$ uses channel-mean imputation; "
        r"ROAD uses noisy linear imputation. Both report fair AUC. Missing "
        r"measurements are --; clean SRC is not applicable.}",
        rf"\label{{tab:blip-faithfulness-avg-eps-{epsilon}}}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lcccccccccccc}",
        r"\toprule", r"\multicolumn{13}{c}{\textbf{BLIP}} \\", r"\midrule",
        r"\textbf{Attack} & \multicolumn{4}{c}{ImageNet} "
        r"& \multicolumn{4}{c}{StanfordPets} & \multicolumn{4}{c}{CUB} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
        "& " + " & ".join([
            r"IMD $\downarrow$", r"SRC $\downarrow$",
            r"Del$_{\text{mean}}$ $\uparrow$", r"ROAD $\uparrow$",
        ] * 3) + r" \\", r"\midrule",
    ]
    for attack in ATTACKS:
        cells = [
            "--" if (dataset, attack, metric) not in averages
            else f"{averages[(dataset, attack, metric)]:.4f}"
            for dataset, _ in DATASETS for metric in METRICS
        ]
        lines.append(" & ".join([attack, *cells]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--attack-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--epsilon", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    epsilon = args.epsilon.removeprefix("eps_")
    if not epsilon.isdigit():
        parser.error("--epsilon must be a nonnegative integer")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    roots = list(dict.fromkeys(root.resolve() for root in args.attack_roots))
    for root in [args.clean_root, *roots]:
        if not root.is_dir():
            raise FileNotFoundError(f"Result folder not found: {root}")
    means = collect_metrics(
        args.clean_root, epsilon, expected_attacks=("Without Attack",),
        metrics=METRICS, require_imputer_metadata=True,
    )
    for root in roots:
        values = collect_metrics(
            root, epsilon, clean_root=args.clean_root,
            expected_attacks=ATTACKS[1:], metrics=METRICS, require_imputer_metadata=True,
        )
        overlap = means.keys() & values.keys()
        if overlap:
            raise ValueError(f"Duplicate dataset/explainer/attack metrics across input roots: {sorted(overlap)}")
        means.update(values)
    averages, counts = aggregate(means)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"BLIP_result_avg_eps_{epsilon}"
    tex = args.output_dir / f"{stem}.tex"
    tex.write_text(build_table(averages, epsilon), encoding="utf-8")
    rows = []
    for (dataset, attack, metric), entries in counts.items():
        rows.append({
            "dataset": dataset, "attack": attack, "metric": metric,
            "mean": averages.get((dataset, attack, metric)),
            "explainer_count": len(entries), "explainer_means": entries,
        })
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps({"epsilon": epsilon, "aggregation": "equal-weight mean of explainer sample means",
                    "cells": rows}, indent=2, allow_nan=False), encoding="utf-8",
    )
    with (args.output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "attack", "metric", "mean", "explainer_count", "explainers"])
        for row in rows:
            writer.writerow([row["dataset"], row["attack"], row["metric"], row["mean"],
                             row["explainer_count"], ";".join(row["explainer_means"])])
    logging.info("BLIP averaged table: %s", tex)


if __name__ == "__main__":
    main()
