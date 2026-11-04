import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_EXPLAINERS = ["gradcam", "eclip", "game", "maskclip", "rise"]
DEFAULT_APPROACHES = ["ins_optimized_output", "del_optimized_output", "ins_del_optimized_output"]
DEFAULT_DATASETS = ["ImageNet", "CUB", "StanfordPet"]
DEFAULT_EPSILONS = [2.0, 4.0, 8.0, 16.0]

EXPLAINER_DISPLAY = {
    "eclip": "GradECLIP",
    "game": "GAME",
    "gradcam": "GradCAM",
    "maskclip": "MaskCLIP",
    "rise": "RISE",
}

APPROACH_DISPLAY = {
    "x_shift_score": "X-Shift",
    "ins_optimized_output": "IOA",
    "del_optimized_output": "DOA",
    "ins_del_optimized_output": "FOA",
    "clean_causual_score": "Clean",
}

APPROACH_ALIASES = {
    "x_shift_score": ["x_shift_score", "xshift_score", "Xshift_score"],
    "ins_optimized_output": ["ins_optimized_output", "ins_optimize_output"],
    "del_optimized_output": ["del_optimized_output", "del_optimize_output"],
    "ins_del_optimized_output": ["ins_del_optimized_output", "ins_del_optimize_output", "final_output"],
    "clean_causual_score": ["clean_causual_score", "clean_causal_score"],
}

DATASET_ALIASES = {
    "imagenet": "imagenet",
    "image_net": "imagenet",
    "cub": "cub",
    "stanfordpet": "standfordpet",
    "standfordpet": "standfordpet",
    "stanford_pet": "standfordpet",
    "standford_pet": "standfordpet",
    "pet": "standfordpet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export LaTeX confidence table from confidence_by_epsilon_dataset_explainer.csv. "
            "Rows are grouped by explainer with one row per approach, plus one global mean row "
            "computed over all table rows for each column."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to confidence_by_epsilon_dataset_explainer.csv",
    )
    parser.add_argument(
        "--output-tex",
        type=str,
        default=None,
        help="Output LaTeX path. Default: <input-csv-dir>/table_confidence_by_epsilon.tex",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Datasets to show as column groups, e.g. ImageNet CUB StandfordPet",
    )
    parser.add_argument(
        "--epsilons",
        nargs="*",
        type=float,
        default=DEFAULT_EPSILONS,
        help="Epsilon columns per dataset group, e.g. 2 4 8",
    )
    parser.add_argument(
        "--explainers",
        nargs="*",
        default=DEFAULT_EXPLAINERS,
        help="Explainer order for row blocks.",
    )
    parser.add_argument(
        "--approaches",
        nargs="*",
        default=DEFAULT_APPROACHES,
        help="Approach rows inside each explainer block.",
    )
    parser.add_argument(
        "--include-average",
        action="store_true",
        help="Add one extra column-group named Average (mean over selected datasets).",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimals for confidence values (default: 2).",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Confidence scores across perturbation strengths, datasets, explainers, and approaches.",
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="tab:confidence_by_epsilon",
        help="LaTeX label.",
    )
    parser.add_argument(
        "--no-resizebox",
        action="store_true",
        help="Disable \\resizebox{\\linewidth}{!}{...}",
    )
    return parser.parse_args()


def normalize_key(text: str) -> str:
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def canonical_dataset_name(name: str) -> str:
    norm = normalize_key(name)
    return DATASET_ALIASES.get(norm, norm)


def canonical_approach(name: str) -> str:
    norm = normalize_key(name)
    for canonical, aliases in APPROACH_ALIASES.items():
        if any(normalize_key(alias) == norm for alias in aliases):
            return canonical
    return name


def fmt_value(value: Optional[float], decimals: int) -> str:
    if value is None:
        return "--"
    if not np.isfinite(value):
        return "--"
    return f"{float(value):.{decimals}f}"


def safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not nums:
        return None
    return float(np.asarray(nums, dtype=np.float64).mean())


def parse_float_or_none(text: object) -> Optional[float]:
    try:
        x = float(text)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def load_conf_map(csv_path: Path) -> Dict[Tuple[str, str, str, float], float]:
    table: Dict[Tuple[str, str, str, float], float] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = canonical_dataset_name(str(row.get("dataset", "")).strip().strip(","))
            explainer = normalize_key(row.get("explainer", ""))
            approach = canonical_approach(str(row.get("approach_canonical", row.get("approach_key", ""))))
            eps = parse_float_or_none(row.get("epsilon"))
            mean_conf = parse_float_or_none(row.get("mean_confidence"))
            if not dataset or not explainer or not approach or eps is None or mean_conf is None:
                continue

            table[(dataset, explainer, approach, float(eps))] = float(mean_conf)

    return table


def value_for(
    conf_map: Dict[Tuple[str, str, str, float], float],
    dataset: str,
    explainer: str,
    approach: str,
    epsilon: float,
) -> Optional[float]:
    ds = canonical_dataset_name(dataset)
    return conf_map.get((ds, normalize_key(explainer), canonical_approach(approach), float(epsilon)))


def build_table(args: argparse.Namespace, conf_map: Dict[Tuple[str, str, str, float], float]) -> str:
    datasets = list(args.datasets)
    epsilons = [float(x) for x in args.epsilons]
    explainers = list(args.explainers)
    approaches = [canonical_approach(x) for x in args.approaches]

    col_groups = list(datasets)
    if args.include_average:
        col_groups.append("Average")

    # 2 leading columns: Explainer + Approach
    num_metric_cols = len(col_groups) * len(epsilons)
    col_spec = "ll" + ("c" * num_metric_cols)

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{args.caption}}}")
    lines.append(f"\\label{{{args.label}}}")

    if not args.no_resizebox:
        lines.append("\\resizebox{\\linewidth}{!}{%")

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    first_header_cells = ["", ""]
    for name in col_groups:
        first_header_cells.append(f"\\multicolumn{{{len(epsilons)}}}{{c}}{{\\textbf{{{name}}}}}")
    lines.append(" & ".join(first_header_cells) + " \\\\")

    cmid_parts: List[str] = []
    start_col = 3
    for _ in col_groups:
        end_col = start_col + len(epsilons) - 1
        cmid_parts.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    lines.extend(cmid_parts)

    second_header_cells = ["\\textbf{Explainer}", "\\textbf{Approach}"]
    for _ in col_groups:
        for eps in epsilons:
            second_header_cells.append(rf"$\epsilon={eps:g}$")
    lines.append(" & ".join(second_header_cells) + " \\\\")
    lines.append("\\midrule")

    global_rows_values: List[List[Optional[float]]] = []

    for exp_idx, explainer in enumerate(explainers):
        exp_key = normalize_key(explainer)
        exp_name = EXPLAINER_DISPLAY.get(exp_key, explainer)

        row_count = len(approaches)
        first_row = True

        for approach in approaches:
            cells: List[str] = []
            if first_row:
                cells.append(f"\\multirow[c]{{{row_count}}}{{*}}{{{exp_name}}}")
                first_row = False
            else:
                cells.append("")

            cells.append(APPROACH_DISPLAY.get(approach, approach))

            per_col_values: List[Optional[float]] = []
            for dataset in datasets:
                for eps in epsilons:
                    v = value_for(conf_map, dataset, exp_key, approach, eps)
                    per_col_values.append(v)
                    cells.append(fmt_value(v, args.decimals))

            if args.include_average:
                for eps in epsilons:
                    vals = [value_for(conf_map, ds, exp_key, approach, eps) for ds in datasets]
                    avg_v = safe_mean(vals)
                    per_col_values.append(avg_v)
                    cells.append(fmt_value(avg_v, args.decimals))

            global_rows_values.append(per_col_values)
            lines.append(" & ".join(cells) + " \\\\")
        if exp_idx != len(explainers) - 1:
            lines.append("\\midrule")

    lines.append("\\midrule")
    mean_cells = ["", "\\textbf{Mean}"]
    col_count = len(global_rows_values[0]) if global_rows_values else 0
    for col_idx in range(col_count):
        col_vals = [row[col_idx] for row in global_rows_values]
        m = safe_mean(col_vals)
        mean_cells.append(f"\\textbf{{{fmt_value(m, args.decimals)}}}")
    lines.append(" & ".join(mean_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    if not args.no_resizebox:
        lines.append("}")

    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    if not input_csv.exists() or not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_tex = (
        Path(args.output_tex).expanduser().resolve()
        if args.output_tex
        else input_csv.parent / "table_confidence_by_epsilon.tex"
    )

    conf_map = load_conf_map(input_csv)
    if not conf_map:
        raise RuntimeError("No valid confidence rows found in CSV.")

    latex = build_table(args, conf_map)

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    with output_tex.open("w", encoding="utf-8") as f:
        f.write(latex)

    print(f"Saved: {output_tex}")


if __name__ == "__main__":
    main()
