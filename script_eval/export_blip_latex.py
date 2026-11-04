import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple


DISPLAY_NAMES = {
    "eclip": "GradECLIP",
    "gradeclip": "GradECLIP",
    "game": "GAME",
    "gradcam": "GradCAM",
    "maskclip": "MaskCLIP",
    "rise": "RISE",
}

DATASETS = (
    ("ImageNet", {"imagenet"}),
    ("StanfordPets", {"stanfordpet", "stanfordpets", "standfordpet", "standfordpets"}),
    ("CUB", {"cub"}),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BLIP fair-AUC summaries to a LaTeX table."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("BLIP_result"),
        help="BLIP result root containing one directory per dataset.",
    )
    parser.add_argument(
        "--epsilon",
        type=str,
        default=None,
        help="Optional single epsilon to export (for example: 2 or 16).",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=None,
        help="Output .tex file. Default: <input-root>/blip_fair_auc.tex",
    )
    return parser.parse_args()


def _read_metric(summary_path: Path, metric: str) -> Optional[Tuple[float, float]]:
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)

    metric_data = summary.get("metrics", {}).get(metric)
    if not isinstance(metric_data, dict):
        return None

    mean = metric_data.get("mean_fair_auc")
    std = metric_data.get("std_fair_auc")
    if not isinstance(mean, (int, float)) or not math.isfinite(mean):
        return None
    if not isinstance(std, (int, float)) or not math.isfinite(std):
        return None
    return float(mean), float(std)


def _format_metric(value: Optional[Tuple[float, float]]) -> str:
    if value is None:
        return "--"
    return f"{value[0]:.4f} $\\pm$ {value[1]:.4f}"


def _find_dataset_dir(input_root: Path, aliases: set[str]) -> Optional[Path]:
    for path in input_root.iterdir():
        if path.is_dir() and "".join(c for c in path.name.lower() if c.isalnum()) in aliases:
            return path
    return None


def collect_rows(
    input_root: Path, selected_epsilon: Optional[str] = None
) -> Tuple[
    List[str],
    List[str],
    Dict[Tuple[str, str, str], Dict[str, Optional[Tuple[float, float]]]],
]:
    dataset_dirs = []
    for display_name, aliases in DATASETS:
        dataset_dir = _find_dataset_dir(input_root, aliases)
        if dataset_dir is None:
            raise FileNotFoundError(
                f"Missing {display_name} result directory under: {input_root}"
            )
        dataset_dirs.append((display_name, dataset_dir))

    if selected_epsilon is not None:
        epsilon_name = selected_epsilon.removeprefix("eps_")
        epsilon_names = [epsilon_name]
    else:
        epsilon_names = sorted(
            {
                path.name.removeprefix("eps_")
                for _, dataset_dir in dataset_dirs
                for path in dataset_dir.glob("eps_*")
                if path.is_dir()
            },
            key=lambda value: (int(value) if value.isdigit() else math.inf, value),
        )
    if not epsilon_names:
        raise FileNotFoundError(f"No eps_* result folders found under: {input_root}")

    explainers = set()
    metrics: Dict[Tuple[str, str, str], Optional[Tuple[float, float]]] = {}
    found_summary = False

    for epsilon in epsilon_names:
        for dataset_name, dataset_dir in dataset_dirs:
            epsilon_dir = dataset_dir / f"eps_{epsilon}"
            if not epsilon_dir.is_dir():
                continue
            for explainer_dir in sorted(path for path in epsilon_dir.iterdir() if path.is_dir()):
                summary_path = (
                    explainer_dir / "eval_curve_summary_fair_auc" / "summary_fair_auc.json"
                )
                if not summary_path.is_file():
                    continue
                found_summary = True
                explainer = DISPLAY_NAMES.get(
                    explainer_dir.name.lower(), explainer_dir.name.replace("_", r"\_")
                )
                explainers.add(explainer)
                metrics[(epsilon, explainer, dataset_name)] = {
                    "del": _read_metric(summary_path, "del"),
                    "ins": _read_metric(summary_path, "ins"),
                }

    if not found_summary:
        epsilon_text = f" for eps_{epsilon_names[0]}" if selected_epsilon is not None else ""
        raise FileNotFoundError(
            f"No BLIP fair-AUC summaries found{epsilon_text}. Expected "
            "<input-root>/<dataset>/eps_*/<explainer>/"
            "eval_curve_summary_fair_auc/summary_fair_auc.json. "
            "Run script_eval/run_eval.py for each dataset and epsilon first."
        )
    return epsilon_names, sorted(explainers), metrics


def build_table(
    epsilon_names: List[str],
    explainers: List[str],
    metrics: Dict[Tuple[str, str, str], Dict[str, Optional[Tuple[float, float]]]],
) -> str:
    column_spec = "ll" + "cc" * len(DATASETS)
    header = " & ".join(
        [r"$\epsilon$", r"\textbf{Explainer}"]
        + [rf"\multicolumn{{2}}{{c}}{{{dataset}}}" for dataset, _ in DATASETS]
    )
    subheader = " & ".join(
        ["", ""]
        + [
            r"Del. fair-AUC $\downarrow$ & Ins. fair-AUC $\uparrow$"
            for _ in DATASETS
        ]
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{BLIP fair-AUC results by attack budget, explainer, and dataset.}",
        r"\label{tab:blip-fair-auc}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header + r" \\",
    ]
    start_col = 3
    for _ in DATASETS:
        lines.append(rf"\cmidrule(lr){{{start_col}-{start_col + 1}}}")
        start_col += 2
    lines.extend([subheader + r" \\", r"\midrule"])

    for epsilon in epsilon_names:
        for explainer in explainers:
            cells = [epsilon, explainer]
            for dataset_name, _ in DATASETS:
                values = metrics.get((epsilon, explainer, dataset_name))
                if values is None:
                    cells.extend(["--", "--"])
                else:
                    cells.extend(
                        [_format_metric(values["del"]), _format_metric(values["ins"])]
                    )
            lines.append(" & ".join(cells) + r" \\")
        if epsilon != epsilon_names[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root is not a directory: {input_root}")

    output_path = (
        args.output_tex.expanduser().resolve()
        if args.output_tex
        else input_root / (
            f"blip_fair_auc_eps_{args.epsilon.removeprefix('eps_')}.tex"
            if args.epsilon
            else "blip_fair_auc.tex"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epsilon_names, explainers, metrics = collect_rows(input_root, args.epsilon)
    output_path.write_text(build_table(epsilon_names, explainers, metrics), encoding="utf-8")
    print(f"Saved LaTeX table: {output_path}")


if __name__ == "__main__":
    main()
