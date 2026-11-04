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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BLIP fair-AUC summaries to a LaTeX table."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("BLIP_result/ImageNet"),
        help="Directory containing eps_* result folders.",
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


def collect_rows(input_root: Path) -> List[Tuple[str, str, str, str]]:
    rows = []
    epsilon_dirs = sorted(
        (path for path in input_root.glob("eps_*") if path.is_dir()),
        key=lambda path: (
            int(path.name.removeprefix("eps_"))
            if path.name.removeprefix("eps_").isdigit()
            else math.inf,
            path.name,
        ),
    )

    for epsilon_dir in epsilon_dirs:
        epsilon = epsilon_dir.name.removeprefix("eps_")
        for explainer_dir in sorted(path for path in epsilon_dir.iterdir() if path.is_dir()):
            summary_path = (
                explainer_dir / "eval_curve_summary_fair_auc" / "summary_fair_auc.json"
            )
            if not summary_path.is_file():
                continue

            deletion = _read_metric(summary_path, "del")
            insertion = _read_metric(summary_path, "ins")
            display_name = DISPLAY_NAMES.get(
                explainer_dir.name.lower(), explainer_dir.name.replace("_", r"\_")
            )
            rows.append(
                (
                    epsilon,
                    display_name,
                    _format_metric(deletion),
                    _format_metric(insertion),
                )
            )

    if not rows:
        raise FileNotFoundError(
            "No BLIP fair-AUC summaries found. Expected "
            "<input-root>/eps_*/<explainer>/eval_curve_summary_fair_auc/"
            "summary_fair_auc.json. Run script_eval/run_eval.py for each eps_* folder first."
        )
    return rows


def build_table(rows: List[Tuple[str, str, str, str]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{BLIP fair-AUC results by attack budget and explainer.}",
        r"\label{tab:blip-fair-auc}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"$\epsilon$ & Explainer & Deletion fair-AUC $\downarrow$ & Insertion fair-AUC $\uparrow$ \\",
        r"\midrule",
    ]
    for epsilon, explainer, deletion, insertion in rows:
        lines.append(
            f"{epsilon} & {explainer} & {deletion} & {insertion} " + r"\\"
        )
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
        else input_root / "blip_fair_auc.tex"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_table(collect_rows(input_root)), encoding="utf-8")
    print(f"Saved LaTeX table: {output_path}")


if __name__ == "__main__":
    main()
