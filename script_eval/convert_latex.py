import argparse
import json
import math
from pathlib import Path


EXPLAINER_ROWS = [
    ("GradECLIP", "eclip"),
    ("GAME", "game"),
    ("GradCAM", "gradcam"),
    ("MaskCLIP", "maskclip"),
    ("RISE", "rise"),
]

ATTACK_ROWS = [
    ("X-Shift", "Xshift_score"),
    ("Ours (Insertion-Oriented)", "ins_optimize_output"),
    ("Ours (Deletion-Oriented)", "del_optimize_output"),
]

# Header hiển thị StanfordPets, nhưng key trong JSON là StandfordPet
DATASET_COLUMNS = [
    ("ImageNet", "ImageNet"),
    ("StanfordPets", "StandfordPet"),
    ("CUB", "CUB"),
]


def fmt_value(v):
    if v is None:
        return "--"
    try:
        x = float(v)
    except Exception:
        return "--"
    if math.isnan(x) or math.isinf(x):
        return "--"
    return f"{x:.4f}"


def get_metrics(root, dataset_key, attack_key, explainer_key):
    d = (
        root.get("datasets", {})
        .get(dataset_key, {})
        .get(attack_key, {})
        .get(explainer_key, {})
    )
    spearman = d.get("spearman", {}).get("mean")
    cosine_block = d.get("adv_text_cosine")
    cosine = None
    if isinstance(cosine_block, dict):
        cosine = cosine_block.get("mean")
    return cosine, spearman


def build_table(data):
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1}")
    lines.append("\\setlength{\\tabcolsep}{2.0pt}")
    lines.append("")
    lines.append("\\caption{")
    lines.append("Comparison of semantic preservation and explanation shift under different adversarial attacks on CLIP ViT-B/16.")
    lines.append("Higher Cosine Similarity ($\\uparrow$) indicates better preservation of image-text semantics, whereas lower Spearman Correlation ($\\downarrow$) indicates a larger shift between adversarial and clean saliency maps.")
    lines.append("}")
    lines.append("\\label{tab:consistency}")
    lines.append("")
    lines.append("\\begin{tabular}{llcccccc}")
    lines.append("\\toprule")
    lines.append("& & \\multicolumn{2}{c}{ImageNet} & \\multicolumn{2}{c}{StanfordPets} & \\multicolumn{2}{c}{CUB} \\\\")
    lines.append("\\cmidrule(lr){3-4}")
    lines.append("\\cmidrule(lr){5-6}")
    lines.append("\\cmidrule(lr){7-8}")
    lines.append("")
    lines.append("\\textbf{Explainer} & \\textbf{Attack} & \\textbf{Cosine} & \\textbf{Spearman} & \\textbf{Cosine} & \\textbf{Spearman} & \\textbf{Cosine} & \\textbf{Spearman} \\\\")
    lines.append("")
    lines.append("\\midrule")
    lines.append("")

    for exp_display, exp_key in EXPLAINER_ROWS:
        first = True
        for attack_display, attack_key in ATTACK_ROWS:
            vals = []
            for _, ds_key in DATASET_COLUMNS:
                cos, spe = get_metrics(data, ds_key, attack_key, exp_key)
                vals.extend([fmt_value(cos), fmt_value(spe)])

            if first:
                prefix = f"\\multirow[c]{{3}}{{*}}{{{exp_display}}}"
                first = False
            else:
                prefix = ""

            line = (
                f"{prefix} & {attack_display} & "
                f"{vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} & {vals[5]} \\\\"
            )
            lines.append(line)

        lines.append("")
        lines.append("\\midrule")
        lines.append("")

    # đổi midrule cuối thành bottomrule cho đẹp
    if len(lines) >= 2 and lines[-2] == "\\midrule":
        lines = lines[:-2]

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Fill LaTeX consistency table from saliency_spearman_summary.json")
    parser.add_argument(
        "--input-json",
        type=str,
        default=r"D:\\neurips_result\\result\\saliency_spearman_report\\saliency_spearman_summary.json",
        help="Path to saliency_spearman_summary.json",
    )
    parser.add_argument(
        "--output-tex",
        type=str,
        default=r"D:\\neurips_result\\result\\saliency_spearman_report\\table_consistency_filled.tex",
        help="Output LaTeX table path",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json).expanduser().resolve()
    output_path = Path(args.output_tex).expanduser().resolve()

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    table_tex = build_table(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(table_tex)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()