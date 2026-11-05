"""Export BLIP faithfulness results without model inference or GPU dependencies."""

import argparse
import json
import logging
import math
import os
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


DATASETS = (
    ("ImageNet", {"imagenet"}),
    ("StanfordPets", {"stanfordpet", "stanfordpets", "standfordpet", "standfordpets"}),
    ("CUB", {"cub"}),
)
EXPLAINERS = (
    ("GradECLIP", {"eclip", "gradeclip"}),
    ("GAME", {"game"}),
    ("GradCAM", {"gradcam"}),
    ("MaskCLIP", {"maskclip"}),
    ("RISE", {"rise"}),
)
ATTACKS = ("Without Attack", "X-Shift", "IOA", "DOA", "IOA+DOA")
ATTACK_ALIASES = {
    "clean": "Without Attack",
    "clean_score": "Without Attack",
    "clean_causual_score": "Without Attack",
    "xshift": "X-Shift",
    "x-shift": "X-Shift",
    "xshift_score": "X-Shift",
    "ins": "IOA",
    "ioa": "IOA",
    "ins_optimize_output": "IOA",
    "del": "DOA",
    "doa": "DOA",
    "del_optimize_output": "DOA",
    "del+ins": "IOA+DOA",
    "ioa+doa": "IOA+DOA",
    "doa+ioa": "IOA+DOA",
    "joint": "IOA+DOA",
}
METRICS = ("IMD", "SRC", "ROAD")


def finite_number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            return float(value)
    return None


def fair_auc(curve):
    if not isinstance(curve, list) or len(curve) < 2:
        raise ValueError("A curve must contain at least two finite numbers")
    values = [finite_number(value) for value in curve]
    if any(value is None for value in values):
        raise ValueError("Nonfinite or nonnumeric curve value")
    maximum = max(values)
    if maximum <= 0:
        raise ValueError("Cannot normalize a curve whose maximum is nonpositive")
    return sum(
        (left + right) / 2 for left, right in zip(values, values[1:])
    ) / ((len(values) - 1) * maximum)


def identify(path, root, data):
    parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
    dataset = next(
        (name for name, aliases in DATASETS if any(p in aliases for p in parts)), None
    )
    explainer = next(
        (name for name, aliases in EXPLAINERS if any(p in aliases for p in parts)), None
    )
    attacks = {ATTACK_ALIASES[p] for p in parts if p in ATTACK_ALIASES}
    if not attacks:
        mode = data.get("attack_mode")
        if isinstance(mode, str) and mode.lower() in ATTACK_ALIASES:
            attacks.add(ATTACK_ALIASES[mode.lower()])
    if len(attacks) > 1:
        raise ValueError(f"Ambiguous attack directories: {path}")
    epsilon = next((p[4:] for p in parts if p.startswith("eps_")), None)
    return dataset, explainer, next(iter(attacks), None), epsilon


def result_files(root, epsilon):
    def on_error(error):
        raise error

    for directory, subdirs, filenames in os.walk(root, onerror=on_error):
        subdirs[:] = sorted(
            name for name in subdirs
            if not name.startswith(("steps_", "eval_curve_summary"))
            and (not name.lower().startswith("eps_") or name.lower() == f"eps_{epsilon}")
        )
        for filename in sorted(filenames):
            yield Path(directory) / filename


def find_saliency(sample_dir, explainer, attack):
    aliases = next(aliases for name, aliases in EXPLAINERS if name == explainer)
    prefixes = {
        "IOA": "ins", "DOA": "del", "IOA+DOA": "joint",
    }
    prefix = prefixes.get(attack)
    for pattern in (
        f"{prefix}_del_adv_{{alias}}_saliency.npy" if prefix else None,
        "del_xshift_{alias}_saliency.npy" if attack == "X-Shift" else None,
        "{alias}_saliency.npy",
        "{alias}_heatmap.npy",
    ):
        if pattern is None:
            continue
        candidates = [
            directory / pattern.format(alias=alias)
            for directory in (sample_dir, sample_dir / "saliency")
            for alias in sorted(aliases)
            if (directory / pattern.format(alias=alias)).is_file()
        ]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous saliency maps for {sample_dir}: {candidates}")
        if candidates:
            return candidates[0]
    return None


def index_clean_maps(root, epsilon):
    clean_maps = {}
    for path in result_files(root, epsilon):
        if path.suffix != ".npy":
            continue
        dataset, explainer, attack, file_epsilon = identify(path, root, {})
        if attack != "Without Attack" or dataset is None or explainer is None:
            continue
        sample_dir = path.parent.parent if path.parent.name == "saliency" else path.parent
        selected = find_saliency(sample_dir, explainer, attack)
        if selected != path:
            continue
        key = (dataset, explainer, sample_dir.name)
        previous = clean_maps.get(key)
        if previous is not None and previous[0] != path:
            if previous[1] == file_epsilon:
                raise ValueError(f"Duplicate clean sample {key}: {previous[0]} and {path}")
            if previous[1] is not None:
                continue
        clean_maps[key] = (path, file_epsilon)
    return {key: value[0] for key, value in clean_maps.items()}


def saliency_spearman(clean_path, attack_path):
    maps = []
    for path in (clean_path, attack_path):
        values = np.load(path, allow_pickle=False).squeeze()
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"Expected a finite 2D saliency map: {path}")
        maps.append(values)
    if maps[0].shape != maps[1].shape:
        raise ValueError(f"Saliency shape mismatch: {clean_path} and {attack_path}")
    if any(np.ptp(values) == 0 for values in maps):
        logging.warning("Constant saliency map; SRC unavailable: %s / %s", clean_path, attack_path)
        return None
    return float(spearmanr(maps[0].ravel(), maps[1].ravel())[0])


def collect_metrics(root, epsilon, clean_root=None, expected_attacks=ATTACKS):
    # Keep sample-level pairs: never subtract means over unmatched sample sets.
    samples = {}
    scanned = 0
    filenames = {
        "curve_information.json", "insertion_information.json",
        "deletion_information.json", "road_deletion_information.json",
    }
    clean_maps = index_clean_maps(clean_root or root, epsilon)
    logging.info("Indexed %d clean saliency maps for eps_%s", len(clean_maps), epsilon)
    for path in result_files(root, epsilon):
        if path.name not in filenames:
            continue
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        dataset, explainer, attack, file_epsilon = identify(path, root, data)
        if not all((dataset, explainer, attack)):
            logging.warning("Unrecognized dataset/explainer/attack; skipping %s", path)
            continue
        if file_epsilon != epsilon and not (
            file_epsilon is None and attack == "Without Attack"
        ):
            continue
        scanned += 1
        key = (dataset, explainer, attack, path.parent)
        record = samples.setdefault(key, {})
        try:
            if path.name == "road_deletion_information.json":
                value = finite_number(data.get("deletion_fair_auc"))
                if value is not None:
                    record["ROAD"] = value
                continue
            rerun = data.get("rerun", {})
            for mode, prefix in (("ins", "insertion"), ("del", "deletion")):
                block = rerun.get(mode, {}) if isinstance(rerun, dict) else {}
                curve = block.get("curve") if isinstance(block, dict) else None
                if curve is None:
                    curve = data.get(f"{prefix}_curve")
                if curve is not None:
                    record[mode] = fair_auc(curve)
            for metric, field in (("SRC", "spearman"), ("ROAD", "road_fair_auc")):
                value = finite_number(data.get(field))
                if value is not None:
                    record[metric] = value
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc

    if not scanned:
        raise FileNotFoundError(
            f"No recognized per-sample results for eps_{epsilon} under {root}. "
            "Expected <dataset>/eps_<n>/<explainer>/<ins|del|del+ins>/<sample>/"
            "curve_information.json (or named attack folders)."
        )
    values = {}
    for (dataset, explainer, attack, sample_dir), record in samples.items():
        if attack != "Without Attack" and "SRC" not in record:
            clean_path = clean_maps.get((dataset, explainer, sample_dir.name))
            attack_path = find_saliency(sample_dir, explainer, attack)
            if clean_path is None or attack_path is None:
                logging.warning(
                    "Missing clean/attack saliency pair; SRC blank: %s (clean=%s, attack=%s)",
                    sample_dir, clean_path, attack_path,
                )
            else:
                correlation = saliency_spearman(clean_path, attack_path)
                if correlation is not None:
                    record["SRC"] = correlation
        if "ins" in record and "del" in record:
            record["IMD"] = record["ins"] - record["del"]
        for metric in METRICS:
            if metric in record:
                values.setdefault((dataset, explainer, attack, metric), []).append(record[metric])
    means = {key: statistics.mean(items) for key, items in values.items()}
    for dataset, _ in DATASETS:
        for attack in expected_attacks:
            count = sum(
                1 for key in means if key[0] == dataset and key[2] == attack
            )
            if not count:
                logging.warning("No metrics for %s / %s; cells left blank", dataset, attack)
    return means


def build_clean_table(means):
    lines = [
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{ImageNet} & \multicolumn{3}{c}{StanfordPets} "
        r"& \multicolumn{3}{c}{CUB} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"\textbf{Explainer} & \textbf{Attack} & "
        + " & ".join(
            [r"IMD $\downarrow$", r"SRC $\downarrow$", r"ROAD $\uparrow$"] * 3
        ) + r" \\",
        r"\midrule",
    ]
    for explainer, _ in EXPLAINERS:
        cells = []
        for dataset, _ in DATASETS:
            for metric in METRICS:
                value = means.get((dataset, explainer, "Without Attack", metric))
                cells.append("--" if value is None or metric == "SRC" else f"{value:.4f}")
        lines.append(" & ".join([explainer, "Without Attack"] + cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_cell(means, dataset, explainer, attack, metric):
    value = means.get((dataset, explainer, attack, metric))
    if value is None or (attack == "Without Attack" and metric == "SRC"):
        return ""
    available = [
        means[(dataset, explainer, candidate, metric)]
        for candidate in ATTACKS
        if (dataset, explainer, candidate, metric) in means
        and not (candidate == "Without Attack" and metric == "SRC")
    ]
    best = max(available) if metric == "ROAD" else min(available)
    text = f"{value:.4f}"
    if value == best:
        text = rf"\textbf{{\textcolor{{blue!60}}{{{text}}}}}"
    text = rf"{{\scriptsize {text}}}"
    baseline = "X-Shift" if metric == "SRC" else "Without Attack"
    base_value = means.get((dataset, explainer, baseline, metric))
    if attack not in ("Without Attack", baseline) and base_value is not None:
        delta = value - base_value
        delta_text = f"{delta:+.3f}".replace("+0.", "+.").replace("-0.", "-.")
        if (delta > 0 if metric == "ROAD" else delta < 0):
            delta_text = rf"\textcolor{{red}}{{{delta_text}}}"
        text += rf"{{\raisebox{{-0.65ex}}{{\hspace{{0.12em}}\tiny {delta_text}}}}}"
    return text


def build_table(means, epsilon):
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\renewcommand{\arraystretch}{1}", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Comparison of adversarial attacks on BLIP across datasets "
        + rf"($\epsilon={epsilon}/255$). "
        + r"IMD is insertion fair-AUC minus deletion fair-AUC, each curve normalized "
        r"by its own maximum. Lower IMD and SRC, and higher ROAD indicate more "
        r"effective attacks. Blue indicates best available scores; red deltas "
        r"indicate improvements over Without Attack (IMD/ROAD) or X-Shift (SRC). "
        r"Missing measurements are blank.}",
        r"\label{tab:blip-faithfulness}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
        r"& & \multicolumn{3}{c}{ImageNet} & \multicolumn{3}{c}{StanfordPets} "
        r"& \multicolumn{3}{c}{CUB} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"\textbf{Explainer} & \textbf{Attack} & "
        + " & ".join(
            [r"IMD $\downarrow$", r"SRC $\downarrow$", r"ROAD $\uparrow$"] * 3
        ) + r" \\", r"\midrule",
    ]
    for index, (explainer, _) in enumerate(EXPLAINERS):
        if index:
            lines.append(r"\midrule")
        for attack_index, attack in enumerate(ATTACKS):
            first = rf"\multirow[c]{{5}}{{*}}{{{explainer}}}" if attack_index == 0 else ""
            label = rf"\textbf{{{attack}}}" if attack_index >= 2 else attack
            cells = [
                render_cell(means, dataset, explainer, attack, metric)
                for dataset, _ in DATASETS for metric in METRICS
            ]
            lines.append(" & ".join([first, label] + cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("BLIP_result"))
    parser.add_argument("--epsilon", help="Single pixel budget, e.g. 2 or eps_2")
    parser.add_argument(
        "--clean-only", action="store_true",
        help="Export only clean IMD/SRC/ROAD rows from a completed clean output root",
    )
    parser.add_argument(
        "--clean-root", type=Path,
        help="Optional separate clean root: <dataset>/clean/<explainer>/<sample>/saliency/*.npy",
    )
    parser.add_argument("--output-tex", type=Path)
    args = parser.parse_args()
    if args.epsilon is None and not args.clean_only:
        parser.error("--epsilon is required unless --clean-only is set")
    epsilon = (args.epsilon or "0").removeprefix("eps_")
    if not epsilon.isdigit():
        parser.error("--epsilon must be a nonnegative integer (optionally prefixed eps_)")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = args.input_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input root is not a directory: {root}")
    clean_root = args.clean_root.expanduser().resolve() if args.clean_root else None
    if clean_root is not None and not clean_root.is_dir():
        raise FileNotFoundError(f"Clean root is not a directory: {clean_root}")
    expected_attacks = ("Without Attack",) if args.clean_only else ATTACKS
    means = collect_metrics(root, epsilon, clean_root, expected_attacks)
    if args.clean_only:
        output = args.output_tex or root / "blip_clean_faithfulness.tex"
        table = build_clean_table(means)
    else:
        output = args.output_tex or root / f"blip_faithfulness_eps_{epsilon}.tex"
        table = build_table(means, epsilon)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    logging.info("Saved %s (%d populated metric cells)", output, len(means))


if __name__ == "__main__":
    main()
