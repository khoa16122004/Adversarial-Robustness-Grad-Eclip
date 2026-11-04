import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


DEFAULT_APPROACHES = [
    "x_shift_score",
    "ins_optimized_output",
    "del_optimized_output",
    "ins_del_optimized_output",
]

APPROACH_ALIASES: Dict[str, List[str]] = {
    "clean_causual_score": ["clean_causual_score", "clean_causal_score"],
    "x_shift_score": ["x_shift_score", "xshift_score", "Xshift_score"],
    "ins_optimized_output": ["ins_optimized_output", "ins_optimize_output"],
    "del_optimized_output": ["del_optimized_output", "del_optimize_output"],
    "ins_del_optimized_output": ["ins_del_optimized_output", "ins_del_optimize_output", "final_output"],
}

APPROACH_DISPLAY = {
    "clean_causual_score": "Clean",
    "x_shift_score": "X-Shift",
    "ins_optimized_output": "IOA",
    "del_optimized_output": "DOA",
    "ins_del_optimized_output": "FOA",
}

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_FILTER_PRESETS: Dict[str, Path] = {
    "cub": REPO_ROOT / "classification_reuslt" / "cub" / "vit_b_16_100_random_50.json",
    "imagenet": REPO_ROOT / "classification_reuslt" / "imagenet" / "vit_b_16_100_random_50.json",
    "standfordpet": REPO_ROOT / "classification_reuslt" / "pet" / "vit_b_16_100_random_50.json",
}

DATASET_FILTER_ALIASES: Dict[str, str] = {
    "cub": "cub",
    "c_u_b": "cub",
    "imagenet": "imagenet",
    "image_net": "imagenet",
    "standfordpet": "standfordpet",
    "stanfordpet": "standfordpet",
    "standford_pet": "standfordpet",
    "stanford_pet": "standfordpet",
    "pet": "standfordpet",
}

DATASET_WRAPPER_ALIASES: Dict[str, List[str]] = {
    "cub": ["cub"],
    "imagenet": ["imagenet", "image_net"],
    "standfordpet": ["standfordpet", "stanfordpet", "pet", "stanford_pet", "standford_pet"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report mean confidence by epsilon for each dataset/explainer/approach. "
            "Reads curve_information.json recursively from result_{epsilon} folders."
        )
    )
    parser.add_argument(
        "--results-parent",
        type=str,
        required=True,
        help="Parent folder containing result_{epsilon} directories.",
    )
    parser.add_argument(
        "--epsilon-pattern",
        type=str,
        default="result_*",
        help="Glob pattern for epsilon folders (default: result_*).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="CLIP_ViTB16",
        help="Model folder under each dataset (default: CLIP_ViTB16).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names. Default: auto-detect from first epsilon folder.",
    )
    parser.add_argument(
        "--approaches",
        nargs="*",
        default=DEFAULT_APPROACHES,
        help="Approaches to report. Aliases are supported.",
    )
    parser.add_argument(
        "--explainers",
        "--explainer",
        nargs="*",
        default=None,
        help="Optional explainer names. Default: auto-detect per dataset/approach.",
    )
    parser.add_argument(
        "--include-clean-baseline",
        action="store_true",
        help="Also include clean_causual_score from one epsilon (default clean epsilon=16).",
    )
    parser.add_argument(
        "--clean-folder",
        type=str,
        default="clean_causual_score",
        help="Clean folder key when --include-clean-baseline is used.",
    )
    parser.add_argument(
        "--clean-epsilon",
        type=float,
        default=16.0,
        help="Use clean baseline from this epsilon folder (default: 16).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <results-parent>/epsilon_confidence_report",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional output CSV file path.",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Optional output JSON file path.",
    )
    parser.add_argument(
        "--target-epsilons",
        nargs="*",
        type=float,
        default=None,
        help="Optional epsilon grid to emit. Default: all discovered epsilon folders.",
    )
    parser.add_argument(
        "--emit-missing-grid",
        action="store_true",
        help="Emit rows for missing epsilon/dataset/explainer/approach combinations.",
    )
    parser.add_argument(
        "--fill-missing-from-epsilon",
        type=float,
        default=None,
        help=(
            "If set, missing combinations are filled from this epsilon when available "
            "(e.g. --fill-missing-from-epsilon 16)."
        ),
    )
    parser.add_argument(
        "--complete-only",
        dest="complete_only",
        action="store_true",
        default=False,
        help=(
            "Keep only dataset/approach/explainer combos that have all target epsilons. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        dest="complete_only",
        action="store_false",
        help="Allow partial combos that do not cover every target epsilon (default behavior).",
    )
    parser.add_argument(
        "--sample-filter-file",
        type=str,
        default=None,
        help=(
            "Optional global JSON/TXT sample filter file. "
            "If omitted, per-dataset default filter preset is used when available."
        ),
    )
    parser.add_argument(
        "--disable-default-dataset-filter",
        action="store_true",
        help="Disable automatic dataset-based sample filter presets.",
    )
    return parser.parse_args()


def _normalize_key(text: str) -> str:
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def resolve_dataset_filter_key(dataset_name: str) -> Optional[str]:
    return DATASET_FILTER_ALIASES.get(_normalize_key(dataset_name))


def resolve_default_filter_path_for_dataset(dataset_name: str) -> Optional[Path]:
    key = resolve_dataset_filter_key(dataset_name)
    if key is None:
        return None
    return DATASET_FILTER_PRESETS.get(key)


def resolve_effective_approach_root(approach_root: Path, dataset_name: str) -> Path:
    """Support layouts like ins_del_optimized_output/imagenet/<explainer>/..."""
    if not approach_root.exists() or not approach_root.is_dir():
        return approach_root

    # If root already directly contains curve files somewhere, keep it.
    if any(approach_root.rglob("curve_information.json")):
        # Still allow descent when wrapper exists and root does not directly hold explainer dirs.
        pass

    dataset_key = resolve_dataset_filter_key(dataset_name) or _normalize_key(dataset_name)
    candidate_names = DATASET_WRAPPER_ALIASES.get(dataset_key, [dataset_key])
    candidate_norms = {_normalize_key(x) for x in candidate_names}

    # Prefer explicit dataset wrapper child if present.
    for child in sorted(approach_root.iterdir()):
        if not child.is_dir():
            continue
        if _normalize_key(child.name) in candidate_norms:
            return child

    # Fallback: if there is exactly one child folder and it holds curve files, descend into it.
    children = [p for p in sorted(approach_root.iterdir()) if p.is_dir()]
    if len(children) == 1:
        only = children[0]
        if any(only.rglob("curve_information.json")):
            return only

    return approach_root


def _aliases_for_key(key: str) -> List[str]:
    aliases = APPROACH_ALIASES.get(key)
    if aliases is not None:
        return list(aliases)

    norm = _normalize_key(key)
    for vals in APPROACH_ALIASES.values():
        for alias in vals:
            if _normalize_key(alias) == norm:
                return list(vals)
    return [key]


def canonical_approach_name(name: str) -> str:
    norm = _normalize_key(name)
    for canonical, aliases in APPROACH_ALIASES.items():
        if any(_normalize_key(alias) == norm for alias in aliases):
            return canonical
    return name


def display_approach_name(name: str) -> str:
    canonical = canonical_approach_name(name)
    return APPROACH_DISPLAY.get(canonical, canonical)


def discover_epsilon_folders(results_parent: Path, pattern: str) -> List[Tuple[float, Path]]:
    regex = re.compile(r"result[_-]?([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
    out: List[Tuple[float, Path]] = []

    for p in sorted(results_parent.glob(pattern)):
        if not p.is_dir():
            continue
        m = regex.search(p.name)
        if m is None:
            continue
        try:
            eps = float(m.group(1))
        except ValueError:
            continue
        out.append((eps, p))

    out.sort(key=lambda x: x[0])
    return out


def pick_clean_result_root(epsilon_folders: Sequence[Tuple[float, Path]], clean_epsilon: float) -> Optional[Path]:
    for eps, path in epsilon_folders:
        if np.isclose(eps, clean_epsilon, atol=1e-12):
            return path
    return None


def discover_datasets(epsilon_folders: Sequence[Tuple[float, Path]], selected: Optional[List[str]]) -> List[str]:
    if selected:
        cleaned: List[str] = []
        for name in selected:
            s = str(name).strip().strip(",")
            if not s:
                continue
            k = _normalize_key(s)
            if k == "stanfordpet":
                s = "StandfordPet"
            cleaned.append(s)
        return cleaned
    if not epsilon_folders:
        return []

    first_root = epsilon_folders[0][1]
    return sorted([p.name for p in first_root.iterdir() if p.is_dir()])


def resolve_existing_approach_folder(dataset_model_root: Path, requested_key: str) -> Optional[str]:
    for alias in _aliases_for_key(requested_key):
        if (dataset_model_root / alias).exists():
            return alias
    return None


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_token(text: str) -> str:
    return str(text).strip().replace("\\", "/").lower()


def _token_variants(text: str) -> Set[str]:
    raw = str(text).strip()
    if not raw:
        return set()

    out: Set[str] = set()
    raw_norm = raw.replace("\\", "/")
    out.add(raw_norm)
    out.add(raw_norm.lower())

    name = Path(raw_norm).name
    if name:
        out.add(name)
        out.add(name.lower())
        stem = Path(name).stem
        if stem:
            out.add(stem)
            out.add(stem.lower())

    return {_norm_token(x) for x in out if str(x).strip()}


def load_sample_filter_tokens(path: Path) -> Set[str]:
    tokens: Set[str] = set()
    if not path.exists() or not path.is_file():
        return tokens

    suffix = path.suffix.lower()
    if suffix in {".json", ".js"}:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return tokens

        if isinstance(data, dict):
            for key, value in data.items():
                tokens.update(_token_variants(str(key)))
                if isinstance(value, str):
                    tokens.update(_token_variants(value))
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, str):
                            tokens.update(_token_variants(v))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    tokens.update(_token_variants(item))
                elif isinstance(item, dict):
                    for k, v in item.items():
                        tokens.update(_token_variants(str(k)))
                        if isinstance(v, str):
                            tokens.update(_token_variants(v))
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return tokens
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tokens.update(_token_variants(s))

    return tokens


def sample_id_matches_filter(sample_id: str, filter_tokens: Optional[Set[str]]) -> bool:
    if filter_tokens is None:
        return True
    if not filter_tokens:
        return False
    sample_tokens = _token_variants(sample_id)
    return any(token in filter_tokens for token in sample_tokens)


def _extract_confidence_value(data: dict) -> Optional[float]:
    # Priority: attacked confidence if present, then clean confidence.
    for key in ["adv_prob", "adv_score", "retain_prob", "target_prob"]:
        v = _safe_float(data.get(key))
        if v is not None and np.isfinite(v):
            return float(v)

    for key in ["clean_prob", "clean_score"]:
        v = _safe_float(data.get(key))
        if v is not None and np.isfinite(v):
            return float(v)

    # Fallback to first value from rerun curves if scalar fields are unavailable.
    rerun = data.get("rerun")
    if isinstance(rerun, dict):
        for metric in ["del", "ins"]:
            metric_obj = rerun.get(metric)
            if not isinstance(metric_obj, dict):
                continue
            curve = metric_obj.get("curve")
            if isinstance(curve, list) and len(curve) > 0:
                v = _safe_float(curve[0])
                if v is not None and np.isfinite(v):
                    return float(v)

    for curve_key in ["deletion_curve", "insertion_curve"]:
        curve = data.get(curve_key)
        if isinstance(curve, list) and len(curve) > 0:
            v = _safe_float(curve[0])
            if v is not None and np.isfinite(v):
                return float(v)

    return None


def discover_explainers(approach_root: Path, selected_explainers: Optional[List[str]]) -> List[str]:
    if selected_explainers:
        return list(selected_explainers)
    if not approach_root.exists() or not approach_root.is_dir():
        return []

    out: List[str] = []
    for p in sorted(approach_root.iterdir()):
        if p.is_dir() and not p.name.startswith("eval_curve_summary"):
            out.append(p.name)
    return out


def load_confidences(explainer_root: Path, filter_tokens: Optional[Set[str]] = None) -> Tuple[List[float], int, int]:
    values: List[float] = []
    parsed = 0
    bad_json = 0

    for path in sorted(explainer_root.rglob("curve_information.json")):
        sample_id = path.parent.name
        if sample_id.startswith("eval_curve_summary"):
            continue
        if not sample_id_matches_filter(sample_id, filter_tokens):
            continue

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            bad_json += 1
            continue

        parsed += 1
        if not isinstance(data, dict):
            continue

        conf = _extract_confidence_value(data)
        if conf is not None and np.isfinite(conf):
            values.append(float(conf))

    return values, parsed, bad_json


def summarize_values(values: Sequence[float]) -> Optional[Dict[str, float]]:
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


def main() -> None:
    args = parse_args()

    results_parent = Path(args.results_parent).expanduser().resolve()
    if not results_parent.exists() or not results_parent.is_dir():
        raise FileNotFoundError(f"Invalid --results-parent: {results_parent}")

    epsilon_folders = discover_epsilon_folders(results_parent, args.epsilon_pattern)
    if not epsilon_folders:
        raise RuntimeError("No epsilon folders found.")

    datasets = discover_datasets(epsilon_folders, args.datasets)
    if not datasets:
        raise RuntimeError("No datasets found.")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_parent / "epsilon_confidence_report"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        Path(args.csv_path).expanduser().resolve()
        if args.csv_path
        else output_dir / "confidence_by_epsilon_dataset_explainer.csv"
    )
    json_path = (
        Path(args.json_path).expanduser().resolve()
        if args.json_path
        else output_dir / "confidence_by_epsilon_dataset_explainer.json"
    )

    approaches = list(args.approaches)
    if args.include_clean_baseline:
        approaches = [args.clean_folder] + approaches

    clean_root = pick_clean_result_root(epsilon_folders, args.clean_epsilon)
    selected_epsilons = (
        sorted({float(x) for x in args.target_epsilons})
        if args.target_epsilons
        else [float(eps) for eps, _ in epsilon_folders]
    )

    global_filter_tokens: Optional[Set[str]] = None
    if args.sample_filter_file:
        global_filter_path = Path(args.sample_filter_file).expanduser().resolve()
        global_filter_tokens = load_sample_filter_tokens(global_filter_path)
        print(f"Sample filter (global): {global_filter_path} tokens={len(global_filter_tokens)}")

    rows: List[Dict[str, object]] = []

    for dataset in datasets:
        dataset_filter_tokens: Optional[Set[str]] = global_filter_tokens
        if dataset_filter_tokens is None and not args.disable_default_dataset_filter:
            preset_path = resolve_default_filter_path_for_dataset(dataset)
            if preset_path is not None:
                dataset_filter_tokens = load_sample_filter_tokens(preset_path)
                print(f"Sample filter ({dataset}): {preset_path} tokens={len(dataset_filter_tokens)}")

        for requested_approach in approaches:
            canonical = canonical_approach_name(requested_approach)

            roots_to_scan: List[Tuple[float, Path, str]] = []
            if canonical == canonical_approach_name(args.clean_folder):
                if clean_root is None:
                    continue
                dataset_model_root = clean_root / dataset / args.model_name
                resolved = resolve_existing_approach_folder(dataset_model_root, args.clean_folder)
                if resolved is None:
                    continue
                roots_to_scan.append((float(args.clean_epsilon), dataset_model_root / resolved, resolved))
            else:
                for eps, result_root in epsilon_folders:
                    dataset_model_root = result_root / dataset / args.model_name
                    resolved = resolve_existing_approach_folder(dataset_model_root, requested_approach)
                    if resolved is None:
                        continue
                    roots_to_scan.append((float(eps), dataset_model_root / resolved, resolved))

            for eps, approach_root, resolved_approach in roots_to_scan:
                effective_approach_root = resolve_effective_approach_root(approach_root, dataset)
                explainers = discover_explainers(effective_approach_root, args.explainers)
                if not explainers:
                    continue

                for explainer in explainers:
                    explainer_root = effective_approach_root / explainer
                    if not explainer_root.exists() or not explainer_root.is_dir():
                        continue

                    values, parsed_json_count, bad_json_count = load_confidences(
                        explainer_root,
                        filter_tokens=dataset_filter_tokens,
                    )
                    stats = summarize_values(values)

                    row: Dict[str, object] = {
                        "epsilon": float(eps),
                        "dataset": dataset,
                        "model_name": args.model_name,
                        "approach_key": requested_approach,
                        "approach_canonical": canonical,
                        "approach_display": display_approach_name(canonical),
                        "approach_resolved": resolved_approach,
                        "explainer": explainer,
                        "num_curve_information_files": int(parsed_json_count),
                        "num_bad_json": int(bad_json_count),
                        "num_valid_confidence": 0,
                        "mean_confidence": None,
                        "std_confidence": None,
                        "min_confidence": None,
                        "max_confidence": None,
                        "is_imputed": False,
                        "imputed_from_epsilon": None,
                    }

                    if stats is not None:
                        row["num_valid_confidence"] = int(stats["count"])
                        row["mean_confidence"] = float(stats["mean"])
                        row["std_confidence"] = float(stats["std"])
                        row["min_confidence"] = float(stats["min"])
                        row["max_confidence"] = float(stats["max"])

                    rows.append(row)

    if args.complete_only:
        target_eps_set = {float(x) for x in selected_epsilons}
        eps_by_combo: Dict[Tuple[str, str, str], set] = {}
        for row in rows:
            combo = (
                str(row["dataset"]),
                str(row["approach_canonical"]),
                str(row["explainer"]),
            )
            eps_by_combo.setdefault(combo, set()).add(float(row["epsilon"]))

        complete_combos = {
            combo for combo, eps_set in eps_by_combo.items() if target_eps_set.issubset(eps_set)
        }
        rows = [
            row
            for row in rows
            if (
                str(row["dataset"]),
                str(row["approach_canonical"]),
                str(row["explainer"]),
            )
            in complete_combos
        ]

    if args.emit_missing_grid or args.fill_missing_from_epsilon is not None:
        rows_by_key: Dict[Tuple[str, str, str, float], Dict[str, object]] = {}
        explainers_by_combo: Dict[Tuple[str, str], set] = {}
        approach_key_by_canonical: Dict[str, str] = {}

        for row in rows:
            dataset = str(row["dataset"])
            canonical = str(row["approach_canonical"])
            explainer = str(row["explainer"])
            epsilon = float(row["epsilon"])
            rows_by_key[(dataset, canonical, explainer, epsilon)] = row
            explainers_by_combo.setdefault((dataset, canonical), set()).add(explainer)
            approach_key_by_canonical.setdefault(canonical, str(row["approach_key"]))

        for dataset in datasets:
            for requested_approach in approaches:
                canonical = canonical_approach_name(requested_approach)
                approach_key = approach_key_by_canonical.get(canonical, requested_approach)

                if args.explainers:
                    explainer_list = [str(x).strip().strip(",") for x in args.explainers if str(x).strip().strip(",")]
                else:
                    explainer_list = sorted(explainers_by_combo.get((dataset, canonical), set()))

                for explainer in explainer_list:
                    fallback_row = None
                    if args.fill_missing_from_epsilon is not None:
                        fallback_row = rows_by_key.get(
                            (dataset, canonical, explainer, float(args.fill_missing_from_epsilon))
                        )

                    for eps in selected_epsilons:
                        key = (dataset, canonical, explainer, float(eps))
                        if key in rows_by_key:
                            continue

                        if fallback_row is not None:
                            filled = dict(fallback_row)
                            filled["epsilon"] = float(eps)
                            filled["is_imputed"] = True
                            filled["imputed_from_epsilon"] = float(args.fill_missing_from_epsilon)
                            rows.append(filled)
                            rows_by_key[key] = filled
                        else:
                            empty_row: Dict[str, object] = {
                                "epsilon": float(eps),
                                "dataset": dataset,
                                "model_name": args.model_name,
                                "approach_key": approach_key,
                                "approach_canonical": canonical,
                                "approach_display": display_approach_name(canonical),
                                "approach_resolved": approach_key,
                                "explainer": explainer,
                                "num_curve_information_files": 0,
                                "num_bad_json": 0,
                                "num_valid_confidence": 0,
                                "mean_confidence": None,
                                "std_confidence": None,
                                "min_confidence": None,
                                "max_confidence": None,
                                "is_imputed": True,
                                "imputed_from_epsilon": None,
                            }
                            rows.append(empty_row)
                            rows_by_key[key] = empty_row

    rows.sort(
        key=lambda r: (
            str(r["dataset"]),
            str(r["explainer"]),
            str(r["approach_canonical"]),
            float(r["epsilon"]),
        )
    )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "epsilon",
            "dataset",
            "model_name",
            "approach_key",
            "approach_canonical",
            "approach_display",
            "approach_resolved",
            "explainer",
            "num_curve_information_files",
            "num_bad_json",
            "num_valid_confidence",
            "mean_confidence",
            "std_confidence",
            "min_confidence",
            "max_confidence",
            "is_imputed",
            "imputed_from_epsilon",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "results_parent": str(results_parent),
        "epsilon_pattern": args.epsilon_pattern,
        "model_name": args.model_name,
        "datasets": datasets,
        "approaches_requested": approaches,
        "include_clean_baseline": bool(args.include_clean_baseline),
        "clean_folder": args.clean_folder,
        "clean_epsilon": float(args.clean_epsilon),
        "target_epsilons": selected_epsilons,
        "complete_only": bool(args.complete_only),
        "emit_missing_grid": bool(args.emit_missing_grid),
        "fill_missing_from_epsilon": args.fill_missing_from_epsilon,
        "rows": rows,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
