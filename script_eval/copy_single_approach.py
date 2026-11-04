import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ATTACK_ROWS: List[Tuple[str, str]] = [
    ("IOA", "ins_optimized_output"),
    ("DOA", "del_optimized_output"),
    ("FOA", "ins_del_optimized_output"),
]

ATTACK_ALIASES: Dict[str, List[str]] = {
    "ins_optimized_output": ["ins_optimized_output", "ins_optimize_output"],
    "del_optimized_output": ["del_optimized_output", "del_optimize_output"],
    "ins_del_optimized_output": ["ins_del_optimized_output", "ins_del_optimize_output", "final_output"],
}

LABEL_TO_KEY: Dict[str, str] = {label.lower(): key for label, key in ATTACK_ROWS}


def _normalize_key(text: str) -> str:
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def discover_datasets(result_root: Path, selected: Optional[List[str]]) -> List[str]:
    if selected:
        return selected
    return sorted([p.name for p in result_root.iterdir() if p.is_dir()])


def parse_approach_to_key(value: str) -> str:
    norm = _normalize_key(value)

    # Accept human labels: IOA/DOA/FOA
    if norm in LABEL_TO_KEY:
        return LABEL_TO_KEY[norm]

    # Accept canonical keys and any alias.
    for canonical, aliases in ATTACK_ALIASES.items():
        if norm == _normalize_key(canonical):
            return canonical
        for alias in aliases:
            if norm == _normalize_key(alias):
                return canonical

    raise ValueError(
        "Unknown --approach. Use IOA/DOA/FOA or one of these folders: "
        "ins_optimized_output, del_optimized_output, ins_del_optimized_output, "
        "ins_optimize_output, del_optimize_output, ins_del_optimize_output, final_output"
    )


def resolve_existing_approach_folder(dataset_model_root: Path, canonical_key: str) -> Optional[str]:
    candidates = ATTACK_ALIASES.get(canonical_key, [canonical_key])
    for candidate in candidates:
        if (dataset_model_root / candidate).exists():
            return candidate
    if (dataset_model_root / canonical_key).exists():
        return canonical_key
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy only one selected approach (IOA/DOA/FOA) from result tree into a new "
            "output tree while preserving dataset/model/approach/sample structure."
        )
    )
    parser.add_argument(
        "--result-root",
        type=str,
        required=True,
        help="Input result root, e.g. E:/.../result",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root where filtered tree will be written",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="CLIP_ViTB16",
        help="Model folder under each dataset (default: CLIP_ViTB16)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional datasets list (default: auto-detect under result-root)",
    )
    parser.add_argument(
        "--approach",
        type=str,
        required=True,
        help=(
            "Approach to keep. Accepts IOA/DOA/FOA or folder keys/aliases "
            "(ins_optimized_output, del_optimized_output, ins_del_optimized_output, ...)."
        ),
    )
    parser.add_argument(
        "--rename-to-canonical",
        action="store_true",
        help=(
            "If set, destination approach folder name is canonical key "
            "(e.g., ins_del_optimized_output) even if source uses an alias."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing destination folder for copied approach.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result_root = Path(args.result_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not result_root.exists() or not result_root.is_dir():
        raise FileNotFoundError(f"result-root not found: {result_root}")

    canonical_key = parse_approach_to_key(args.approach)
    datasets = discover_datasets(result_root, args.datasets)
    if not datasets:
        raise RuntimeError(f"No datasets found under: {result_root}")

    print(f"Result root: {result_root}")
    print(f"Output root: {output_root}")
    print(f"Model name : {args.model_name}")
    print(f"Datasets   : {', '.join(datasets)}")
    print(f"Approach   : {canonical_key}")
    print(f"Overwrite  : {args.overwrite}")
    print(f"RenameCanon: {args.rename_to_canonical}")

    copied_count = 0
    skipped_count = 0

    for dataset in datasets:
        dataset_model_root = result_root / dataset / args.model_name
        if not dataset_model_root.exists() or not dataset_model_root.is_dir():
            print(f"[SKIP] dataset={dataset}: missing {dataset_model_root}")
            skipped_count += 1
            continue

        resolved_src_name = resolve_existing_approach_folder(dataset_model_root, canonical_key)
        if resolved_src_name is None:
            print(
                f"[SKIP] dataset={dataset}: no folder for approach={canonical_key} "
                f"under {dataset_model_root}"
            )
            skipped_count += 1
            continue

        src_dir = dataset_model_root / resolved_src_name
        dst_name = canonical_key if args.rename_to_canonical else resolved_src_name
        dst_dir = output_root / dataset / args.model_name / dst_name

        if dst_dir.exists():
            if not args.overwrite:
                print(f"[SKIP] dataset={dataset}: destination exists {dst_dir} (use --overwrite)")
                skipped_count += 1
                continue
            shutil.rmtree(dst_dir)

        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, dst_dir)

        copied_count += 1
        print(f"[OK] dataset={dataset}: {src_dir} -> {dst_dir}")

    print("=" * 80)
    print(f"Copied datasets: {copied_count}")
    print(f"Skipped datasets: {skipped_count}")


if __name__ == "__main__":
    main()
