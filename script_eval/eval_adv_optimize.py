import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze optimization traces from adv_score curve_information.json files "
            "and save global + per-sample summaries."
        )
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Root folder to recursively search (e.g., del_optimize_output)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save global outputs. Default: <input-root>/eval_optimize_summary",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="curve_information.json",
        help="Filename pattern for recursive search",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures",
    )
    parser.add_argument(
        "--save-per-folder-summary",
        action="store_true",
        help="Save optimize summarize json/png next to each source file",
    )
    parser.add_argument(
        "--per-folder-json-name",
        type=str,
        default="optimize_information.json",
        help="Output json filename template for each folder",
    )
    parser.add_argument(
        "--per-folder-image-name",
        type=str,
        default="optimize_summary.png",
        help="Output image filename template for each folder",
    )
    return parser.parse_args()


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_values(values: List[float]) -> Optional[Dict[str, float]]:
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


def align_to_min_length(curves: List[np.ndarray]) -> Tuple[np.ndarray, int, int, int]:
    lengths = [c.size for c in curves]
    min_len = min(lengths)
    max_len = max(lengths)
    truncated_count = sum(1 for x in lengths if x != min_len)
    stacked = np.stack([c[:min_len] for c in curves], axis=0)
    return stacked, min_len, max_len, truncated_count


def build_trace_from_data(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    raw_losses = data.get("pgd_loss", [])
    trace = data.get("pgd_trace", [])

    losses: List[float] = []
    if isinstance(raw_losses, list):
        for v in raw_losses:
            fv = _safe_float(v)
            if fv is not None:
                losses.append(fv)

    if not losses and isinstance(trace, list):
        for item in trace:
            if not isinstance(item, dict):
                continue
            fv = _safe_float(item.get("loss"))
            if fv is not None:
                losses.append(fv)

    pt_means: List[float] = []
    if isinstance(trace, list):
        for item in trace:
            if not isinstance(item, dict):
                continue
            p_t = item.get("p_t", [])
            if isinstance(p_t, list) and p_t:
                vals = [x for x in (_safe_float(v) for v in p_t) if x is not None]
                if vals:
                    pt_means.append(float(np.mean(vals)))
                    continue
            pt_means.append(float("nan"))

    loss_arr = np.asarray(losses, dtype=np.float64)
    pt_arr = np.asarray(pt_means, dtype=np.float64)

    if pt_arr.size == 0 and loss_arr.size > 0:
        pt_arr = np.full(loss_arr.size, np.nan, dtype=np.float64)
    if loss_arr.size == 0 and pt_arr.size > 0:
        loss_arr = np.full(pt_arr.size, np.nan, dtype=np.float64)

    if loss_arr.size != pt_arr.size:
        min_len = min(loss_arr.size, pt_arr.size)
        if min_len <= 0:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        loss_arr = loss_arr[:min_len]
        pt_arr = pt_arr[:min_len]

    return loss_arr, pt_arr


def save_per_folder_summary(
    source_file: Path,
    output_json_name: str,
    output_image_name: str,
    clean_score: Optional[float],
    adv_score: Optional[float],
    adv_pred_retain: Optional[bool],
    loss_arr: np.ndarray,
    pt_arr: np.ndarray,
    dpi: int,
) -> None:
    payload = {
        "source_file": str(source_file),
        "num_steps": int(loss_arr.size),
        "clean_score": clean_score,
        "adv_score": adv_score,
        "score_delta": (
            (adv_score - clean_score)
            if (clean_score is not None and adv_score is not None)
            else None
        ),
        "adv_pred_retain": adv_pred_retain,
        "loss_curve": loss_arr.tolist(),
        "p_t_mean_curve": pt_arr.tolist(),
    }

    out_json = source_file.parent / output_json_name
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if loss_arr.size == 0:
        return

    steps = np.arange(1, loss_arr.size + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axes[0].plot(steps, loss_arr, linewidth=2.0, color="tab:red")
    axes[0].set_title("PGD Loss")
    axes[0].set_xlabel("Step")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, pt_arr, linewidth=2.0, color="tab:blue")
    axes[1].set_title("Mean p_t")
    axes[1].set_xlabel("Step")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(source_file.parent / output_image_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist or is not a directory: {input_root}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_root / "eval_optimize_summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_root.rglob(args.pattern))
    if not json_files:
        raise FileNotFoundError(
            f"No files matched pattern '{args.pattern}' under: {input_root}"
        )

    parse_errors: List[str] = []
    clean_scores: List[float] = []
    adv_scores: List[float] = []
    score_deltas: List[float] = []
    retain_flags: List[bool] = []
    loss_curves: List[np.ndarray] = []
    pt_curves: List[np.ndarray] = []
    per_sample_rows: List[Dict[str, object]] = []
    per_folder_saved = 0

    for path in json_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # pylint: disable=broad-except
            parse_errors.append(f"{path}: {exc}")
            continue

        clean_score = _safe_float(data.get("clean_score"))
        if clean_score is None:
            clean_score = _safe_float(data.get("clean_prob"))

        adv_score = _safe_float(data.get("adv_score"))
        if adv_score is None:
            adv_score = _safe_float(data.get("adv_prob"))

        if clean_score is not None:
            clean_scores.append(clean_score)
        if adv_score is not None:
            adv_scores.append(adv_score)
        if clean_score is not None and adv_score is not None:
            score_deltas.append(adv_score - clean_score)

        retain_flag = data.get("adv_pred_retain")
        retain_bool: Optional[bool]
        if isinstance(retain_flag, bool):
            retain_bool = retain_flag
            retain_flags.append(retain_bool)
        else:
            retain_bool = None

        loss_arr, pt_arr = build_trace_from_data(data)
        if loss_arr.size > 0:
            loss_curves.append(loss_arr)
            pt_curves.append(pt_arr)

        per_sample_rows.append(
            {
                "file": str(path),
                "num_steps": int(loss_arr.size),
                "clean_score": clean_score,
                "adv_score": adv_score,
                "score_delta": (
                    (adv_score - clean_score)
                    if (clean_score is not None and adv_score is not None)
                    else None
                ),
                "adv_pred_retain": retain_bool,
                "loss_start": float(loss_arr[0]) if loss_arr.size > 0 else None,
                "loss_end": float(loss_arr[-1]) if loss_arr.size > 0 else None,
                "pt_start": float(pt_arr[0]) if pt_arr.size > 0 else None,
                "pt_end": float(pt_arr[-1]) if pt_arr.size > 0 else None,
            }
        )

        if args.save_per_folder_summary:
            save_per_folder_summary(
                source_file=path,
                output_json_name=args.per_folder_json_name,
                output_image_name=args.per_folder_image_name,
                clean_score=clean_score,
                adv_score=adv_score,
                adv_pred_retain=retain_bool,
                loss_arr=loss_arr,
                pt_arr=pt_arr,
                dpi=args.dpi,
            )
            per_folder_saved += 1

    summary: Dict[str, object] = {
        "input_root": str(input_root),
        "pattern": args.pattern,
        "num_files_scanned": len(json_files),
        "num_parse_errors": len(parse_errors),
        "parse_errors": parse_errors,
        "clean_score": summarize_values(clean_scores),
        "adv_score": summarize_values(adv_scores),
        "score_delta_adv_minus_clean": summarize_values(score_deltas),
        "adv_pred_retain": {
            "count": int(len(retain_flags)),
            "retain_count": int(sum(1 for x in retain_flags if x)),
            "retain_rate": (
                float(sum(1 for x in retain_flags if x) / len(retain_flags))
                if retain_flags
                else None
            ),
        },
        "per_folder_summaries_saved": int(per_folder_saved),
        "per_sample": per_sample_rows,
    }

    if loss_curves:
        loss_aligned, min_len, max_len, truncated_count = align_to_min_length(loss_curves)
        pt_aligned = np.stack([c[:min_len] for c in pt_curves], axis=0)

        mean_loss = np.nanmean(loss_aligned, axis=0)
        std_loss = np.nanstd(loss_aligned, axis=0)
        mean_pt = np.nanmean(pt_aligned, axis=0)
        std_pt = np.nanstd(pt_aligned, axis=0)

        steps = np.arange(1, min_len + 1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(steps, mean_loss, linewidth=2.0, color="tab:red", label="mean loss")
        axes[0].fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, alpha=0.25, color="tab:red")
        axes[0].set_title("PGD Loss (mean +/- std)")
        axes[0].set_xlabel("Step")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, mean_pt, linewidth=2.0, color="tab:blue", label="mean p_t")
        axes[1].fill_between(steps, mean_pt - std_pt, mean_pt + std_pt, alpha=0.25, color="tab:blue")
        axes[1].set_title("Mean p_t (mean +/- std)")
        axes[1].set_xlabel("Step")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        mean_curve_plot = output_dir / "optimize_mean_curves.png"
        fig.savefig(mean_curve_plot, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        np.save(output_dir / "optimize_mean_loss.npy", mean_loss)
        np.save(output_dir / "optimize_std_loss.npy", std_loss)
        np.save(output_dir / "optimize_mean_pt.npy", mean_pt)
        np.save(output_dir / "optimize_std_pt.npy", std_pt)

        summary["trace_summary"] = {
            "num_curves": int(len(loss_curves)),
            "curve_length_min": int(min_len),
            "curve_length_max": int(max_len),
            "num_truncated_to_min_length": int(truncated_count),
            "mean_curve_plot": str(mean_curve_plot),
            "mean_loss_path": str(output_dir / "optimize_mean_loss.npy"),
            "std_loss_path": str(output_dir / "optimize_std_loss.npy"),
            "mean_pt_path": str(output_dir / "optimize_mean_pt.npy"),
            "std_pt_path": str(output_dir / "optimize_std_pt.npy"),
        }
    else:
        summary["trace_summary"] = None

    summary_path = output_dir / "optimize_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Scanned files: {len(json_files)}")
    print(f"Parse errors: {len(parse_errors)}")
    if args.save_per_folder_summary:
        print(f"Per-folder summaries saved: {per_folder_saved}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
