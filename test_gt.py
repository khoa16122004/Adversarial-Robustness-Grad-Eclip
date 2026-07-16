import argparse
import csv
import json
import os

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor

import test as base
from clip_utils import build_zero_shot_classifier
from generate_emap import CLIPExplainRunner
from imagenet_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES


def load_imagenet_label_map(index_json_path):
    with open(index_json_path, "r", encoding="utf-8") as f:
        class_dict = json.load(f)

    if not isinstance(class_dict, dict) or len(class_dict) == 0:
        raise ValueError(f"Invalid label json format: {index_json_path}")

    sample_key = next(iter(class_dict.keys()))
    folder_to_label = {}

    if str(sample_key).isdigit():
        for label_str, values in class_dict.items():
            if not isinstance(values, list) or len(values) < 1:
                continue
            folder_to_label[str(values[0])] = int(label_str)
        return folder_to_label

    for wnid, values in class_dict.items():
        if isinstance(values, list) and len(values) > 0:
            folder_to_label[str(wnid)] = int(values[0])
        elif isinstance(values, int):
            folder_to_label[str(wnid)] = int(values)

    if not folder_to_label:
        raise ValueError(f"Could not parse label mapping from: {index_json_path}")

    return folder_to_label


def infer_gt_label(entry, attack_root, folder_to_label):
    rel = os.path.relpath(entry["clean_path"], attack_root).replace("\\", "/")
    folder = rel.split("/")[0]
    if folder in folder_to_label:
        return int(folder_to_label[folder])
    return None


def plot_multi_method_comparison_gt(results_by_method, score_key, out_path):
    methods = list(results_by_method.keys())
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(methods)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=140, sharex="col")

    for method, result in results_by_method.items():
        x_del = result["x_del"]
        x_ins = result["x_ins"]
        clean = result["clean_mean"]
        adv = result["adv_mean"]

        style = {
            "linewidth": 2.0,
            "marker": "o",
            "markersize": 4.0,
            "alpha": 0.95,
            "color": colors[method],
            "label": method,
        }
        axes[0, 0].plot(x_del, clean[f"del_{score_key}"], **style)
        axes[0, 1].plot(x_ins, clean[f"ins_{score_key}"], **style)
        axes[1, 0].plot(x_del, adv[f"del_{score_key}"], **style)
        axes[1, 1].plot(x_ins, adv[f"ins_{score_key}"], **style)

    axes[0, 0].set_title("Clean - Deletion")
    axes[0, 1].set_title("Clean - Insertion")
    axes[1, 0].set_title("Adv - Deletion")
    axes[1, 1].set_title("Adv - Insertion")

    axes[0, 0].set_xlabel("Removed Pixel Ratio")
    axes[1, 0].set_xlabel("Removed Pixel Ratio")
    axes[0, 1].set_xlabel("Inserted Pixel Ratio")
    axes[1, 1].set_xlabel("Inserted Pixel Ratio")

    y_label = "Cosine" if score_key == "cos" else "Accuracy"
    for ax in axes.flatten():
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25, linestyle="--")
        if score_key == "acc":
            ax.set_ylim(0.0, 1.0)
        else:
            ax.relim()
            ax.autoscale_view(scaley=True)
            ax.margins(y=0.08)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False, title="Methods")
    plt.suptitle(f"GT-only Method Comparison ({y_label})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.0, 0.0, 0.88, 0.96])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_method_gt(method, args, entries, folder_to_label, clip_model, explainer, zero_shot_weights, normalize_only, device):
    clean_del_acc_sum = None
    clean_del_cos_sum = None
    clean_ins_acc_sum = None
    clean_ins_cos_sum = None

    adv_del_acc_sum = None
    adv_del_cos_sum = None
    adv_ins_acc_sum = None
    adv_ins_cos_sum = None

    x_del = None
    x_ins = None
    count = 0

    clean_table = base.init_metrics_bucket()
    adv_table = base.init_metrics_bucket()

    for entry in base.tqdm(entries, desc=f"Evaluating GT ({method})"):
        gt_label = infer_gt_label(entry, args.attack_root, folder_to_label)
        if gt_label is None:
            continue

        try:
            clean_img = base.load_image_from_tensor_or_png(entry.get("clean_tensor_path"), entry["clean_path"])
            adv_img = base.load_image_from_tensor_or_png(entry.get("adv_tensor_path"), entry["adv_path"])
        except Exception:
            continue

        try:
            clean_curves = base.build_curves_for_variant(
                image=clean_img,
                target_label=gt_label,
                method=method,
                clip_model=clip_model,
                explainer=explainer,
                zero_shot_weights=zero_shot_weights,
                normalize_only=normalize_only,
                device=device,
                l_steps=args.L,
                gap=args.gap,
                batch_size=args.batch_size,
                precomputed_heatmap=None,
            )
            adv_curves = base.build_curves_for_variant(
                image=adv_img,
                target_label=gt_label,
                method=method,
                clip_model=clip_model,
                explainer=explainer,
                zero_shot_weights=zero_shot_weights,
                normalize_only=normalize_only,
                device=device,
                l_steps=args.L,
                gap=args.gap,
                batch_size=args.batch_size,
                precomputed_heatmap=None,
            )
        except Exception:
            continue

        if x_del is None:
            x_del = clean_curves["x_del"]
            x_ins = clean_curves["x_ins"]

            clean_del_acc_sum = np.zeros_like(clean_curves["del_acc"], dtype=np.float64)
            clean_del_cos_sum = np.zeros_like(clean_curves["del_cos"], dtype=np.float64)
            clean_ins_acc_sum = np.zeros_like(clean_curves["ins_acc"], dtype=np.float64)
            clean_ins_cos_sum = np.zeros_like(clean_curves["ins_cos"], dtype=np.float64)

            adv_del_acc_sum = np.zeros_like(adv_curves["del_acc"], dtype=np.float64)
            adv_del_cos_sum = np.zeros_like(adv_curves["del_cos"], dtype=np.float64)
            adv_ins_acc_sum = np.zeros_like(adv_curves["ins_acc"], dtype=np.float64)
            adv_ins_cos_sum = np.zeros_like(adv_curves["ins_cos"], dtype=np.float64)

        clean_del_acc_sum += clean_curves["del_acc"]
        clean_del_cos_sum += clean_curves["del_cos"]
        clean_ins_acc_sum += clean_curves["ins_acc"]
        clean_ins_cos_sum += clean_curves["ins_cos"]

        adv_del_acc_sum += adv_curves["del_acc"]
        adv_del_cos_sum += adv_curves["del_cos"]
        adv_ins_acc_sum += adv_curves["ins_acc"]
        adv_ins_cos_sum += adv_curves["ins_cos"]

        base.add_metrics(clean_table, base.compute_scalar_scores(clean_curves))
        base.add_metrics(adv_table, base.compute_scalar_scores(adv_curves))
        count += 1

    if count == 0:
        raise RuntimeError(f"No samples were successfully evaluated for method: {method}")

    clean_mean = {
        "del_acc": clean_del_acc_sum / count,
        "del_cos": clean_del_cos_sum / count,
        "ins_acc": clean_ins_acc_sum / count,
        "ins_cos": clean_ins_cos_sum / count,
    }
    adv_mean = {
        "del_acc": adv_del_acc_sum / count,
        "del_cos": adv_del_cos_sum / count,
        "ins_acc": adv_ins_acc_sum / count,
        "ins_cos": adv_ins_cos_sum / count,
    }

    return {
        "method": method,
        "num_samples_evaluated": count,
        "x_del": x_del,
        "x_ins": x_ins,
        "clean_mean": clean_mean,
        "adv_mean": adv_mean,
        "clean_metrics": base.finalize_metrics(clean_table),
        "adv_metrics": base.finalize_metrics(adv_table),
    }


def save_scalar_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "split", "metric", "cosine", "acc", "num_samples_evaluated"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="GT-only deletion/insertion/IMD evaluation and plotting for PGD attack outputs"
    )
    parser.add_argument("--attack-root", required=True, help="Folder produced by pgd_attack.py")
    parser.add_argument("--index-json", default="imgnet1k_label.json", help="Path to ImageNet label mapping json")
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated explain methods; if omitted, auto-discover from metadata/files",
    )
    parser.add_argument("--clip-model", default="ViT-B/16", help="CLIP model name")
    parser.add_argument("--L", type=int, default=100, help="Total perturbation steps")
    parser.add_argument("--gap", type=int, default=10, help="Evaluate every N steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for metric inference")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on sample folders")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--output-prefix", default="gt_eval", help="Prefix for summary outputs")
    args = parser.parse_args()

    base.setup_plot_style()

    if args.L < 1 or args.gap < 1:
        raise ValueError("--L and --gap must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    entries = base.find_sample_entries(args.attack_root)
    if args.max_samples is not None:
        entries = entries[: args.max_samples]
    if not entries:
        raise ValueError("No valid sample folders found under attack root.")

    folder_to_label = load_imagenet_label_map(args.index_json)

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        methods = base.discover_methods(args.attack_root, entries)

    if not methods:
        raise ValueError("No explain methods found.")

    clip_model, preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval()
    explainer = CLIPExplainRunner(clipmodel=clip_model, preprocess=preprocess, device=device)

    zero_shot_weights = build_zero_shot_classifier(
        clip_model,
        classnames=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        device=device,
        use_tqdm=True,
    )

    normalize_only = Compose(
        [
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )

    results_by_method = {}
    summary = {
        "attack_root": args.attack_root,
        "target": "gt",
        "clip_model": args.clip_model,
        "methods": {},
    }
    scalar_rows = []

    for method in methods:
        result = evaluate_method_gt(
            method=method,
            args=args,
            entries=entries,
            folder_to_label=folder_to_label,
            clip_model=clip_model,
            explainer=explainer,
            zero_shot_weights=zero_shot_weights,
            normalize_only=normalize_only,
            device=device,
        )
        results_by_method[method] = result

        summary["methods"][method] = {
            "num_samples_evaluated": result["num_samples_evaluated"],
            "x_del": result["x_del"].tolist(),
            "x_ins": result["x_ins"].tolist(),
            "clean": {
                "deletion_accuracy": result["clean_mean"]["del_acc"].tolist(),
                "deletion_cosine": result["clean_mean"]["del_cos"].tolist(),
                "insertion_accuracy": result["clean_mean"]["ins_acc"].tolist(),
                "insertion_cosine": result["clean_mean"]["ins_cos"].tolist(),
            },
            "adv": {
                "deletion_accuracy": result["adv_mean"]["del_acc"].tolist(),
                "deletion_cosine": result["adv_mean"]["del_cos"].tolist(),
                "insertion_accuracy": result["adv_mean"]["ins_acc"].tolist(),
                "insertion_cosine": result["adv_mean"]["ins_cos"].tolist(),
            },
            "table_metrics": {
                "columns": ["cosine", "acc"],
                "target": "gt",
                "clean": result["clean_metrics"],
                "adv": result["adv_metrics"],
            },
        }

        for split_name, split_metrics in [
            ("clean", result["clean_metrics"]),
            ("adv", result["adv_metrics"]),
        ]:
            for metric_name in ["deletion", "insertion", "imd"]:
                vals = split_metrics[metric_name]
                scalar_rows.append(
                    {
                        "method": method,
                        "split": split_name,
                        "metric": metric_name,
                        "cosine": vals["cosine"],
                        "acc": vals["acc"],
                        "num_samples_evaluated": result["num_samples_evaluated"],
                    }
                )

    cosine_comp_path = os.path.join(args.attack_root, f"{args.output_prefix}_methods_cosine.png")
    acc_comp_path = os.path.join(args.attack_root, f"{args.output_prefix}_methods_acc.png")
    plot_multi_method_comparison_gt(results_by_method, "cos", cosine_comp_path)
    plot_multi_method_comparison_gt(results_by_method, "acc", acc_comp_path)

    summary["comparison_figures"] = {
        "cosine": cosine_comp_path,
        "accuracy": acc_comp_path,
    }

    summary_path = os.path.join(args.attack_root, f"{args.output_prefix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    scalar_csv_path = os.path.join(args.attack_root, f"{args.output_prefix}_scalar_metrics.csv")
    save_scalar_csv(scalar_csv_path, scalar_rows)

    print(f"Saved summary: {summary_path}")
    print(f"Saved scalar csv: {scalar_csv_path}")
    print(f"Saved comparison (cosine): {cosine_comp_path}")
    print(f"Saved comparison (accuracy): {acc_comp_path}")


if __name__ == "__main__":
    main()
