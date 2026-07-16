import argparse
import json
import os

import matplotlib.pyplot as plt


def setup_plot_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")


def derive_default_prefix(summary_path):
    name = os.path.basename(summary_path)
    suffix = "_summary.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return "pred_eval"


def curve_auc(x, y):
    x_vals = [float(v) for v in x]
    y_vals = [float(v) for v in y]
    auc = 0.0
    for i in range(len(x_vals) - 1):
        auc += (x_vals[i + 1] - x_vals[i]) * (y_vals[i] + y_vals[i + 1]) * 0.5
    return auc


def plot_multi_method_comparison(summary_methods, score_key, out_path):
    methods = sorted(summary_methods.keys())
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(methods)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=140, sharex="col")

    for method in methods:
        method_data = summary_methods[method]
        x_del = method_data["x_del"]
        x_ins = method_data["x_ins"]

        clean = method_data["clean"]
        adv = method_data["adv"]

        if score_key == "pred_prob":
            clean_del = clean["deletion_pred_prob"]
            clean_ins = clean["insertion_pred_prob"]
            adv_del = adv["deletion_pred_prob"]
            adv_ins = adv["insertion_pred_prob"]
        else:
            clean_del = clean["deletion_gt_prob"]
            clean_ins = clean["insertion_gt_prob"]
            adv_del = adv["deletion_gt_prob"]
            adv_ins = adv["insertion_gt_prob"]

        style = {
            "linewidth": 2.0,
            "marker": "o",
            "markersize": 4.0,
            "alpha": 0.95,
            "color": colors[method],
            "label": method,
        }

        axes[0, 0].plot(x_del, clean_del, **style)
        axes[0, 1].plot(x_ins, clean_ins, **style)
        axes[1, 0].plot(x_del, adv_del, **style)
        axes[1, 1].plot(x_ins, adv_ins, **style)

    axes[0, 0].set_title("Clean - Deletion")
    axes[0, 1].set_title("Clean - Insertion")
    axes[1, 0].set_title("Adv - Deletion")
    axes[1, 1].set_title("Adv - Insertion")

    axes[0, 0].set_xlabel("Removed Pixel Ratio")
    axes[1, 0].set_xlabel("Removed Pixel Ratio")
    axes[0, 1].set_xlabel("Inserted Pixel Ratio")
    axes[1, 1].set_xlabel("Inserted Pixel Ratio")

    y_label = "Pred Prob" if score_key == "pred_prob" else "GT Prob"
    for ax in axes.flatten():
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_ylim(0.0, 1.0)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False, title="Methods")
    plt.suptitle(f"Pred/GT Prob Method Comparison ({y_label})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.0, 0.0, 0.88, 0.96])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_method_clean_adv_comparison(method, method_data, score_key, out_path):
    x_del = method_data["x_del"]
    x_ins = method_data["x_ins"]
    clean = method_data["clean"]
    adv = method_data["adv"]

    if score_key == "pred_prob":
        clean_del = clean["deletion_pred_prob"]
        clean_ins = clean["insertion_pred_prob"]
        adv_del = adv["deletion_pred_prob"]
        adv_ins = adv["insertion_pred_prob"]
        y_label = "Pred Prob"
    else:
        clean_del = clean["deletion_gt_prob"]
        clean_ins = clean["insertion_gt_prob"]
        adv_del = adv["deletion_gt_prob"]
        adv_ins = adv["insertion_gt_prob"]
        y_label = "GT Prob"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)

    axes[0].plot(x_del, clean_del, marker="o", linewidth=2.2, markersize=4, label="Clean", color="#1f77b4")
    axes[0].plot(x_del, adv_del, marker="o", linewidth=2.2, markersize=4, label="Adv", color="#d62728")
    axes[0].set_title("Deletion")
    axes[0].set_xlabel("Removed Pixel Ratio")
    axes[0].set_ylabel(y_label)
    axes[0].grid(True, alpha=0.25, linestyle="--")

    axes[1].plot(x_ins, clean_ins, marker="o", linewidth=2.2, markersize=4, label="Clean", color="#1f77b4")
    axes[1].plot(x_ins, adv_ins, marker="o", linewidth=2.2, markersize=4, label="Adv", color="#d62728")
    axes[1].set_title("Insertion")
    axes[1].set_xlabel("Inserted Pixel Ratio")
    axes[1].set_ylabel(y_label)
    axes[1].grid(True, alpha=0.25, linestyle="--")

    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)

    clean_del_auc = curve_auc(x_del, clean_del)
    clean_ins_auc = curve_auc(x_ins, clean_ins)
    adv_del_auc = curve_auc(x_del, adv_del)
    adv_ins_auc = curve_auc(x_ins, adv_ins)
    axes[0].text(0.98, 0.04, f"Clean AUC={clean_del_auc:.3f}\nAdv AUC={adv_del_auc:.3f}", ha="right", va="bottom", transform=axes[0].transAxes, fontsize=9)
    axes[1].text(0.98, 0.04, f"Clean AUC={clean_ins_auc:.3f}\nAdv AUC={adv_ins_auc:.3f}", ha="right", va="bottom", transform=axes[1].transAxes, fontsize=9)

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.suptitle(f"{method} | Clean vs Adv ({y_label})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate comparison charts (pred_prob/gt_prob) from pred_eval summary JSON"
    )
    parser.add_argument(
        "--summary-path",
        default="outputs/pgd_adv_images/pred_eval_summary.json",
        help="Path to summary json generated by test.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save chart images. Defaults to summary file directory.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix. Defaults to prefix inferred from summary filename.",
    )
    args = parser.parse_args()

    setup_plot_style()

    if not os.path.exists(args.summary_path):
        raise FileNotFoundError(f"Summary file not found: {args.summary_path}")

    with open(args.summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    summary_methods = summary.get("methods")
    if not isinstance(summary_methods, dict) or not summary_methods:
        raise ValueError("Invalid summary format: missing non-empty 'methods' field")

    output_dir = args.output_dir or os.path.dirname(args.summary_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    output_prefix = args.output_prefix or derive_default_prefix(args.summary_path)

    pred_prob_path = os.path.join(output_dir, f"{output_prefix}_methods_pred_prob.png")
    gt_prob_path = os.path.join(output_dir, f"{output_prefix}_methods_gt_prob.png")

    plot_multi_method_comparison(summary_methods, "pred_prob", pred_prob_path)
    plot_multi_method_comparison(summary_methods, "gt_prob", gt_prob_path)

    per_method_dir = os.path.join(output_dir, f"{output_prefix}_per_method")
    os.makedirs(per_method_dir, exist_ok=True)
    for method, method_data in sorted(summary_methods.items()):
        method_pred_path = os.path.join(per_method_dir, f"{method}_clean_vs_adv_pred_prob.png")
        method_gt_path = os.path.join(per_method_dir, f"{method}_clean_vs_adv_gt_prob.png")
        plot_method_clean_adv_comparison(method, method_data, "pred_prob", method_pred_path)
        plot_method_clean_adv_comparison(method, method_data, "gt_prob", method_gt_path)

    print(f"Saved comparison (pred_prob): {pred_prob_path}")
    print(f"Saved comparison (gt_prob): {gt_prob_path}")
    print(f"Saved per-method clean-vs-adv charts in: {per_method_dir}")


if __name__ == "__main__":
    main()
