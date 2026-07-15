import argparse
import csv
import json
import os

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from clip_utils import build_zero_shot_classifier
from generate_emap import CLIPExplainRunner
from imagenet_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES


def setup_plot_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")


def make_grids(h, w):
    shifts_x = torch.arange(0, w, 1)
    shifts_y = torch.arange(0, h, 1)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return torch.stack((shift_x, shift_y), dim=1)


def random_pixel(image, poses):
    random_patch = torch.rand(len(poses), 3).numpy() * 255.0
    xs, ys = zip(*poses)
    image[ys, xs, :] = random_patch
    return image


def add_pixel(image, input_img, poses):
    xs, ys = zip(*poses)
    input_img[ys, xs, :] = image[ys, xs, :]
    return input_img


def deletion_sequence(image, heatmap, normalize_only, device, l_steps, cal_gap):
    image_array = np.array(image).copy()
    image_array.setflags(write=1)

    h, w = heatmap.shape
    grids = make_grids(h, w)
    order = np.argsort(-heatmap.reshape(-1))
    area = h * w
    pixel_once = max(1, int(area / (2 * l_steps)))

    tensors = []
    fractions = []

    for step in range(1, l_steps + 1):
        slice_idx = order[(step - 1) * pixel_once : step * pixel_once]
        image_array = random_pixel(image_array, grids[slice_idx].tolist())

        if step % cal_gap == 0:
            pil_image = Image.fromarray(np.uint8(image_array))
            tensors.append(normalize_only(pil_image).to(device).unsqueeze(0))
            fractions.append(min(1.0, (step * pixel_once) / area))

    return torch.cat(tensors, dim=0), np.array(fractions)


def insertion_sequence(image, heatmap, normalize_only, device, l_steps, cal_gap):
    image_array = np.array(image).copy()

    h, w = heatmap.shape
    grids = make_grids(h, w)
    order = np.argsort(-heatmap.reshape(-1))
    area = h * w
    pixel_once = max(1, int(area / (2 * l_steps)))

    input_img = np.zeros(image_array.shape, dtype=np.uint8)
    tensors = []
    fractions = []

    for step in range(1, l_steps + 1):
        slice_idx = order[(step - 1) * pixel_once : step * pixel_once]
        input_img = add_pixel(image_array, input_img, grids[slice_idx].tolist())

        if step % cal_gap == 0:
            pil_image = Image.fromarray(np.uint8(input_img))
            tensors.append(normalize_only(pil_image).to(device).unsqueeze(0))
            fractions.append(min(1.0, (step * pixel_once) / area))

    return torch.cat(tensors, dim=0), np.array(fractions)


def metrics_for_batch(clip_model, zero_shot_weights, image_batch, target_label, target_text_embedding, batch_size):
    acc_list = []
    cos_list = []

    with torch.no_grad():
        text_feat = target_text_embedding / target_text_embedding.norm(dim=-1, keepdim=True)
        total = image_batch.shape[0]
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            feats = clip_model.encode_image(image_batch[start:end])
            logits = 100.0 * feats @ zero_shot_weights
            probs = logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1)
            acc = (preds == target_label).float()

            feats = feats / feats.norm(dim=-1, keepdim=True)
            cos = (feats @ text_feat.T).squeeze(-1)

            acc_list.append(acc.detach().cpu().numpy())
            cos_list.append(cos.detach().cpu().numpy())

    return np.concatenate(acc_list), np.concatenate(cos_list)


def normalize_heatmap(hm):
    heatmap = np.asarray(hm, dtype=np.float32)
    if heatmap.ndim == 3:
        heatmap = heatmap[..., 0]
    heatmap = heatmap.copy()
    heatmap -= heatmap.min()
    denom = heatmap.max()
    if denom > 0:
        heatmap /= denom
    return heatmap


def build_curves_for_variant(
    image,
    target_label,
    method,
    clip_model,
    explainer,
    zero_shot_weights,
    normalize_only,
    device,
    l_steps,
    gap,
    batch_size,
    precomputed_heatmap=None,
):
    w, h = image.size
    resize = Resize((h, w))

    target_text_embedding = zero_shot_weights[:, target_label].unsqueeze(0)
    target_texts = [IMAGENET_CLASSNAMES[target_label]]

    if precomputed_heatmap is not None:
        heatmap = normalize_heatmap(precomputed_heatmap)
    else:
        heatmap = explainer.generate_hm(method, image, target_text_embedding, target_texts, resize).detach().cpu().numpy()
        heatmap = normalize_heatmap(heatmap)

    original_tensor = normalize_only(image).unsqueeze(0).to(device)
    original_acc, original_cos = metrics_for_batch(
        clip_model,
        zero_shot_weights,
        original_tensor,
        target_label,
        target_text_embedding,
        batch_size,
    )

    black_img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    black_tensor = normalize_only(black_img).unsqueeze(0).to(device)
    black_acc, black_cos = metrics_for_batch(
        clip_model,
        zero_shot_weights,
        black_tensor,
        target_label,
        target_text_embedding,
        batch_size,
    )

    del_batch, del_fraction = deletion_sequence(image, heatmap, normalize_only, device, l_steps, gap)
    ins_batch, ins_fraction = insertion_sequence(image, heatmap, normalize_only, device, l_steps, gap)

    del_acc, del_cos = metrics_for_batch(
        clip_model,
        zero_shot_weights,
        del_batch,
        target_label,
        target_text_embedding,
        batch_size,
    )
    ins_acc, ins_cos = metrics_for_batch(
        clip_model,
        zero_shot_weights,
        ins_batch,
        target_label,
        target_text_embedding,
        batch_size,
    )

    x_del = np.concatenate(([0.0], del_fraction))
    x_ins = np.concatenate(([0.0], ins_fraction))

    return {
        "x_del": x_del,
        "x_ins": x_ins,
        "del_acc": np.concatenate(([original_acc[0]], del_acc)),
        "del_cos": np.concatenate(([original_cos[0]], del_cos)),
        "ins_acc": np.concatenate(([black_acc[0]], ins_acc)),
        "ins_cos": np.concatenate(([black_cos[0]], ins_cos)),
    }


def step_mean(curve):
    if curve.shape[0] <= 1:
        return float(curve[0])
    return float(np.mean(curve[1:]))


def compute_scalar_scores(curves):
    del_acc = step_mean(curves["del_acc"])
    del_cos = step_mean(curves["del_cos"])
    ins_acc = step_mean(curves["ins_acc"])
    ins_cos = step_mean(curves["ins_cos"])
    return {
        "deletion": {"acc": del_acc, "cosine": del_cos},
        "insertion": {"acc": ins_acc, "cosine": ins_cos},
        "imd": {"acc": ins_acc - del_acc, "cosine": ins_cos - del_cos},
    }


def init_metrics_bucket():
    return {
        "deletion": {"acc": 0.0, "cosine": 0.0},
        "insertion": {"acc": 0.0, "cosine": 0.0},
        "imd": {"acc": 0.0, "cosine": 0.0},
        "count": 0,
    }


def add_metrics(bucket, scores):
    for metric_name in ["deletion", "insertion", "imd"]:
        bucket[metric_name]["acc"] += float(scores[metric_name]["acc"])
        bucket[metric_name]["cosine"] += float(scores[metric_name]["cosine"])
    bucket["count"] += 1


def finalize_metrics(bucket):
    c = max(1, int(bucket["count"]))
    out = {}
    for metric_name in ["deletion", "insertion", "imd"]:
        out[metric_name] = {
            "acc": bucket[metric_name]["acc"] / c,
            "cosine": bucket[metric_name]["cosine"] / c,
        }
    out["count"] = bucket["count"]
    return out


def load_image_from_tensor_or_png(tensor_path, png_path):
    if tensor_path and os.path.exists(tensor_path):
        arr = np.load(tensor_path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = np.clip(arr, 0.0, 1.0)
            return Image.fromarray((arr * 255.0).astype(np.uint8))
    return Image.open(png_path).convert("RGB")


def find_sample_entries(attack_root):
    entries = []
    for root, _, files in os.walk(attack_root):
        if "metadata.json" not in files:
            continue

        metadata_path = os.path.join(root, "metadata.json")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        clean_rel = meta.get("clean_image_path")
        adv_rel = meta.get("adv_image_path")
        clean_tensor_rel = meta.get("clean_tensor_01_path")
        adv_tensor_rel = meta.get("adv_tensor_01_path")
        clean_map_npy_rel = meta.get("clean_map_npy_path")
        adv_map_npy_rel = meta.get("adv_map_npy_path")
        clean_map_npy_paths = meta.get("clean_map_npy_paths")
        adv_map_npy_paths = meta.get("adv_map_npy_paths")
        clean_pred_label = meta.get("from_pred_label")
        adv_pred_label = meta.get("to_pred_label")

        if clean_rel is None or adv_rel is None or clean_pred_label is None:
            continue

        clean_path = os.path.join(attack_root, str(clean_rel))
        adv_path = os.path.join(attack_root, str(adv_rel))
        if not os.path.exists(clean_path) or not os.path.exists(adv_path):
            continue

        entries.append(
            {
                "metadata_path": metadata_path,
                "clean_path": clean_path,
                "adv_path": adv_path,
                "clean_tensor_path": os.path.join(attack_root, str(clean_tensor_rel)) if clean_tensor_rel else None,
                "adv_tensor_path": os.path.join(attack_root, str(adv_tensor_rel)) if adv_tensor_rel else None,
                "clean_map_npy_path": os.path.join(attack_root, str(clean_map_npy_rel)) if clean_map_npy_rel else None,
                "adv_map_npy_path": os.path.join(attack_root, str(adv_map_npy_rel)) if adv_map_npy_rel else None,
                "clean_map_npy_paths": clean_map_npy_paths if isinstance(clean_map_npy_paths, dict) else None,
                "adv_map_npy_paths": adv_map_npy_paths if isinstance(adv_map_npy_paths, dict) else None,
                "clean_pred_label": int(clean_pred_label),
                "adv_pred_label": int(adv_pred_label) if adv_pred_label is not None else int(clean_pred_label),
            }
        )

    return entries


def discover_methods(attack_root, entries):
    methods = set()

    for entry in entries:
        path_map = entry.get("clean_map_npy_paths") or {}
        methods.update(path_map.keys())

    if methods:
        return sorted([m for m in methods if m])

    for root, _, files in os.walk(attack_root):
        for name in files:
            if name.startswith("clean_") and name.endswith(".npy"):
                methods.add(name[len("clean_") : -len(".npy")])

    return sorted([m for m in methods if m])


def get_precomputed_map_path(entry, method, split, attack_root):
    if split == "clean":
        paths = entry.get("clean_map_npy_paths") or {}
        default_path = entry.get("clean_map_npy_path")
    else:
        paths = entry.get("adv_map_npy_paths") or {}
        default_path = entry.get("adv_map_npy_path")

    if method in paths:
        return os.path.join(attack_root, str(paths[method]))
    return default_path


def plot_single_method_curves(method, clean_mean, adv_mean, x_del, x_ins, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

    axes[0, 0].plot(x_del, clean_mean["del_acc"], marker="o", label="Deletion Acc")
    axes[0, 0].plot(x_del, clean_mean["del_cos"], marker="o", label="Deletion Cos")
    axes[0, 0].set_title("Clean - Deletion")
    axes[0, 0].set_xlabel("Removed Pixel Ratio")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_ins, clean_mean["ins_acc"], marker="o", label="Insertion Acc")
    axes[0, 1].plot(x_ins, clean_mean["ins_cos"], marker="o", label="Insertion Cos")
    axes[0, 1].set_title("Clean - Insertion")
    axes[0, 1].set_xlabel("Inserted Pixel Ratio")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x_del, adv_mean["del_acc"], marker="o", label="Deletion Acc")
    axes[1, 0].plot(x_del, adv_mean["del_cos"], marker="o", label="Deletion Cos")
    axes[1, 0].set_title("Adv - Deletion")
    axes[1, 0].set_xlabel("Removed Pixel Ratio")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x_ins, adv_mean["ins_acc"], marker="o", label="Insertion Acc")
    axes[1, 1].plot(x_ins, adv_mean["ins_cos"], marker="o", label="Insertion Cos")
    axes[1, 1].set_title("Adv - Insertion")
    axes[1, 1].set_xlabel("Inserted Pixel Ratio")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flatten():
        ax.legend(loc="best", frameon=False)

    plt.suptitle(f"Pred-only curves | method={method}")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_multi_method_comparison(results_by_method, score_key, out_path):
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
    plt.suptitle(f"Pred-only Method Comparison ({y_label})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.0, 0.0, 0.88, 0.96])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_method(method, args, entries, clip_model, explainer, zero_shot_weights, normalize_only, device):
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

    clean_table = init_metrics_bucket()
    adv_table = init_metrics_bucket()

    for entry in tqdm(entries, desc=f"Evaluating ({method})"):
        try:
            clean_img = load_image_from_tensor_or_png(entry.get("clean_tensor_path"), entry["clean_path"])
            adv_img = load_image_from_tensor_or_png(entry.get("adv_tensor_path"), entry["adv_path"])
        except Exception:
            continue

        clean_pred_label = int(entry["clean_pred_label"])
        adv_pred_label = int(entry["adv_pred_label"])

        clean_map = None
        adv_map = None

        clean_map_path = get_precomputed_map_path(entry, method, "clean", args.attack_root)
        adv_map_path = get_precomputed_map_path(entry, method, "adv", args.attack_root)

        if clean_map_path and os.path.exists(clean_map_path):
            try:
                clean_map = np.load(clean_map_path)
            except Exception:
                clean_map = None

        if adv_map_path and os.path.exists(adv_map_path):
            try:
                adv_map = np.load(adv_map_path)
            except Exception:
                adv_map = None

        try:
            clean_curves = build_curves_for_variant(
                image=clean_img,
                target_label=clean_pred_label,
                method=method,
                clip_model=clip_model,
                explainer=explainer,
                zero_shot_weights=zero_shot_weights,
                normalize_only=normalize_only,
                device=device,
                l_steps=args.L,
                gap=args.gap,
                batch_size=args.batch_size,
                precomputed_heatmap=clean_map,
            )
            adv_curves = build_curves_for_variant(
                image=adv_img,
                target_label=adv_pred_label,
                method=method,
                clip_model=clip_model,
                explainer=explainer,
                zero_shot_weights=zero_shot_weights,
                normalize_only=normalize_only,
                device=device,
                l_steps=args.L,
                gap=args.gap,
                batch_size=args.batch_size,
                precomputed_heatmap=adv_map if adv_pred_label == clean_pred_label else None,
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

        add_metrics(clean_table, compute_scalar_scores(clean_curves))
        add_metrics(adv_table, compute_scalar_scores(adv_curves))
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
        "clean_metrics": finalize_metrics(clean_table),
        "adv_metrics": finalize_metrics(adv_table),
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
        description="Pred-only deletion/insertion/IMD evaluation and plotting for PGD attack outputs"
    )
    parser.add_argument("--attack-root", required=True, help="Folder produced by pgd_attack.py")
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
    parser.add_argument("--output-prefix", default="pred_eval", help="Prefix for summary outputs")
    parser.add_argument(
        "--save-per-method-plots",
        action="store_true",
        help="Also save one 2x2 curve figure per method",
    )
    args = parser.parse_args()

    setup_plot_style()

    if args.L < 1 or args.gap < 1:
        raise ValueError("--L and --gap must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    entries = find_sample_entries(args.attack_root)
    if args.max_samples is not None:
        entries = entries[: args.max_samples]
    if not entries:
        raise ValueError("No valid sample folders found under attack root.")

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        methods = discover_methods(args.attack_root, entries)

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
        "target": "pred",
        "clip_model": args.clip_model,
        "methods": {},
    }
    scalar_rows = []

    for method in methods:
        result = evaluate_method(
            method=method,
            args=args,
            entries=entries,
            clip_model=clip_model,
            explainer=explainer,
            zero_shot_weights=zero_shot_weights,
            normalize_only=normalize_only,
            device=device,
        )
        results_by_method[method] = result

        clean_fig_path = None
        if args.save_per_method_plots:
            clean_fig_path = os.path.join(args.attack_root, f"{args.output_prefix}_{method}_curves.png")
            plot_single_method_curves(
                method=method,
                clean_mean=result["clean_mean"],
                adv_mean=result["adv_mean"],
                x_del=result["x_del"],
                x_ins=result["x_ins"],
                out_path=clean_fig_path,
            )

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
                "target": "pred",
                "clean": result["clean_metrics"],
                "adv": result["adv_metrics"],
            },
            "figure_path": clean_fig_path,
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
    plot_multi_method_comparison(results_by_method, "cos", cosine_comp_path)
    plot_multi_method_comparison(results_by_method, "acc", acc_comp_path)

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
