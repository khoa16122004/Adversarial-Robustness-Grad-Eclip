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


def metrics_for_batch(clip_model, zero_shot_weights, image_batch, pred_label, gt_label, batch_size):
    pred_prob_list = []
    gt_prob_list = []

    with torch.no_grad():
        total = image_batch.shape[0]
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            feats = clip_model.encode_image(image_batch[start:end])
            logits = 100.0 * feats @ zero_shot_weights
            probs = logits.softmax(dim=-1)

            pred_prob = probs[:, pred_label]
            gt_prob = probs[:, gt_label]

            pred_prob_list.append(pred_prob.detach().cpu().numpy())
            gt_prob_list.append(gt_prob.detach().cpu().numpy())

    return np.concatenate(pred_prob_list), np.concatenate(gt_prob_list)


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
    explain_label,
    pred_label,
    gt_label,
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

    target_text_embedding = zero_shot_weights[:, explain_label].unsqueeze(0)
    target_texts = [IMAGENET_CLASSNAMES[explain_label]]

    if precomputed_heatmap is not None:
        heatmap = normalize_heatmap(precomputed_heatmap)
    else:
        heatmap = explainer.generate_hm(method, image, target_text_embedding, target_texts, resize).detach().cpu().numpy()
        heatmap = normalize_heatmap(heatmap)

    original_tensor = normalize_only(image).unsqueeze(0).to(device)
    original_pred_prob, original_gt_prob = metrics_for_batch(
        clip_model,
        zero_shot_weights,
        original_tensor,
        pred_label,
        gt_label,
        batch_size,
    )

    black_img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    black_tensor = normalize_only(black_img).unsqueeze(0).to(device)
    black_pred_prob, black_gt_prob = metrics_for_batch(
        clip_model,
        zero_shot_weights,
        black_tensor,
        pred_label,
        gt_label,
        batch_size,
    )

    del_batch, del_fraction = deletion_sequence(image, heatmap, normalize_only, device, l_steps, gap)
    ins_batch, ins_fraction = insertion_sequence(image, heatmap, normalize_only, device, l_steps, gap)

    del_pred_prob, del_gt_prob = metrics_for_batch(
        clip_model,
        zero_shot_weights,
        del_batch,
        pred_label,
        gt_label,
        batch_size,
    )
    ins_pred_prob, ins_gt_prob = metrics_for_batch(
        clip_model,
        zero_shot_weights,
        ins_batch,
        pred_label,
        gt_label,
        batch_size,
    )

    x_del = np.concatenate(([0.0], del_fraction))
    x_ins = np.concatenate(([0.0], ins_fraction))

    return {
        "x_del": x_del,
        "x_ins": x_ins,
        "del_pred_prob": np.concatenate(([original_pred_prob[0]], del_pred_prob)),
        "del_gt_prob": np.concatenate(([original_gt_prob[0]], del_gt_prob)),
        "ins_pred_prob": np.concatenate(([black_pred_prob[0]], ins_pred_prob)),
        "ins_gt_prob": np.concatenate(([black_gt_prob[0]], ins_gt_prob)),
    }


def step_mean(curve):
    if curve.shape[0] <= 1:
        return float(curve[0])
    return float(np.mean(curve[1:]))


def compute_scalar_scores(curves):
    del_pred_prob = step_mean(curves["del_pred_prob"])
    del_gt_prob = step_mean(curves["del_gt_prob"])
    ins_pred_prob = step_mean(curves["ins_pred_prob"])
    ins_gt_prob = step_mean(curves["ins_gt_prob"])
    return {
        "deletion": {"pred_prob": del_pred_prob, "gt_prob": del_gt_prob},
        "insertion": {"pred_prob": ins_pred_prob, "gt_prob": ins_gt_prob},
        "imd": {"pred_prob": ins_pred_prob - del_pred_prob, "gt_prob": ins_gt_prob - del_gt_prob},
    }


def init_metrics_bucket():
    return {
        "deletion": {"pred_prob": 0.0, "gt_prob": 0.0},
        "insertion": {"pred_prob": 0.0, "gt_prob": 0.0},
        "imd": {"pred_prob": 0.0, "gt_prob": 0.0},
        "count": 0,
    }


def add_metrics(bucket, scores):
    for metric_name in ["deletion", "insertion", "imd"]:
        bucket[metric_name]["pred_prob"] += float(scores[metric_name]["pred_prob"])
        bucket[metric_name]["gt_prob"] += float(scores[metric_name]["gt_prob"])
    bucket["count"] += 1


def finalize_metrics(bucket):
    c = max(1, int(bucket["count"]))
    out = {}
    for metric_name in ["deletion", "insertion", "imd"]:
        out[metric_name] = {
            "pred_prob": bucket[metric_name]["pred_prob"] / c,
            "gt_prob": bucket[metric_name]["gt_prob"] / c,
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
        # print(files)
        if "metadata.json" not in files:
            continue

        metadata_path = os.path.join(root, "metadata.json")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            # raise
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
                "gt_label": int(meta["gt_label"]) if meta.get("gt_label") is not None else None,
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

    axes[0, 0].plot(x_del, clean_mean["del_pred_prob"], marker="o", label="Deletion Pred Prob")
    axes[0, 0].plot(x_del, clean_mean["del_gt_prob"], marker="o", label="Deletion GT Prob")
    axes[0, 0].set_title("Clean - Deletion")
    axes[0, 0].set_xlabel("Removed Pixel Ratio")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_ins, clean_mean["ins_pred_prob"], marker="o", label="Insertion Pred Prob")
    axes[0, 1].plot(x_ins, clean_mean["ins_gt_prob"], marker="o", label="Insertion GT Prob")
    axes[0, 1].set_title("Clean - Insertion")
    axes[0, 1].set_xlabel("Inserted Pixel Ratio")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x_del, adv_mean["del_pred_prob"], marker="o", label="Deletion Pred Prob")
    axes[1, 0].plot(x_del, adv_mean["del_gt_prob"], marker="o", label="Deletion GT Prob")
    axes[1, 0].set_title("Adv - Deletion")
    axes[1, 0].set_xlabel("Removed Pixel Ratio")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x_ins, adv_mean["ins_pred_prob"], marker="o", label="Insertion Pred Prob")
    axes[1, 1].plot(x_ins, adv_mean["ins_gt_prob"], marker="o", label="Insertion GT Prob")
    axes[1, 1].set_title("Adv - Insertion")
    axes[1, 1].set_xlabel("Inserted Pixel Ratio")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flatten():
        ax.legend(loc="best", frameon=False)

    plt.suptitle(f"Pred/GT probability curves | method={method}")
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


def curve_auc(x, y):
    return float(np.trapz(y, x))


def plot_sample_auc_panels(method, sample_folder, clean_curves, adv_curves, out_name):
    raise
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=120)
    panels = [
        (
            axes[0, 0],
            clean_curves["x_del"],
            clean_curves["del_pred_prob"],
            clean_curves["del_gt_prob"],
            "Deletion",
        ),
        (
            axes[0, 1],
            clean_curves["x_ins"],
            clean_curves["ins_pred_prob"],
            clean_curves["ins_gt_prob"],
            "Insertion",
        ),
        (
            axes[1, 0],
            adv_curves["x_del"],
            adv_curves["del_pred_prob"],
            adv_curves["del_gt_prob"],
            "Deletion",
        ),
        (
            axes[1, 1],
            adv_curves["x_ins"],
            adv_curves["ins_pred_prob"],
            adv_curves["ins_gt_prob"],
            "Insertion",
        ),
    ]

    for ax, x, pred_y, gt_y, title in panels:
        ax.plot(x, pred_y, color="#1f77b4", linewidth=1.5, label="pred")
        ax.fill_between(x, pred_y, color="#1f77b4", alpha=0.25)
        ax.plot(x, gt_y, color="#ff7f0e", linewidth=1.5, label="gt")
        ax.fill_between(x, gt_y, color="#ff7f0e", alpha=0.20)
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        pred_auc = curve_auc(x, pred_y)
        gt_auc = curve_auc(x, gt_y)
        ax.text(0.5, 0.56, f"Pred AUC={pred_auc:.3f}", ha="center", va="center", fontsize=11, transform=ax.transAxes)
        ax.text(0.5, 0.44, f"GT AUC={gt_auc:.3f}", ha="center", va="center", fontsize=11, transform=ax.transAxes)

    plt.suptitle(f"{method} | pred/gt probability", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(sample_folder, out_name)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def evaluate_method(method, args, entries, folder_to_label, clip_model, explainer, zero_shot_weights, normalize_only, device):
    clean_del_pred_prob_sum = None
    clean_del_gt_prob_sum = None
    clean_ins_pred_prob_sum = None
    clean_ins_gt_prob_sum = None

    adv_del_pred_prob_sum = None
    adv_del_gt_prob_sum = None
    adv_ins_pred_prob_sum = None
    adv_ins_gt_prob_sum = None

    x_del = None
    x_ins = None
    count = 0

    clean_table = init_metrics_bucket()
    adv_table = init_metrics_bucket()

    for entry in tqdm(entries, desc=f"Evaluating ({method})"):
        gt_label = entry.get("gt_label")
        if gt_label is None:
            gt_label = infer_gt_label(entry, args.attack_root, folder_to_label)
        if gt_label is None:
            continue

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
                explain_label=clean_pred_label,
                pred_label=clean_pred_label,
                gt_label=gt_label,
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
                explain_label=adv_pred_label,
                pred_label=adv_pred_label,
                gt_label=gt_label,
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
        

        if args.save_per_sample_plots:
            print('Khoa')
            sample_folder = os.path.dirname(entry["metadata_path"])
            sample_plot_name = f"{args.output_prefix}_{method}_sample_auc.png"
            try:
                saved_path = plot_sample_auc_panels(
                    method=method,
                    sample_folder=sample_folder,
                    clean_curves=clean_curves,
                    adv_curves=adv_curves,
                    out_name=sample_plot_name,
                )
                print(f"Saved sample AUC ({method}): {saved_path}")
            except Exception:
                print("Dog")
                raise
                pass

        if x_del is None:
            x_del = clean_curves["x_del"]
            x_ins = clean_curves["x_ins"]

            clean_del_pred_prob_sum = np.zeros_like(clean_curves["del_pred_prob"], dtype=np.float64)
            clean_del_gt_prob_sum = np.zeros_like(clean_curves["del_gt_prob"], dtype=np.float64)
            clean_ins_pred_prob_sum = np.zeros_like(clean_curves["ins_pred_prob"], dtype=np.float64)
            clean_ins_gt_prob_sum = np.zeros_like(clean_curves["ins_gt_prob"], dtype=np.float64)

            adv_del_pred_prob_sum = np.zeros_like(adv_curves["del_pred_prob"], dtype=np.float64)
            adv_del_gt_prob_sum = np.zeros_like(adv_curves["del_gt_prob"], dtype=np.float64)
            adv_ins_pred_prob_sum = np.zeros_like(adv_curves["ins_pred_prob"], dtype=np.float64)
            adv_ins_gt_prob_sum = np.zeros_like(adv_curves["ins_gt_prob"], dtype=np.float64)

        clean_del_pred_prob_sum += clean_curves["del_pred_prob"]
        clean_del_gt_prob_sum += clean_curves["del_gt_prob"]
        clean_ins_pred_prob_sum += clean_curves["ins_pred_prob"]
        clean_ins_gt_prob_sum += clean_curves["ins_gt_prob"]

        adv_del_pred_prob_sum += adv_curves["del_pred_prob"]
        adv_del_gt_prob_sum += adv_curves["del_gt_prob"]
        adv_ins_pred_prob_sum += adv_curves["ins_pred_prob"]
        adv_ins_gt_prob_sum += adv_curves["ins_gt_prob"]

        add_metrics(clean_table, compute_scalar_scores(clean_curves))
        add_metrics(adv_table, compute_scalar_scores(adv_curves))
        count += 1

    if count == 0:
        raise RuntimeError(f"No samples were successfully evaluated for method: {method}")

    clean_mean = {
        "del_pred_prob": clean_del_pred_prob_sum / count,
        "del_gt_prob": clean_del_gt_prob_sum / count,
        "ins_pred_prob": clean_ins_pred_prob_sum / count,
        "ins_gt_prob": clean_ins_gt_prob_sum / count,
    }
    adv_mean = {
        "del_pred_prob": adv_del_pred_prob_sum / count,
        "del_gt_prob": adv_del_gt_prob_sum / count,
        "ins_pred_prob": adv_ins_pred_prob_sum / count,
        "ins_gt_prob": adv_ins_gt_prob_sum / count,
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
            fieldnames=["method", "split", "metric", "pred_prob", "gt_prob", "num_samples_evaluated"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Deletion/insertion/IMD evaluation with pred and GT probabilities for PGD attack outputs"
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
        "--label-map",
        default="imgnet1k_label.json",
        help="ImageNet class index JSON for inferring GT labels by folder",
    )
    parser.add_argument(
        "--save-per-method-plots",
        action="store_true",
        help="Also save one 2x2 curve figure per method",
    )
    parser.add_argument(
        "--save-per-sample-plots",
        action="store_true",
        help="Save a 2x2 Deletion/Insertion AUC chart in each sample folder",
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

    folder_to_label = load_imagenet_label_map(args.label_map)

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
        "target": "pred_and_gt_prob",
        "clip_model": args.clip_model,
        "methods": {},
    }
    scalar_rows = []

    for method in methods:
        result = evaluate_method(
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
                "deletion_pred_prob": result["clean_mean"]["del_pred_prob"].tolist(),
                "deletion_gt_prob": result["clean_mean"]["del_gt_prob"].tolist(),
                "insertion_pred_prob": result["clean_mean"]["ins_pred_prob"].tolist(),
                "insertion_gt_prob": result["clean_mean"]["ins_gt_prob"].tolist(),
            },
            "adv": {
                "deletion_pred_prob": result["adv_mean"]["del_pred_prob"].tolist(),
                "deletion_gt_prob": result["adv_mean"]["del_gt_prob"].tolist(),
                "insertion_pred_prob": result["adv_mean"]["ins_pred_prob"].tolist(),
                "insertion_gt_prob": result["adv_mean"]["ins_gt_prob"].tolist(),
            },
            "table_metrics": {
                "columns": ["pred_prob", "gt_prob"],
                "target": "pred_and_gt_prob",
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
                        "pred_prob": vals["pred_prob"],
                        "gt_prob": vals["gt_prob"],
                        "num_samples_evaluated": result["num_samples_evaluated"],
                    }
                )

    pred_prob_comp_path = os.path.join(args.attack_root, f"{args.output_prefix}_methods_pred_prob.png")
    gt_prob_comp_path = os.path.join(args.attack_root, f"{args.output_prefix}_methods_gt_prob.png")
    plot_multi_method_comparison(results_by_method, "pred_prob", pred_prob_comp_path)
    plot_multi_method_comparison(results_by_method, "gt_prob", gt_prob_comp_path)

    summary["comparison_figures"] = {
        "pred_prob": pred_prob_comp_path,
        "gt_prob": gt_prob_comp_path,
    }

    summary_path = os.path.join(args.attack_root, f"{args.output_prefix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    scalar_csv_path = os.path.join(args.attack_root, f"{args.output_prefix}_scalar_metrics.csv")
    save_scalar_csv(scalar_csv_path, scalar_rows)

    print(f"Saved summary: {summary_path}")
    print(f"Saved scalar csv: {scalar_csv_path}")
    print(f"Saved comparison (pred_prob): {pred_prob_comp_path}")
    print(f"Saved comparison (gt_prob): {gt_prob_comp_path}")


if __name__ == "__main__":
    main()
