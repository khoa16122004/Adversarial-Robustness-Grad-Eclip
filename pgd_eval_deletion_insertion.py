import argparse
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
        target_label = meta.get("from_pred_label")

        if clean_rel is None or adv_rel is None or target_label is None:
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
                "target_label": int(target_label),
            }
        )

    return entries


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
        heatmap = np.asarray(precomputed_heatmap, dtype=np.float32)
        if heatmap.ndim == 3:
            heatmap = heatmap[..., 0]
    else:
        heatmap = explainer.generate_hm(method, image, target_text_embedding, target_texts, resize).detach().cpu().numpy()

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


def load_image_from_tensor_or_png(tensor_path, png_path):
    if tensor_path and os.path.exists(tensor_path):
        arr = np.load(tensor_path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = np.clip(arr, 0.0, 1.0)
            return Image.fromarray((arr * 255.0).astype(np.uint8))
    return Image.open(png_path).convert("RGB")


def plot_variant_figure(title, x_del, x_ins, del_acc, del_cos, ins_acc, ins_cos, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    axes[0].plot(x_del, del_acc, marker="o", label="Accuracy@orig_class")
    axes[0].plot(x_del, del_cos, marker="o", label="Cosine@orig_class_text")
    axes[0].set_title("Deletion")
    axes[0].set_xlabel("Removed Pixel Ratio")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_ins, ins_acc, marker="o", label="Accuracy@orig_class")
    axes[1].plot(x_ins, ins_cos, marker="o", label="Cosine@orig_class_text")
    axes[1].set_title("Insertion")
    axes[1].set_xlabel("Inserted Pixel Ratio")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate deletion/insertion on PGD output folder for clean/adv images with accuracy and cosine curves"
    )
    parser.add_argument("--attack-root", required=True, help="Folder produced by pgd_attack.py")
    parser.add_argument("--method", default="eclip", help="Explain method for heatmap generation")
    parser.add_argument("--clip-model", default="ViT-B/16", help="CLIP model name")
    parser.add_argument("--L", type=int, default=100, help="Total perturbation steps")
    parser.add_argument("--gap", type=int, default=10, help="Evaluate every N steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for metric inference")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on sample folders")
    parser.add_argument("--output-prefix", default="pgd_eval", help="Prefix for figure/json outputs")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

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

    for entry in tqdm(entries, desc="Evaluating deletion/insertion"):
        try:
            clean_img = load_image_from_tensor_or_png(entry.get("clean_tensor_path"), entry["clean_path"])
            adv_img = load_image_from_tensor_or_png(entry.get("adv_tensor_path"), entry["adv_path"])
        except Exception:
            continue

        target_label = int(entry["target_label"])

        clean_map = None
        adv_map = None
        clean_map_path = entry.get("clean_map_npy_path")
        adv_map_path = entry.get("adv_map_npy_path")
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
                target_label=target_label,
                method=args.method,
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
                target_label=target_label,
                method=args.method,
                clip_model=clip_model,
                explainer=explainer,
                zero_shot_weights=zero_shot_weights,
                normalize_only=normalize_only,
                device=device,
                l_steps=args.L,
                gap=args.gap,
                batch_size=args.batch_size,
                precomputed_heatmap=adv_map,
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

        count += 1

    if count == 0:
        raise RuntimeError("No samples were successfully evaluated.")

    clean_del_acc_mean = clean_del_acc_sum / count
    clean_del_cos_mean = clean_del_cos_sum / count
    clean_ins_acc_mean = clean_ins_acc_sum / count
    clean_ins_cos_mean = clean_ins_cos_sum / count

    adv_del_acc_mean = adv_del_acc_sum / count
    adv_del_cos_mean = adv_del_cos_sum / count
    adv_ins_acc_mean = adv_ins_acc_sum / count
    adv_ins_cos_mean = adv_ins_cos_sum / count

    clean_fig_path = os.path.join(args.attack_root, f"{args.output_prefix}_clean.png")
    adv_fig_path = os.path.join(args.attack_root, f"{args.output_prefix}_adv.png")

    plot_variant_figure(
        title=f"Clean Images | method={args.method} | n={count}",
        x_del=x_del,
        x_ins=x_ins,
        del_acc=clean_del_acc_mean,
        del_cos=clean_del_cos_mean,
        ins_acc=clean_ins_acc_mean,
        ins_cos=clean_ins_cos_mean,
        out_path=clean_fig_path,
    )

    plot_variant_figure(
        title=f"Adversarial Images | method={args.method} | n={count}",
        x_del=x_del,
        x_ins=x_ins,
        del_acc=adv_del_acc_mean,
        del_cos=adv_del_cos_mean,
        ins_acc=adv_ins_acc_mean,
        ins_cos=adv_ins_cos_mean,
        out_path=adv_fig_path,
    )

    summary = {
        "attack_root": args.attack_root,
        "method": args.method,
        "clip_model": args.clip_model,
        "num_samples_evaluated": count,
        "x_del": x_del.tolist(),
        "x_ins": x_ins.tolist(),
        "clean": {
            "deletion_accuracy": clean_del_acc_mean.tolist(),
            "deletion_cosine": clean_del_cos_mean.tolist(),
            "insertion_accuracy": clean_ins_acc_mean.tolist(),
            "insertion_cosine": clean_ins_cos_mean.tolist(),
        },
        "adv": {
            "deletion_accuracy": adv_del_acc_mean.tolist(),
            "deletion_cosine": adv_del_cos_mean.tolist(),
            "insertion_accuracy": adv_ins_acc_mean.tolist(),
            "insertion_cosine": adv_ins_cos_mean.tolist(),
        },
    }

    summary_path = os.path.join(args.attack_root, f"{args.output_prefix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved clean figure: {clean_fig_path}")
    print(f"Saved adv figure: {adv_fig_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Evaluated samples: {count}")


if __name__ == "__main__":
    main()
