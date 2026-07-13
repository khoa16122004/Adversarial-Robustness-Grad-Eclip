import argparse
import json
import os

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize
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
    grids = torch.stack((shift_x, shift_y), dim=1)
    return grids


def random_pixel(image, poses):
    random_patch = torch.rand(len(poses), 3).numpy() * 255.0
    xs, ys = zip(*poses)
    image[ys, xs, :] = random_patch
    return image


def add_pixel(image, input_img, poses):
    xs, ys = zip(*poses)
    input_img[ys, xs, :] = image[ys, xs, :]
    return input_img


def deletion_sequence(image, heatmap, preprocess, device, l_steps, cal_gap):
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
            tensors.append(preprocess(pil_image).to(device).unsqueeze(0))
            fractions.append(min(1.0, (step * pixel_once) / area))

    return torch.cat(tensors, dim=0), np.array(fractions)


def insertion_sequence(image, heatmap, preprocess, device, l_steps, cal_gap):
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
            tensors.append(preprocess(pil_image).to(device).unsqueeze(0))
            fractions.append(min(1.0, (step * pixel_once) / area))

    return torch.cat(tensors, dim=0), np.array(fractions)


def class_probabilities(clipmodel, zero_shot_weights, image_batch, class_idx):
    with torch.no_grad():
        image_features = clipmodel.encode_image(image_batch)
        logits = 100.0 * image_features @ zero_shot_weights
        probs = logits.softmax(dim=-1)
    return probs[:, class_idx].detach().cpu().numpy()


def load_imagenet_label_map(index_json_path):
    with open(index_json_path, "r", encoding="utf-8") as f:
        class_dict = json.load(f)
    folder_to_label = {}
    for label_str, values in class_dict.items():
        folder_to_label[values[0]] = int(label_str)
    return folder_to_label


def collect_image_items(data_path, folder_to_label, max_images=None):
    items = []
    for folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder not in folder_to_label:
            continue
        label = folder_to_label[folder]
        for name in sorted(os.listdir(folder_path)):
            image_path = os.path.join(folder_path, name)
            if os.path.isfile(image_path):
                items.append((image_path, label))
                if max_images is not None and len(items) >= max_images:
                    return items
    return items


def main():
    parser = argparse.ArgumentParser(description="Standalone ImageNet-val deletion/insertion curves")
    parser.add_argument("--data-path", required=True, help="Path to ImageNet val folder")
    parser.add_argument("--index-json", default="imagenet_class_index.json", help="Path to imagenet_class_index.json")
    parser.add_argument(
        "--methods",
        default="eclip,eclip-wo-ksim,game,maskclip,gradcam,rollout,surgery,m2ib,rise",
        help="Comma-separated explain methods",
    )
    parser.add_argument("--target", choices=["gt", "pred"], default="gt", help="Class used to generate heatmap")
    parser.add_argument("--eval-class", choices=["gt", "pred"], default="gt", help="Class whose probability is plotted")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on images")
    parser.add_argument("--L", type=int, default=100, help="Total perturbation steps")
    parser.add_argument("--gap", type=int, default=10, help="Evaluate every N steps")
    parser.add_argument("--output", default="imagenet_val_curves.png", help="Output plot path")
    parser.add_argument("--summary", default="imagenet_val_curves_summary.txt", help="Summary txt path")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    clipmodel, preprocess = clip.load("ViT-B/16", device=device)
    explainer = CLIPExplainRunner(clipmodel=clipmodel, preprocess=preprocess, device=device)

    zero_shot_weights = build_zero_shot_classifier(
        clipmodel,
        classnames=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        num_classes_per_batch=10,
        device=device,
        use_tqdm=True,
    )

    folder_to_label = load_imagenet_label_map(args.index_json)
    image_items = collect_image_items(args.data_path, folder_to_label, max_images=args.max_images)
    if not image_items:
        raise ValueError("No image found for evaluation.")

    expected_points = args.L // args.gap
    curve_len = expected_points + 1

    del_sum = {m: np.zeros(curve_len, dtype=np.float64) for m in methods}
    ins_sum = {m: np.zeros(curve_len, dtype=np.float64) for m in methods}
    counts = {m: 0 for m in methods}
    x_del = None
    x_ins = None

    for image_path, gt_label in tqdm(image_items, desc="Evaluating"):
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        w, h = img.size
        if min(w, h) > 640:
            scale = min(w, h) / 640.0
            img = img.resize((int(w / scale), int(h / scale)))
            w, h = img.size
        resize = Resize((h, w))

        image_tensor = preprocess(img).to(device).unsqueeze(0)
        with torch.no_grad():
            image_features = clipmodel.encode_image(image_tensor)
            logits = 100.0 * image_features @ zero_shot_weights
            probs = logits.softmax(dim=-1)
            pred_label = int(probs.argmax(dim=-1).item())

        explain_label = gt_label if args.target == "gt" else pred_label
        eval_label = gt_label if args.eval_class == "gt" else pred_label

        original_prob = float(probs[0, eval_label].detach().cpu().item())
        txt_embedding = zero_shot_weights[:, explain_label].unsqueeze(0)
        txts = [IMAGENET_CLASSNAMES[explain_label]]

        for hm_type in methods:
            try:
                heatmap = explainer.generate_hm(hm_type, img, txt_embedding, txts, resize).detach().cpu().numpy()
                del_batch, del_fraction = deletion_sequence(img, heatmap, preprocess, device, args.L, args.gap)
                ins_batch, ins_fraction = insertion_sequence(img, heatmap, preprocess, device, args.L, args.gap)

                del_probs = class_probabilities(clipmodel, zero_shot_weights, del_batch, eval_label)
                ins_probs = class_probabilities(clipmodel, zero_shot_weights, ins_batch, eval_label)

                del_curve = np.concatenate(([original_prob], del_probs))
                ins_curve = np.concatenate((ins_probs, [original_prob]))

                if len(del_curve) != curve_len or len(ins_curve) != curve_len:
                    continue

                del_sum[hm_type] += del_curve
                ins_sum[hm_type] += ins_curve
                counts[hm_type] += 1

                if x_del is None:
                    x_del = np.concatenate(([0.0], del_fraction))
                if x_ins is None:
                    x_ins = np.concatenate((ins_fraction, [1.0]))
            except Exception:
                continue

    valid_methods = [m for m in methods if counts[m] > 0]
    if not valid_methods:
        raise RuntimeError("No valid curves were produced. Check method dependencies.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    summary_lines = [f"num_images_input={len(image_items)}"]

    for hm_type in valid_methods:
        del_mean = del_sum[hm_type] / counts[hm_type]
        ins_mean = ins_sum[hm_type] / counts[hm_type]

        axes[0].plot(x_del, del_mean, marker="o", label=hm_type)
        axes[1].plot(x_ins, ins_mean, marker="o", label=hm_type)

        del_auc = float(np.trapz(del_mean, x_del))
        ins_auc = float(np.trapz(ins_mean, x_ins))
        summary_lines.append(
            f"{hm_type}\tcount={counts[hm_type]}\tdel_auc={del_auc:.6f}\tins_auc={ins_auc:.6f}"
        )

    axes[0].set_title("Deletion Curve (mean on ImageNet val)")
    axes[0].set_xlabel("Removed Pixel Ratio")
    axes[0].set_ylabel("P(eval class)")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Insertion Curve (mean on ImageNet val)")
    axes[1].set_xlabel("Inserted Pixel Ratio")
    axes[1].set_ylabel("P(eval class)")
    axes[1].grid(True, alpha=0.3)

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.suptitle(f"ImageNet val | explain={args.target} | eval={args.eval_class}")
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Saved plot: {args.output}")
    print(f"Saved summary: {args.summary}")


if __name__ == "__main__":
    main()
