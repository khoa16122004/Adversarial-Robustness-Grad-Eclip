import argparse
import os
import numpy as np
import torch
import clip
from PIL import Image
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

from generate_emap import CLIPExplainRunner


def make_grids(h, w):
    shifts_x = torch.arange(0, w, 1)
    shifts_y = torch.arange(0, h, 1)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    grids = torch.stack((shift_x, shift_y), dim=1)
    return grids


def compute_cosine_scores(clipmodel, image_batch, text_embedding):
    with torch.no_grad():
        image_features = clipmodel.encode_image(image_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        cosine = (image_features @ text_features.T).squeeze(-1)
    return cosine.detach().cpu().numpy()


def random_pixel(image, poses):
    random_patch = torch.rand(len(poses), 3).numpy() * 255.0
    xs, ys = zip(*poses)
    image[ys, xs, :] = random_patch
    return image


def add_pixel(image, input_img, poses):
    xs, ys = zip(*poses)
    input_img[ys, xs, :] = image[ys, xs, :]
    return input_img


def deletion_sequence(image, heatmap, preprocess, device, L, cal_gap):
    image_array = np.array(image).copy()
    image_array.setflags(write=1)

    h, w = heatmap.shape
    grids = make_grids(h, w)
    order = np.argsort(-heatmap.reshape(-1))
    area = h * w
    pixel_once = max(1, int(area / (2 * L)))

    tensors = []
    fractions = []
    vis_images = []
    step_ids = []

    for step in range(1, L + 1):
        slice_idx = order[(step - 1) * pixel_once : step * pixel_once]
        image_array = random_pixel(image_array, grids[slice_idx].tolist())

        if step % cal_gap == 0:
            pil_image = Image.fromarray(np.uint8(image_array))
            tensors.append(preprocess(pil_image).to(device).unsqueeze(0))
            fractions.append(min(1.0, (step * pixel_once) / area))
            vis_images.append(np.uint8(image_array.copy()))
            step_ids.append(step)

    return torch.cat(tensors, dim=0), np.array(fractions), vis_images, step_ids


def insertion_sequence(image, heatmap, preprocess, device, L, cal_gap):
    image_array = np.array(image).copy()

    h, w = heatmap.shape
    grids = make_grids(h, w)
    order = np.argsort(-heatmap.reshape(-1))
    area = h * w
    pixel_once = max(1, int(area / (2 * L)))

    input_img = np.zeros(image_array.shape, dtype=np.uint8)
    tensors = []
    fractions = []
    vis_images = []
    step_ids = []

    for step in range(1, L + 1):
        slice_idx = order[(step - 1) * pixel_once : step * pixel_once]
        input_img = add_pixel(image_array, input_img, grids[slice_idx].tolist())

        if step % cal_gap == 0:
            pil_image = Image.fromarray(np.uint8(input_img))
            tensors.append(preprocess(pil_image).to(device).unsqueeze(0))
            fractions.append(min(1.0, (step * pixel_once) / area))
            vis_images.append(np.uint8(input_img.copy()))
            step_ids.append(step)

    return torch.cat(tensors, dim=0), np.array(fractions), vis_images, step_ids


def save_process_figure(method_name, del_images, ins_images, del_scores, ins_scores, del_steps, ins_steps, output_path, max_cols=6):
    def select_indices(total, max_keep):
        if total <= max_keep:
            return list(range(total))
        return np.linspace(0, total - 1, num=max_keep, dtype=int).tolist()

    del_idx = select_indices(len(del_images), max_cols)
    ins_idx = select_indices(len(ins_images), max_cols)
    ncols = max(len(del_idx), len(ins_idx))
    ncols = max(1, ncols)

    fig, axes = plt.subplots(2, ncols, figsize=(3.0 * ncols, 5.8), dpi=120)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    fig.suptitle(f"Method: {method_name}", fontsize=16, y=1.02)
    axes[0, 0].text(
        -0.12,
        1.08,
        "Deletion Process",
        transform=axes[0, 0].transAxes,
        fontsize=14,
        ha="left",
        va="center",
    )
    axes[1, 0].text(
        -0.12,
        1.08,
        "Insertion Process",
        transform=axes[1, 0].transAxes,
        fontsize=14,
        ha="left",
        va="center",
    )

    for col in range(ncols):
        ax_del = axes[0, col]
        if col < len(del_idx):
            i = del_idx[col]
            ax_del.imshow(del_images[i])
            ax_del.set_title(f"Cos: {del_scores[i]:.2f}", fontsize=12)
            ax_del.set_xlabel(f"Step: {del_steps[i]}", fontsize=12)
        ax_del.set_xticks([])
        ax_del.set_yticks([])

        ax_ins = axes[1, col]
        if col < len(ins_idx):
            i = ins_idx[col]
            ax_ins.imshow(ins_images[i])
            ax_ins.set_title(f"Cos: {ins_scores[i]:.2f}", fontsize=12)
            ax_ins.set_xlabel(f"Step: {ins_steps[i]}", fontsize=12)
        ax_ins.set_xticks([])
        ax_ins.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def get_heatmap(explainer, hm_type, image, text_embedding, text, resize):
    emap = explainer.generate_hm(hm_type, image, text_embedding, [text], resize)
    return emap.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Single-image deletion/insertion cosine-similarity curves")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--text", required=True, help="Text prompt for cosine similarity")
    parser.add_argument(
        "--methods",
        default="eclip,eclip-wo-ksim,game,maskclip,gradcam,rollout,surgery,m2ib,rise",
        help="Comma-separated explain methods",
    )
    parser.add_argument("--L", type=int, default=100, help="Total perturbation steps")
    parser.add_argument("--gap", type=int, default=10, help="Evaluate every N steps")
    parser.add_argument("--output", default="single_image_curves.png", help="Output plot path")
    parser.add_argument("--process-cols", type=int, default=6, help="Number of columns shown in process visualization")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    clipmodel, preprocess = clip.load("ViT-B/16", device=device)
    explainer = CLIPExplainRunner(clipmodel=clipmodel, preprocess=preprocess, device=device)

    image = Image.open(args.image).convert("RGB")
    w, h = image.size
    resize = Resize((h, w))
    image_array = np.array(image).copy()
    black_array = np.zeros_like(image_array, dtype=np.uint8)

    text_tokens = clip.tokenize([args.text]).to(device)
    with torch.no_grad():
        text_embedding = clipmodel.encode_text(text_tokens)

    image_tensor = preprocess(image).to(device).unsqueeze(0)
    original_cosine = float(compute_cosine_scores(clipmodel, image_tensor, text_embedding)[0])
    black_tensor = preprocess(Image.fromarray(black_array)).to(device).unsqueeze(0)
    black_cosine = float(compute_cosine_scores(clipmodel, black_tensor, text_embedding)[0])

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    deletion_curves = {}
    insertion_curves = {}
    x_del = None
    x_ins = None

    print(f"Using text prompt: {args.text}")

    for hm_type in methods:
        print(f"[method] {hm_type}")
        heatmap = get_heatmap(explainer, hm_type, image, text_embedding, args.text, resize)

        del_batch, del_fraction, del_vis, del_steps = deletion_sequence(image, heatmap, preprocess, device, args.L, args.gap)
        ins_batch, ins_fraction, ins_vis, ins_steps = insertion_sequence(image, heatmap, preprocess, device, args.L, args.gap)

        del_scores = compute_cosine_scores(clipmodel, del_batch, text_embedding)
        ins_scores = compute_cosine_scores(clipmodel, ins_batch, text_embedding)

        deletion_curves[hm_type] = np.concatenate(([original_cosine], del_scores))
        insertion_curves[hm_type] = np.concatenate((ins_scores, [original_cosine]))
        x_del = np.concatenate(([0.0], del_fraction))
        x_ins = np.concatenate((ins_fraction, [1.0]))

        del_process_images = [image_array] + del_vis
        del_process_scores = [original_cosine] + del_scores.tolist()
        del_process_steps = [0] + del_steps

        ins_process_images = [black_array] + ins_vis
        ins_process_scores = [black_cosine] + ins_scores.tolist()
        ins_process_steps = [0] + ins_steps

        output_root, output_ext = os.path.splitext(args.output)
        if output_ext == "":
            output_ext = ".png"
        process_output = f"{output_root}_{hm_type}_process{output_ext}"
        save_process_figure(
            method_name=hm_type,
            del_images=del_process_images,
            ins_images=ins_process_images,
            del_scores=del_process_scores,
            ins_scores=ins_process_scores,
            del_steps=del_process_steps,
            ins_steps=ins_process_steps,
            output_path=process_output,
            max_cols=max(1, args.process_cols),
        )
        print(f"Saved process figure to: {process_output}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    for hm_type in methods:
        axes[0].plot(x_del, deletion_curves[hm_type], marker="o", label=hm_type)
        axes[1].plot(x_ins, insertion_curves[hm_type], marker="o", label=hm_type)

    axes[0].set_title("Deletion Curve")
    axes[0].set_xlabel("Removed Pixel Ratio")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Insertion Curve")
    axes[1].set_xlabel("Inserted Pixel Ratio")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].grid(True, alpha=0.3)

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.suptitle(f"Image: {args.image} | Text: {args.text}")
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved curve figure to: {args.output}")

    print("\nAUC summary (trapezoid):")
    for hm_type in methods:
        del_auc = np.trapz(deletion_curves[hm_type], x_del)
        ins_auc = np.trapz(insertion_curves[hm_type], x_ins)
        print(f"{hm_type:>14s} | deletion AUC={del_auc:.6f} | insertion AUC={ins_auc:.6f}")


if __name__ == "__main__":
    main()
