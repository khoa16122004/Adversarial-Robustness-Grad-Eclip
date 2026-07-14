import argparse
import json
import os

from bleach import clean
import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm

from clip_utils import build_zero_shot_classifier
from generate_emap import CLIPExplainRunner
from imagenet_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode


def _convert_image_to_rgb(image):
    return image.convert("RGB")

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
                items.append((image_path, folder, name, label))
                if max_images is not None and len(items) >= max_images:
                    return items
    return items


def clip_normalization_stats(device):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    return mean, std


def pgd_minimize_similarity(
    clipmodel,
    image_pil,
    target_text_embedding,
    preprocess,
    device,
    eps=8.0 / 255.0,
    alpha=2.0 / 255.0,
    steps=10,
    random_start=False,
):
    mean, std = clip_normalization_stats(device)

    x0 = preprocess(image_pil).unsqueeze(0).to(device)

    eps_tensor = torch.tensor([eps, eps, eps], device=device).view(1, 3, 1, 1) / std
    alpha_tensor = torch.tensor([alpha, alpha, alpha], device=device).view(1, 3, 1, 1) / std

    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std

    x_adv = x0.clone().detach()
    if random_start:
        x_adv = x_adv - torch.empty_like(x_adv).uniform_(-1.0, 1.0) * eps_tensor
        x_adv = torch.max(torch.min(x_adv, x0 + eps_tensor), x0 - eps_tensor)
        x_adv = torch.max(torch.min(x_adv, upper), lower)

    text_feat = target_text_embedding.detach()
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        image_feat = clipmodel.encode_image(x_adv)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        cosine = (image_feat @ text_feat.T).mean()

        grad = torch.autograd.grad(cosine, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv + alpha_tensor * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps_tensor), x0 - eps_tensor)
        x_adv = torch.max(torch.min(x_adv, upper), lower)
        x_adv = x_adv.detach()

    x_adv_pixel = (x_adv * std + mean).clamp(0.0, 1.0)
    adv_pil = TF.to_pil_image(x_adv_pixel.squeeze(0).cpu())
    return adv_pil


def render_heatmap_panel(clean_img, adv_img, clean_hm, adv_hm, method_name, sample_title, out_path):
    clean_arr = np.asarray(clean_img)
    adv_arr = np.asarray(adv_img)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=120)
    fig.suptitle(f"{sample_title} | {method_name}")

    axes[0, 0].imshow(clean_arr)
    axes[0, 0].set_title("Clean image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(clean_arr)
    axes[0, 1].imshow(clean_hm, cmap="jet", alpha=0.45)
    axes[0, 1].set_title("Clean heatmap")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(adv_arr)
    axes[1, 0].set_title("Adversarial image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(adv_arr)
    axes[1, 1].imshow(adv_hm, cmap="jet", alpha=0.45)
    axes[1, 1].set_title("Adversarial heatmap")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def sanitize_name(value):
    return "".join(c if (c.isalnum() or c in ["-", "_", "."]) else "_" for c in value)


def main():
    parser = argparse.ArgumentParser(description="Run PGD on ImageNet-val and save clean/adv maps for each method")
    parser.add_argument("--data-path", required=True, help="Path to ImageNet val folder")
    parser.add_argument("--index-json", default="imgnet1k_label.json", help="Path to class label json")
    parser.add_argument(
        "--methods",
        default="eclip,eclip-wo-ksim,game,maskclip,gradcam,rollout,surgery,m2ib,rise",
        help="Comma-separated explain methods",
    )
    parser.add_argument(
        "--target",
        choices=["gt", "pred"],
        default="gt",
        help="Text class used to generate clean/adv heatmaps (kept fixed for both images)",
    )
    parser.add_argument(
        "--attack-class",
        choices=["gt", "pred"],
        default="gt",
        help="Class text used by PGD objective",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on images")
    parser.add_argument("--resize-max-side", type=int, default=640, help="Resize long side guard for large images")
    parser.add_argument("--pgd-eps", type=float, default=8.0 / 255.0, help="PGD epsilon in pixel scale [0,1]")
    parser.add_argument("--pgd-alpha", type=float, default=2.0 / 255.0, help="PGD step size in pixel scale [0,1]")
    parser.add_argument("--pgd-steps", type=int, default=10, help="Number of PGD steps")
    parser.add_argument("--pgd-random-start", action="store_true", help="Enable random start in PGD")
    parser.add_argument("--output-dir", default="outputs/pgd_maps", help="Directory to save per-sample map outputs")
    parser.add_argument("--save-npy", action="store_true", help="Also save clean/adv raw heatmaps as .npy")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    if args.pgd_steps < 1:
        raise ValueError("--pgd-steps must be >= 1")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("No methods provided.")

    os.makedirs(args.output_dir, exist_ok=True)

    clipmodel, preprocess = clip.load("ViT-B/16", device=device)
    clipmodel.eval()
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

    summary_lines = [
        f"num_images_input={len(image_items)}",
        f"methods={','.join(methods)}",
        f"target={args.target}",
        f"attack_class={args.attack_class}",
        f"pgd=eps:{args.pgd_eps},alpha:{args.pgd_alpha},steps:{args.pgd_steps},random_start:{args.pgd_random_start}",
    ]
    
    spatial_transform = Compose([
        Resize(
            size=224,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        ),
        CenterCrop((224, 224)),
        _convert_image_to_rgb,
    ])

    # Tensor transforms (operate on Tensor)
    tensor_transform = Compose([
        ToTensor(),
        Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    

    for idx, (image_path, folder, image_name, gt_label) in enumerate(tqdm(image_items, desc="PGD map saving"), start=1):
        try:
            clean_img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            summary_lines.append(f"skip\t{image_path}\terror={repr(exc)}")
            continue

        clean_img = spatial_transform(clean_img)
        clean_tensor = tensor_transform(clean_img).to(device).unsqueeze(0)
        
        with torch.no_grad():
            clean_features = clipmodel.encode_image(clean_tensor)
            clean_logits = 100.0 * clean_features @ zero_shot_weights
            clean_probs = clean_logits.softmax(dim=-1)
            clean_pred_label = int(clean_probs.argmax(dim=-1).item())

        attack_label = gt_label if args.attack_class == "gt" else clean_pred_label
        explain_label = gt_label if args.target == "gt" else clean_pred_label

        attack_text_embedding = zero_shot_weights[:, attack_label].unsqueeze(0)
        explain_text_embedding = zero_shot_weights[:, explain_label].unsqueeze(0)
        explain_texts = [IMAGENET_CLASSNAMES[explain_label]]

        adv_img = pgd_minimize_similarity(
            clipmodel=clipmodel,
            image_pil=clean_img,
            target_text_embedding=attack_text_embedding,
            preprocess=preprocess,
            device=device,
            eps=args.pgd_eps,
            alpha=args.pgd_alpha,
            steps=args.pgd_steps,
            random_start=args.pgd_random_start,
        )

        adv_img = spatial_transform(adv_img)
        adv_tensor = tensor_transform(adv_img).to(device).unsqueeze(0)
        with torch.no_grad():
            adv_features = clipmodel.encode_image(adv_tensor)
            adv_logits = 100.0 * adv_features @ zero_shot_weights
            adv_probs = adv_logits.softmax(dim=-1)
            adv_pred_label = int(adv_probs.argmax(dim=-1).item())


        sample_id = f"{idx:05d}_{sanitize_name(folder)}_{sanitize_name(os.path.splitext(image_name)[0])}"
        sample_dir = os.path.join(args.output_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        clean_save_path = os.path.join(sample_dir, "clean.png")
        adv_save_path = os.path.join(sample_dir, "adv.png")
        clean_img.save(clean_save_path)
        adv_img.save(adv_save_path)

        summary_lines.append(
            "\t".join(
                [
                    "sample",
                    sample_id,
                    f"gt={gt_label}",
                    f"clean_pred={clean_pred_label}",
                    f"adv_pred={adv_pred_label}",
                    f"attack_label={attack_label}",
                    f"explain_label={explain_label}",
                ]
            )
        )

        for method in methods:
            try:
                clean_hm = explainer.generate_hm(method, clean_img, explain_text_embedding, explain_texts, resize)
                adv_hm = explainer.generate_hm(method, adv_img, explain_text_embedding, explain_texts, resize)
                clean_hm = clean_hm.detach().cpu().numpy()
                adv_hm = adv_hm.detach().cpu().numpy()

                vis_out = os.path.join(sample_dir, f"map_{sanitize_name(method)}.png")
                render_heatmap_panel(
                    clean_img=clean_img,
                    adv_img=adv_img,
                    clean_hm=clean_hm,
                    adv_hm=adv_hm,
                    method_name=method,
                    sample_title=sample_id,
                    out_path=vis_out,
                )

                if args.save_npy:
                    np.save(os.path.join(sample_dir, f"map_{sanitize_name(method)}_clean.npy"), clean_hm)
                    np.save(os.path.join(sample_dir, f"map_{sanitize_name(method)}_adv.npy"), adv_hm)
            except Exception as exc:
                summary_lines.append(f"method_fail\t{sample_id}\t{method}\terror={repr(exc)}")

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
