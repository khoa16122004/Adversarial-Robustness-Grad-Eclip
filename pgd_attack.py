import argparse
import json
import os

import clip
import matplotlib.cm as cm
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm

from clip_utils import build_zero_shot_classifier
from generate_emap import CLIPExplainRunner
from imagenet_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES


def clip_normalization_stats(device):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    return mean, std


def resolve_image_path(raw_path, input_image_dir):
    raw_path = str(raw_path).replace("\\", "/")

    if os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path, None

    candidates = []

    if input_image_dir is not None:
        candidates.append(os.path.join(input_image_dir, raw_path))

        marker = "/val/"
        if marker in raw_path:
            rel_after_val = raw_path.split(marker, 1)[1]
            candidates.append(os.path.join(input_image_dir, rel_after_val))

        parts = [p for p in raw_path.split("/") if p]
        if len(parts) >= 2:
            candidates.append(os.path.join(input_image_dir, parts[-2], parts[-1]))

    for cand in candidates:
        if os.path.exists(cand):
            return cand, os.path.relpath(cand, input_image_dir).replace("\\", "/") if input_image_dir else None

    return None, None


def parse_samples(sample_json_path, input_image_dir, max_images=None):
    with open(sample_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []

    if isinstance(data, dict) and "correct_samples" in data:
        for item in data.get("correct_samples", []):
            rel_path = str(item.get("image_path", ""))
            if not rel_path:
                continue
            abs_path, inferred_rel = resolve_image_path(rel_path, input_image_dir)
            if abs_path is None:
                continue
            final_rel = rel_path if not os.path.isabs(rel_path) else (inferred_rel or rel_path)
            samples.append({"source_path": abs_path, "rel_path": final_rel})
            if max_images is not None and len(samples) >= max_images:
                break

    elif isinstance(data, dict):
        # Selection format: {"n01440764": "n01440764/xxx.JPEG" or absolute path}
        for wnid in sorted(data.keys()):
            raw_path = str(data[wnid])
            abs_path, inferred_rel = resolve_image_path(raw_path, input_image_dir)
            if abs_path is None:
                continue
            if inferred_rel:
                rel_path = inferred_rel
            else:
                file_name = os.path.basename(raw_path.replace("\\", "/"))
                rel_path = f"{wnid}/{file_name}"
            samples.append({"source_path": abs_path, "rel_path": rel_path})
            if max_images is not None and len(samples) >= max_images:
                break

    elif isinstance(data, list):
        for item in data:
            rel_path = str(item.get("image_path", "")) if isinstance(item, dict) else ""
            if not rel_path:
                continue
            abs_path, inferred_rel = resolve_image_path(rel_path, input_image_dir)
            if abs_path is None:
                continue
            final_rel = rel_path if not os.path.isabs(rel_path) else (inferred_rel or rel_path)
            samples.append({"source_path": abs_path, "rel_path": final_rel})
            if max_images is not None and len(samples) >= max_images:
                break

    else:
        raise ValueError("Unsupported JSON format for samples.")

    return samples


def heatmap_to_color_image(heatmap_tensor):
    if isinstance(heatmap_tensor, torch.Tensor):
        heatmap = heatmap_tensor.detach().cpu().numpy().astype(np.float32)
    else:
        heatmap = np.asarray(heatmap_tensor, dtype=np.float32)
    heatmap -= heatmap.min()
    denom = heatmap.max()
    if denom > 0:
        heatmap /= denom
    color = cm.get_cmap("jet")(heatmap)[..., :3]
    color = (color * 255.0).astype(np.uint8)
    return Image.fromarray(color)


def normalize_01(array_like):
    arr = np.asarray(array_like, dtype=np.float32)
    arr -= arr.min()
    denom = arr.max()
    if denom > 0:
        arr /= denom
    return arr


def pgd_minimize_similarity(
    clip_model,
    zero_shot_weights,
    x0,
    initial_pred_label,
    target_text_embedding,
    device,
    eps=8.0 / 255.0,
    alpha=2.0 / 255.0,
    steps=10,
    random_start=False,
):
    mean, std = clip_normalization_stats(device)

    eps_tensor = torch.tensor([eps, eps, eps], device=device).view(1, 3, 1, 1) / std
    alpha_tensor = torch.tensor([alpha, alpha, alpha], device=device).view(1, 3, 1, 1) / std

    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std

    x_adv = x0.clone().detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-1.0, 1.0) * eps_tensor
        x_adv = torch.max(torch.min(x_adv, x0 + eps_tensor), x0 - eps_tensor)
        x_adv = torch.max(torch.min(x_adv, upper), lower)

    text_feat = target_text_embedding.detach()
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    sim_history = []
    init_pred_prob_history = []

    with torch.no_grad():
        init_image_feat = clip_model.encode_image(x_adv)
        init_image_feat = init_image_feat / init_image_feat.norm(dim=-1, keepdim=True)
        init_sim = float((init_image_feat @ text_feat.T).item())
        init_logits = 100.0 * init_image_feat @ zero_shot_weights
        init_probs = init_logits.softmax(dim=-1)
        init_pred_prob = float(init_probs[0, initial_pred_label].item())
        sim_history.append(init_sim)
        init_pred_prob_history.append(init_pred_prob)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        image_feat = clip_model.encode_image(x_adv)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        cosine = (image_feat @ text_feat.T).mean()

        grad = torch.autograd.grad(cosine, x_adv, retain_graph=False, create_graph=False)[0]

        # Minimize similarity to the predicted-class text.
        x_adv = x_adv - alpha_tensor * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps_tensor), x0 - eps_tensor)
        x_adv = torch.max(torch.min(x_adv, upper), lower)
        x_adv = x_adv.detach()

        with torch.no_grad():
            step_feat = clip_model.encode_image(x_adv)
            step_feat = step_feat / step_feat.norm(dim=-1, keepdim=True)
            step_sim = float((step_feat @ text_feat.T).item())
            step_logits = 100.0 * step_feat @ zero_shot_weights
            step_probs = step_logits.softmax(dim=-1)
            step_init_pred_prob = float(step_probs[0, initial_pred_label].item())
            sim_history.append(step_sim)
            init_pred_prob_history.append(step_init_pred_prob)

    return x_adv, sim_history, init_pred_prob_history


def main():
    parser = argparse.ArgumentParser(
        description="PGD attack on CLIP after spatial transform, minimizing similarity to initial predicted-class text"
    )
    parser.add_argument("--samples-json", required=True, help="Input samples JSON path")
    parser.add_argument("--input-image-dir", required=True, help="Image root directory (val dir)")
    parser.add_argument("--output-dir", default="outputs/pgd_adv_images", help="Directory to save attacked images")
    parser.add_argument("--clip-model", default="ViT-B/16", help="CLIP model name for clip.load")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of samples")
    parser.add_argument("--pgd-eps", type=float, default=8.0 / 255.0, help="L-inf epsilon in [0,1]")
    parser.add_argument("--pgd-alpha", type=float, default=2.0 / 255.0, help="PGD step size in [0,1]")
    parser.add_argument("--pgd-steps", type=int, default=10, help="PGD iterations")
    parser.add_argument("--pgd-random-start", action="store_true", help="Enable random start")
    parser.add_argument("--explain-method", default="eclip", help="Explain method name used for clean/adv maps")
    parser.add_argument("--save-ext", default=".png", help="Output extension (.png or .jpg)")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    if args.pgd_steps < 1:
        raise ValueError("--pgd-steps must be >= 1")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    samples = parse_samples(args.samples_json, args.input_image_dir, max_images=args.max_images)
    if not samples:
        raise ValueError("No valid samples resolved from JSON and input image dir.")

    os.makedirs(args.output_dir, exist_ok=True)

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

    mean, std = clip_normalization_stats(device)

    summary = {
        "model": f"CLIP-{args.clip_model}",
        "samples_json": args.samples_json,
        "input_image_dir": args.input_image_dir,
        "output_dir": args.output_dir,
        "max_images": args.max_images,
        "pgd": {
            "eps": args.pgd_eps,
            "alpha": args.pgd_alpha,
            "steps": args.pgd_steps,
            "random_start": args.pgd_random_start,
        },
        "processed": [],
    }

    for sample in tqdm(samples, desc="PGD attacking"):
        source_path = sample["source_path"]
        rel_path = sample["rel_path"].replace("\\", "/")

        try:
            image = Image.open(source_path).convert("RGB")
        except Exception:
            continue

        # x0 is already after CLIP spatial transform (resize/crop/normalize).
        x0 = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            clean_feat = clip_model.encode_image(x0)
            clean_logits = 100.0 * clean_feat @ zero_shot_weights
            clean_probs = clean_logits.softmax(dim=-1)
            pred_label = int(clean_probs.argmax(dim=-1).item())
            clean_sim = float(clean_probs[0, pred_label].item())

        pred_text_embedding = zero_shot_weights[:, pred_label].unsqueeze(0)
        x_adv, sim_history, init_pred_prob_history = pgd_minimize_similarity(
            clip_model=clip_model,
            zero_shot_weights=zero_shot_weights,
            x0=x0,
            initial_pred_label=pred_label,
            target_text_embedding=pred_text_embedding,
            device=device,
            eps=args.pgd_eps,
            alpha=args.pgd_alpha,
            steps=args.pgd_steps,
            random_start=args.pgd_random_start,
        )

        with torch.no_grad():
            adv_feat = clip_model.encode_image(x_adv)
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            adv_logits = 100.0 * adv_feat @ zero_shot_weights
            adv_probs = adv_logits.softmax(dim=-1)
            adv_pred_label = int(adv_probs.argmax(dim=-1).item())
            text_feat = pred_text_embedding / pred_text_embedding.norm(dim=-1, keepdim=True)
            adv_sim = float((adv_feat @ text_feat.T).item())

        clean_pixel = (x0 * std + mean).clamp(0.0, 1.0)
        adv_pixel = (x_adv * std + mean).clamp(0.0, 1.0)
        clean_pil = TF.to_pil_image(clean_pixel.squeeze(0).cpu())
        adv_pil = TF.to_pil_image(adv_pixel.squeeze(0).cpu())

        base_rel, _ = os.path.splitext(rel_path)
        sample_dir = os.path.join(args.output_dir, base_rel)
        map_dir = os.path.join(sample_dir, "map")
        os.makedirs(map_dir, exist_ok=True)

        clean_out_rel = f"{base_rel}/clean_image{args.save_ext}".replace("\\", "/")
        adv_out_rel = f"{base_rel}/adv_image{args.save_ext}".replace("\\", "/")
        clean_out_path = os.path.join(args.output_dir, clean_out_rel)
        adv_out_path = os.path.join(args.output_dir, adv_out_rel)

        clean_pil.save(clean_out_path)
        adv_pil.save(adv_out_path)

        map_resize = Resize((clean_pil.height, clean_pil.width))
        txt_embedding = zero_shot_weights[:, pred_label].unsqueeze(0)
        txts = [IMAGENET_CLASSNAMES[pred_label]]

        clean_hm = explainer.generate_hm(args.explain_method, clean_pil, txt_embedding, txts, map_resize)
        adv_hm = explainer.generate_hm(args.explain_method, adv_pil, txt_embedding, txts, map_resize)

        clean_hm_np = normalize_01(clean_hm.detach().cpu().numpy())
        adv_hm_np = normalize_01(adv_hm.detach().cpu().numpy())

        clean_map_img = heatmap_to_color_image(clean_hm_np)
        adv_map_img = heatmap_to_color_image(adv_hm_np)

        clean_map_path = os.path.join(map_dir, f"clean_{args.explain_method}.png")
        adv_map_path = os.path.join(map_dir, f"adv_{args.explain_method}.png")
        clean_map_img.save(clean_map_path)
        adv_map_img.save(adv_map_path)

        clean_tensor_np = clean_pixel.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        adv_tensor_np = adv_pixel.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        clean_tensor_np = np.clip(clean_tensor_np, 0.0, 1.0)
        adv_tensor_np = np.clip(adv_tensor_np, 0.0, 1.0)

        clean_tensor_path = os.path.join(sample_dir, "clean_tensor_01.npy")
        adv_tensor_path = os.path.join(sample_dir, "adv_tensor_01.npy")
        clean_map_npy_path = os.path.join(map_dir, f"clean_{args.explain_method}.npy")
        adv_map_npy_path = os.path.join(map_dir, f"adv_{args.explain_method}.npy")

        np.save(clean_tensor_path, clean_tensor_np)
        np.save(adv_tensor_path, adv_tensor_np)
        np.save(clean_map_npy_path, clean_hm_np)
        np.save(adv_map_npy_path, adv_hm_np)

        sample_metadata = {
            "source_path": source_path,
            "relative_path": rel_path,
            "clean_image_path": clean_out_rel,
            "adv_image_path": adv_out_rel,
            "explain_method": args.explain_method,
            "clean_map_path": f"{base_rel}/map/clean_{args.explain_method}.png".replace("\\", "/"),
            "adv_map_path": f"{base_rel}/map/adv_{args.explain_method}.png".replace("\\", "/"),
            "clean_map_npy_path": f"{base_rel}/map/clean_{args.explain_method}.npy".replace("\\", "/"),
            "adv_map_npy_path": f"{base_rel}/map/adv_{args.explain_method}.npy".replace("\\", "/"),
            "clean_tensor_01_path": f"{base_rel}/clean_tensor_01.npy".replace("\\", "/"),
            "adv_tensor_01_path": f"{base_rel}/adv_tensor_01.npy".replace("\\", "/"),
            "from_pred_label": pred_label,
            "from_pred_name": IMAGENET_CLASSNAMES[pred_label],
            "to_pred_label": adv_pred_label,
            "to_pred_name": IMAGENET_CLASSNAMES[adv_pred_label],
            "class_transition": f"{pred_label}->{adv_pred_label}",
            "clean_pred_prob": clean_sim,
            "adv_text_cosine": adv_sim,
            "sim_history_to_initial_text": sim_history,
            "init_pred_prob_history": init_pred_prob_history,
        }

        metadata_path = os.path.join(sample_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(sample_metadata, f, ensure_ascii=False, indent=2)

        summary["processed"].append(sample_metadata)

    summary_path = os.path.join(args.output_dir, "attack_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved attacked images to: {args.output_dir}")
    print(f"Saved summary: {summary_path}")
    print(f"Processed samples: {len(summary['processed'])}")


if __name__ == "__main__":
    main()
