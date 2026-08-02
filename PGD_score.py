import argparse
import json
import os

from tqdm import tqdm

try:
    from CLIP import clip
except ImportError:
    import clip

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize
from torchvision.utils import save_image

from util import (
    DEFAULT_PROMPT_TEMPLATE,
    build_blur_substrate,
    build_causal_metric_model,
    build_zero_shot_clip_classifier,
    denorm_ImageNet1k,
    generate_hm,
    load_classnames,
    normalize_ImageNet1k,
    predict_zero_shot_clip,
    save_causal_metric_summary,
    save_saliency_outputs,
)
from RISE.evaluation import CausalMetric, auc


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "PGD similarity attack + clean deletion/insertion evaluation for CLIP zero-shot explanations"
        )
    )
    parser.add_argument("--clip-model", default="ViT-B/16", help="CLIP model name")
    parser.add_argument(
        "--clip-checkpoint",
        default=None,
        help="Optional local checkpoint path passed to clip.load instead of --clip-model",
    )
    parser.add_argument(
        "--hm-type",
        default="eclip",
        choices=["selfattn", "gradcam", "maskclip", "eclip", "eclip-wo-ksim", "game", "rollout", "surgery", "m2ib", "rise"],
        help="Explanation method passed to CLIPExplainRunner.generate_hm",
    )
    parser.add_argument(
        "--target-source",
        default="pred",
        choices=["pred", "gt"],
        help="Use predicted label or ground-truth label prompt for the saliency map",
    )
    parser.add_argument("--gt-label", type=int, default=None, help="Optional ImageNet class index for GT prompt")
    parser.add_argument("--step", type=int, default=224, help="Pixels modified per causal-metric step")
    parser.add_argument("--kernel-size", type=int, default=11, help="Gaussian blur kernel size for insertion")
    parser.add_argument("--kernel-sigma", type=int, default=5, help="Gaussian blur sigma for insertion")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--output-dir", default="test_eval_outputs", help="Where to save generated images")
    parser.add_argument("--img-dir", default="Imagenet/val", help="Directory containing input images")
    parser.add_argument("--sample-path", default="vit_b_16_1k.json", help="Path to JSON file containing image samples")
    parser.add_argument(
        "--classnames-path",
        default=None,
        help="Optional path to class names (.txt or .json). Default: ImageNet class names",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template for zero-shot text, use {} as class placeholder",
    )
    parser.add_argument("--save-process", action="store_true", help="Save every deletion/insertion step image")
    parser.add_argument("--eps", type=float, default=32.0, help="Maximum perturbation for PGD attack (in pixel values)")
    parser.add_argument("--alpha", type=float, default=8.0, help="Step size for PGD attack (in pixel values)")
    parser.add_argument("--pgd-steps", type=int, default=50, help="Number of PGD steps")
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="CausalMetric verbosity: 0 no plot, 1 final step only, 2 show every step",
    )
    return parser.parse_args()


def resolve_target_label(args, pred_label, num_classes):
    if args.target_source == "pred":
        return pred_label
    if args.gt_label is None:
        raise ValueError("--gt-label is required when --target-source gt")
    if not (0 <= args.gt_label < num_classes):
        raise ValueError("--gt-label must be a valid class index for the loaded class names")
    return int(args.gt_label)


def run_similarity_pgd(
    clip_model,
    image_raw,
    text_embedding,
    eps,
    alpha,
    pgd_steps,
):
    delta = torch.zeros_like(image_raw, requires_grad=True)

    trace = []
    with torch.no_grad():
        clean_norm = normalize_ImageNet1k(image_raw)
        clean_img_embed = F.normalize(clip_model.encode_image(clean_norm), dim=-1)
        clean_similarity = float((clean_img_embed * text_embedding).sum(dim=-1).item())

    for step_idx in range(pgd_steps):
        x_adv_raw = torch.clamp(image_raw + delta, 0.0, 1.0)
        x_adv_norm = normalize_ImageNet1k(x_adv_raw)

        img_embed = F.normalize(clip_model.encode_image(x_adv_norm), dim=-1)
        similarity = (img_embed * text_embedding).sum(dim=-1).mean()
        loss = similarity

        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()

        with torch.no_grad():
            # Minimize similarity: gradient descent on similarity objective.
            delta -= alpha * delta.grad.sign()
            delta.clamp_(-eps, eps)

        delta = delta.detach().requires_grad_(True)

        trace.append(
            {
                "pgd_step": step_idx + 1,
                "similarity": float(similarity.item()),
                "loss": float(loss.item()),
            }
        )

    with torch.no_grad():
        x_adv_raw = torch.clamp(image_raw + delta, 0.0, 1.0)
        x_adv_norm = normalize_ImageNet1k(x_adv_raw)
        adv_img_embed = F.normalize(clip_model.encode_image(x_adv_norm), dim=-1)
        adv_similarity = float((adv_img_embed * text_embedding).sum(dim=-1).item())

    details = {
        "clean_similarity": clean_similarity,
        "adv_similarity": adv_similarity,
        "similarity_trace": trace,
    }
    return x_adv_raw, details


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    classnames = load_classnames(args.classnames_path)
    blur_fn = build_blur_substrate(args.kernel_size, args.kernel_sigma)

    clip_source = args.clip_checkpoint or args.clip_model
    if args.clip_checkpoint and not os.path.isfile(args.clip_checkpoint):
        raise FileNotFoundError(f"--clip-checkpoint not found: {args.clip_checkpoint}")

    clip_model, preprocess = clip.load(clip_source, device=device)
    clip_model.eval()

    classifier, _ = build_zero_shot_clip_classifier(
        clip_model,
        device=device,
        classnames=classnames,
        prompt_template=args.prompt_template,
        num_classes_per_batch=10,
        use_tqdm=True,
    )
    metric_model = build_causal_metric_model(classifier)
    output_dir = os.path.join(args.output_dir, args.hm_type, "pgd")
    os.makedirs(output_dir, exist_ok=True)

    with open(args.sample_path, "r", encoding="utf-8") as f:
        sample_list = json.load(f)

    for folder_name, image_name in tqdm(sample_list.items()):
        sample_dir = os.path.join(output_dir, folder_name)
        os.makedirs(sample_dir, exist_ok=True)

        img_path = os.path.join(args.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")

        input_resolution = clip_model.visual.input_resolution
        resized_image = image.resize((input_resolution, input_resolution), Image.BICUBIC)
        image_normalized = preprocess(resized_image).unsqueeze(0)
        image_raw = denorm_ImageNet1k(image_normalized).to(device)
        metric_resize = Resize(tuple(image_normalized.shape[-2:]))

        _, _, pred_label, pred_confidence = predict_zero_shot_clip(classifier, image_normalized, device)

        target_label = resolve_target_label(args, pred_label, len(classnames))
        target_texts = [classnames[target_label]]

        with torch.no_grad():
            text_tokens = clip.tokenize(target_texts).to(device)
            text_embedding = clip_model.encode_text(text_tokens)
            text_embedding = F.normalize(text_embedding, dim=-1)

        x_adv_raw, pgd_details = run_similarity_pgd(
            clip_model=clip_model,
            image_raw=image_raw,
            text_embedding=text_embedding,
            eps=args.eps / 255.0,
            alpha=args.alpha / 255.0,
            pgd_steps=args.pgd_steps,
        )

        x_adv = x_adv_raw.detach().cpu()
        save_image(x_adv, os.path.join(sample_dir, "adversarial_image_pgd.png"))
        x_adv_normalized_cpu = normalize_ImageNet1k(x_adv)
        x_adv_normalized = x_adv_normalized_cpu.to(device)

        _, _, adv_pred_label, adv_pred_confidence = predict_zero_shot_clip(classifier, x_adv_normalized, device)
        metric_class_name = classnames[adv_pred_label]
        adv_target_texts = [metric_class_name]

        with torch.no_grad():
            adv_text_tokens = clip.tokenize(adv_target_texts).to(device)
            adv_text_embedding = clip_model.encode_text(adv_text_tokens)
            adv_text_embedding = F.normalize(adv_text_embedding, dim=-1)

        print(
            f"[{folder_name}] pred_class(clean)={classnames[pred_label]} ({pred_confidence:.4f}) | "
            f"pred_class(adv)={classnames[adv_pred_label]} ({adv_pred_confidence:.4f})"
        )

        heatmap = generate_hm(
            clip_model,
            args.hm_type,
            x_adv_normalized,
            adv_text_embedding,
            adv_target_texts,
            metric_resize,
            preprocess,
        )

        rerun_results = {}
        for rerun_mode in ["del", "ins"]:
            rerun_step_function = (lambda x: torch.zeros_like(x)) if rerun_mode == "del" else blur_fn
            clean_metric = CausalMetric(metric_model, rerun_mode, args.step, rerun_step_function)

            rerun_process_dir = os.path.join(sample_dir, f"steps_{rerun_mode}")
            if args.save_process:
                os.makedirs(rerun_process_dir, exist_ok=True)

            save_saliency_outputs(
                heatmap.detach().cpu().numpy(),
                resized_image,
                sample_dir,
                stem=f"{rerun_mode}_pgd_{args.hm_type}_saliency",
            )

            curve = clean_metric.single_run(
                x_adv_normalized_cpu,
                heatmap.detach().cpu().numpy(),
                verbose=args.verbose,
                save_to=rerun_process_dir if args.save_process else None,
            )

            save_causal_metric_summary(
                image_tensor=x_adv_normalized_cpu,
                final_tensor=torch.zeros_like(x_adv_normalized_cpu) if rerun_mode == "del" else x_adv_normalized_cpu,
                scores=curve,
                output_path=os.path.join(sample_dir, f"{rerun_mode}_summary.png"),
                mode=rerun_mode,
                class_name=metric_class_name,
                preprocess=preprocess,
            )

            rerun_results[rerun_mode] = {
                "curve": curve.tolist(),
                "auc": float(auc(curve)),
            }

        curve_information = {
            "attack_type": "pgd_similarity",
            "target_source": args.target_source,
            "target_label": int(target_label),
            "target_classname": classnames[target_label],
            "adv_target_label": int(adv_pred_label),
            "adv_target_classname": metric_class_name,
            "pred_label_clean": int(pred_label),
            "pred_classname_clean": classnames[pred_label],
            "pred_confidence_clean": float(pred_confidence),
            "pred_label_adv": int(adv_pred_label),
            "pred_classname_adv": classnames[adv_pred_label],
            "pred_confidence_adv": float(adv_pred_confidence),
            "clean_similarity": pgd_details["clean_similarity"],
            "adv_similarity": pgd_details["adv_similarity"],
            "similarity_trace": pgd_details["similarity_trace"],
            "rerun": rerun_results,
        }

        with open(os.path.join(sample_dir, "curve_information.json"), "w", encoding="utf-8") as f:
            json.dump(curve_information, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
