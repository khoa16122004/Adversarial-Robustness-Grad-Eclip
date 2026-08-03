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
            "X-Shift attack (prediction-preserving, saliency-shifting) + rerun deletion/insertion evaluation"
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

    parser.add_argument("--eps", type=float, default=32.0, help="Maximum perturbation for X-Shift attack (pixel values)")
    parser.add_argument("--alpha", type=float, default=4.0, help="Step size for X-Shift attack (pixel values)")
    parser.add_argument(
        "--xshift-steps",
        "--pgd-steps",
        dest="xshift_steps",
        type=int,
        default=50,
        help="Number of X-Shift iterations (alias: --pgd-steps)",
    )
    parser.add_argument("--xai-topk-ratio", type=float, default=0.1, help="Top-K ratio used by L_xai over image patches")
    parser.add_argument("--xai-alpha", type=float, default=0.5, help="Weight for non-top patches in L_xai")
    parser.add_argument("--lambda-pred", type=float, default=1.0, help="Weight for prediction-preservation loss")
    parser.add_argument("--lambda-patch", type=float, default=0.3, help="Weight for patch margin loss")
    parser.add_argument("--lambda-ent", type=float, default=0.05, help="Weight for entropy concentration loss")
    parser.add_argument("--margin", type=float, default=0.05, help="Margin for patch target-dominance loss")
    parser.add_argument(
        "--l0-k",
        type=int,
        default=5000,
        help="Max number of perturbed entries in delta after projection (set <=0 to disable)",
    )

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


def encode_vit_tokens(clip_model, image_normalized):
    visual = clip_model.visual
    if not hasattr(visual, "class_embedding") or not hasattr(visual, "transformer"):
        raise ValueError("X-Shift currently supports ViT-based CLIP visual encoders only.")

    x = image_normalized.type(clip_model.dtype)
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)

    class_token = visual.class_embedding.to(x.dtype)
    class_token = class_token + torch.zeros(
        x.shape[0],
        1,
        x.shape[-1],
        dtype=x.dtype,
        device=x.device,
    )
    x = torch.cat([class_token, x], dim=1)

    pos_embed = visual.positional_embedding.to(x.dtype)
    if x.shape[1] != pos_embed.shape[0]:
        raise ValueError(
            "Input token count does not match CLIP positional embedding. "
            "Use model default input resolution for X-Shift."
        )

    x = x + pos_embed
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)

    x = visual.ln_post(x)
    if getattr(visual, "proj", None) is not None:
        x = x @ visual.proj

    cls_feature = x[:, 0, :]
    patch_features = x[:, 1:, :]
    return cls_feature, patch_features


def topk_l0_project(delta, k):
    if k is None or k <= 0:
        return delta

    flat = delta.view(delta.shape[0], -1)
    n = flat.shape[1]
    k = min(k, n)
    if k >= n:
        return delta

    values = flat.abs()
    topk_idx = values.topk(k=k, dim=1, largest=True, sorted=False).indices

    mask = torch.zeros_like(flat)
    mask.scatter_(1, topk_idx, 1.0)
    return (flat * mask).view_as(delta)


def run_xshift_attack(
    clip_model,
    classifier,
    image_raw,
    original_label,
    target_label,
    all_text_features,
    eps,
    alpha,
    xshift_steps,
    xai_topk_ratio,
    xai_alpha,
    lambda_pred,
    lambda_patch,
    lambda_ent,
    margin,
    l0_k,
):
    device = image_raw.device
    delta = torch.zeros_like(image_raw, requires_grad=True)

    target_text_feature = all_text_features[:, target_label]

    trace = []
    with torch.no_grad():
        clean_norm = normalize_ImageNet1k(image_raw)
        _, clean_patches = encode_vit_tokens(clip_model, clean_norm)
        clean_patches = F.normalize(clean_patches.float(), dim=-1)
        clean_similarity = torch.einsum("bpd,d->bp", clean_patches, target_text_feature.float()).mean().item()

    for step_idx in range(xshift_steps):
        x_adv_raw = torch.clamp(image_raw + delta, 0.0, 1.0)
        x_adv_norm = normalize_ImageNet1k(x_adv_raw)

        cls_feature, patch_features = encode_vit_tokens(clip_model, x_adv_norm)
        cls_feature = F.normalize(cls_feature.float(), dim=-1)
        patch_features = F.normalize(patch_features.float(), dim=-1)

        similarity_target = torch.einsum("bpd,d->bp", patch_features, target_text_feature.float())
        num_patches = similarity_target.shape[1]
        k_top = int(round(float(num_patches) * float(xai_topk_ratio)))
        k_top = max(1, min(num_patches, k_top))

        topk_indices = similarity_target.topk(k_top, dim=1, largest=True).indices
        mask = torch.zeros_like(similarity_target)
        mask.scatter_(1, topk_indices, 1.0)

        loss_top = -((similarity_target * mask).sum(dim=1) / float(k_top)).mean()
        if num_patches > k_top:
            loss_other = ((similarity_target * (1.0 - mask)).sum(dim=1) / float(num_patches - k_top)).mean()
        else:
            loss_other = torch.zeros(1, device=device, dtype=similarity_target.dtype).squeeze(0)
        loss_xai = loss_top + xai_alpha * loss_other

        patch_sim = torch.einsum("bpd,dc->bpc", patch_features, all_text_features.float())
        target_score = patch_sim[:, :, target_label]
        other_score = patch_sim.clone()
        other_score[:, :, target_label] = -1e9
        max_other = other_score.max(dim=-1).values
        loss_patch = F.relu(max_other - target_score + margin).mean()

        prob_patches = F.softmax(similarity_target, dim=1)
        loss_entropy = (prob_patches * torch.log(prob_patches + 1e-8)).sum(dim=1).mean()

        logits = classifier(x_adv_norm)
        label_tensor = torch.tensor([original_label], device=device, dtype=torch.long)
        loss_pred = F.cross_entropy(logits, label_tensor)

        loss = loss_xai + lambda_pred * loss_pred + lambda_patch * loss_patch + lambda_ent * loss_entropy

        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()

        with torch.no_grad():
            delta += alpha * delta.grad.sign()
            delta.clamp_(-eps, eps)
            delta.copy_(topk_l0_project(delta, l0_k))
            x_adv_raw = torch.clamp(image_raw + delta, 0.0, 1.0)
            delta.copy_(x_adv_raw - image_raw)

        delta = delta.detach().requires_grad_(True)

        trace.append(
            {
                "xshift_step": step_idx + 1,
                "loss": float(loss.item()),
                "loss_xai": float(loss_xai.item()),
                "loss_pred": float(loss_pred.item()),
                "loss_patch": float(loss_patch.item()),
                "loss_entropy": float(loss_entropy.item()),
                "similarity_target_mean": float(similarity_target.mean().item()),
            }
        )

    with torch.no_grad():
        x_adv_raw = torch.clamp(image_raw + delta, 0.0, 1.0)
        x_adv_norm = normalize_ImageNet1k(x_adv_raw)
        _, adv_patches = encode_vit_tokens(clip_model, x_adv_norm)
        adv_patches = F.normalize(adv_patches.float(), dim=-1)
        adv_similarity = torch.einsum("bpd,d->bp", adv_patches, target_text_feature.float()).mean().item()

    details = {
        "clean_similarity": float(clean_similarity),
        "adv_similarity": float(adv_similarity),
        "trace": trace,
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
    output_dir = os.path.join(args.output_dir, args.hm_type, "xshift")
    os.makedirs(output_dir, exist_ok=True)

    all_text_features = classifier.zero_shot_weights.detach().to(device)
    all_text_features = F.normalize(all_text_features, dim=0)

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

        x_adv_raw, xshift_details = run_xshift_attack(
            clip_model=clip_model,
            classifier=classifier,
            image_raw=image_raw,
            original_label=pred_label,
            target_label=target_label,
            all_text_features=all_text_features,
            eps=args.eps / 255.0,
            alpha=args.alpha / 255.0,
            xshift_steps=args.xshift_steps,
            xai_topk_ratio=args.xai_topk_ratio,
            xai_alpha=args.xai_alpha,
            lambda_pred=args.lambda_pred,
            lambda_patch=args.lambda_patch,
            lambda_ent=args.lambda_ent,
            margin=args.margin,
            l0_k=args.l0_k,
        )

        x_adv = x_adv_raw.detach().cpu()
        save_image(x_adv, os.path.join(sample_dir, "adversarial_image_xshift.png"))
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
                stem=f"{rerun_mode}_xshift_{args.hm_type}_saliency",
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
            "attack_type": "xshift",
            "target_source": args.target_source,
            "target_label": int(target_label),
            "target_classname": classnames[target_label],
            "pred_label_clean": int(pred_label),
            "pred_classname_clean": classnames[pred_label],
            "pred_confidence_clean": float(pred_confidence),
            "pred_label_adv": int(adv_pred_label),
            "pred_classname_adv": classnames[adv_pred_label],
            "pred_confidence_adv": float(adv_pred_confidence),
            "clean_similarity": xshift_details["clean_similarity"],
            "adv_similarity": xshift_details["adv_similarity"],
            "xshift_trace": xshift_details["trace"],
            "config": {
                "eps": args.eps,
                "alpha": args.alpha,
                "xshift_steps": args.xshift_steps,
                "xai_topk_ratio": args.xai_topk_ratio,
                "xai_alpha": args.xai_alpha,
                "lambda_pred": args.lambda_pred,
                "lambda_patch": args.lambda_patch,
                "lambda_ent": args.lambda_ent,
                "margin": args.margin,
                "l0_k": args.l0_k,
            },
            "rerun": rerun_results,
        }

        with open(os.path.join(sample_dir, "curve_information.json"), "w", encoding="utf-8") as f:
            json.dump(curve_information, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
