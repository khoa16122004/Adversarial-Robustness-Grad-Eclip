"""Prediction-preserving X-Shift attack and causal reruns for SigLIP."""

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize
from torchvision.utils import save_image
from tqdm import tqdm

from RISE.evaluation import CausalMetric, auc
from siglip_attack_adapter import (
    HM_METHODS, SigLIPAttackAdapter, build_zero_shot_siglip_classifier,
)
from test_siglip_explainers import EXPLAINER_VERSION
from util import (
    build_blur_substrate, build_causal_metric_model, denorm_ImageNet1k,
    load_classnames, normalize_ImageNet1k, predict_zero_shot_clip,
    save_causal_metric_summary, save_saliency_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--sample-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("SIGLIP_result_xshift"))
    parser.add_argument("--classnames-path", type=Path)
    parser.add_argument("--prompt-template", default="a photo of {}")
    parser.add_argument("--hm-types", nargs="+", choices=tuple(HM_METHODS), default=["eclip", "game", "gradcam", "maskclip", "rise"])
    parser.add_argument("--target-source", choices=("pred", "gt"), default="pred")
    parser.add_argument("--gt-label", type=int)
    parser.add_argument("--step", type=int, default=224)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=float, default=5)
    parser.add_argument("--eps", type=float, default=16)
    parser.add_argument("--alpha", type=float, default=4)
    parser.add_argument("--xshift-steps", type=int, default=50)
    parser.add_argument("--xai-topk-ratio", type=float, default=0.1)
    parser.add_argument("--xai-alpha", type=float, default=0.5)
    parser.add_argument("--lambda-pred", type=float, default=1.0)
    parser.add_argument("--lambda-patch", type=float, default=0.3)
    parser.add_argument("--lambda-ent", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--game-start-layer", type=int, default=-1)
    parser.add_argument("--rise-masks", type=int, default=2000)
    parser.add_argument("--rise-grid", type=int, default=8)
    parser.add_argument("--rise-prob", type=float, default=0.5)
    parser.add_argument("--rise-batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-process", action="store_true")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if not args.img_dir.is_dir():
        parser.error(f"--img-dir: directory not found: {args.img_dir}")
    if not args.sample_path.is_file():
        parser.error(f"--sample-path: file not found: {args.sample_path}")
    if args.target_source == "gt" and args.gt_label is None:
        parser.error("--gt-label is required with --target-source gt")
    if min(args.step, args.xshift_steps, args.rise_masks, args.rise_grid, args.rise_batch_size) <= 0:
        parser.error("--step, --xshift-steps, --rise-masks, --rise-grid and --rise-batch-size must be positive")
    if args.eps < 0 or args.alpha < 0 or args.seed < 0:
        parser.error("--eps, --alpha and --seed must be nonnegative")
    if not 0 < args.xai_topk_ratio <= 1 or not 0 < args.rise_prob < 1:
        parser.error("--xai-topk-ratio must be in (0, 1] and --rise-prob in (0, 1)")
    args.hm_types = list(dict.fromkeys(args.hm_types))
    return args


def encode_siglip_tokens(adapter, image_normalized):
    """Return normalized projected SigLIP patch features in the shared image space."""
    model = adapter.model
    pixels = adapter.to_siglip_pixels(image_normalized)
    vision = model.vision_model
    hidden = vision.embeddings(pixels)
    for layer in vision.encoder.layers:
        hidden = layer(hidden, attention_mask=None)
    hidden = vision.post_layernorm(hidden)
    projection = getattr(model, "visual_projection", None)
    if projection is not None:
        hidden = projection(hidden)
    return hidden


def run_xshift_attack(
    adapter, classifier, image_raw, original_label, target_label, all_text_features,
    eps, alpha, xshift_steps, xai_topk_ratio=0.1, xai_alpha=0.5,
    lambda_pred=1.0, lambda_patch=0.3, lambda_ent=0.05, margin=0.05,
    token_encoder=encode_siglip_tokens,
):
    device = image_raw.device
    text_features = F.normalize(all_text_features.to(device).float(), dim=0)
    target_text = text_features[:, target_label]
    with torch.no_grad():
        clean_normalized = normalize_ImageNet1k(image_raw)
        clean_patches = F.normalize(token_encoder(adapter, clean_normalized).float(), dim=-1)
        clean_patch_scores = torch.einsum("bpd,d->bp", clean_patches, target_text)
        clean_similarity = float(clean_patch_scores.mean().item())
        clean_pred_label = int(classifier(clean_normalized).argmax(dim=-1).item())

    delta = torch.zeros_like(image_raw, requires_grad=True)
    selected_image = image_raw.detach().clone()
    selected_step = 0
    selected_change = 0.0
    trace = []
    for step_index in range(xshift_steps):
        adv_raw = (image_raw + delta).clamp(0, 1)
        adv_normalized = normalize_ImageNet1k(adv_raw)
        patch_features = F.normalize(token_encoder(adapter, adv_normalized).float(), dim=-1)
        patch_scores = torch.einsum("bpd,d->bp", patch_features, target_text)
        patch_count = patch_scores.shape[1]
        top_count = max(1, min(patch_count, round(patch_count * xai_topk_ratio)))
        top_indices = patch_scores.topk(top_count, dim=1).indices
        top_mask = torch.zeros_like(patch_scores).scatter_(1, top_indices, 1.0)
        loss_top = -((patch_scores * top_mask).sum(1) / top_count).mean()
        if patch_count > top_count:
            loss_other = ((patch_scores * (1 - top_mask)).sum(1) / (patch_count - top_count)).mean()
        else:
            loss_other = patch_scores.new_zeros(())
        loss_xai = loss_top + xai_alpha * loss_other

        all_patch_scores = patch_features @ text_features
        target_scores = all_patch_scores[:, :, target_label]
        other_scores = all_patch_scores.clone()
        other_scores[:, :, target_label] = -torch.inf
        loss_patch = F.relu(other_scores.max(dim=-1).values - target_scores + margin).mean()
        patch_probabilities = F.softmax(patch_scores, dim=1)
        loss_entropy = (patch_probabilities * torch.log(patch_probabilities + 1e-8)).sum(1).mean()
        logits = classifier(adv_normalized)
        label = torch.tensor([original_label], device=device)
        loss_pred = F.cross_entropy(logits, label)
        loss = loss_xai + lambda_pred * loss_pred + lambda_patch * loss_patch + lambda_ent * loss_entropy

        if delta.grad is not None:
            delta.grad = None
        loss.backward()
        with torch.no_grad():
            delta -= alpha * delta.grad.sign()
            delta.clamp_(-eps, eps)
            adv_raw = (image_raw + delta).clamp(0, 1)
            delta.copy_(adv_raw - image_raw)
            adv_normalized = normalize_ImageNet1k(adv_raw)
            pred_label = int(classifier(adv_normalized).argmax(dim=-1).item())
            patch_eval = F.normalize(token_encoder(adapter, adv_normalized).float(), dim=-1)
            scores_eval = torch.einsum("bpd,d->bp", patch_eval, target_text)
            map_change = float((scores_eval - clean_patch_scores).abs().mean().item())
            preserved = pred_label == int(original_label)
            if preserved and (
                map_change > selected_change
                or (map_change == selected_change and step_index + 1 > selected_step)
            ):
                selected_image = adv_raw.detach().clone()
                selected_step = step_index + 1
                selected_change = map_change
        trace.append({
            "xshift_step": step_index + 1,
            "loss": float(loss.item()),
            "loss_xai": float(loss_xai.item()),
            "loss_pred": float(loss_pred.item()),
            "loss_patch": float(loss_patch.item()),
            "loss_entropy": float(loss_entropy.item()),
            "pred_label": pred_label,
            "class_preserved": preserved,
            "map_change": map_change,
        })
        delta = delta.detach().requires_grad_(True)

    with torch.no_grad():
        adv_normalized = normalize_ImageNet1k(selected_image)
        patch_features = F.normalize(token_encoder(adapter, adv_normalized).float(), dim=-1)
        adv_similarity = float(torch.einsum("bpd,d->bp", patch_features, target_text).mean().item())
        selected_pred_label = int(classifier(adv_normalized).argmax(dim=-1).item())
    return selected_image, {
        "clean_similarity": clean_similarity,
        "adv_similarity": adv_similarity,
        "selected_step": selected_step,
        "selected_map_change": selected_change,
        "selected_pred_label": selected_pred_label,
        "clean_pred_label": clean_pred_label,
        "trace": trace,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable; use --device cpu or a CUDA-enabled PyTorch environment")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    classnames = load_classnames(args.classnames_path)
    if len(classnames) < 2:
        raise ValueError("At least two class names are required")
    adapter = SigLIPAttackAdapter(args, device)
    if adapter.image_size != 224:
        raise ValueError("X-Shift causal reruns currently require a 224x224 SigLIP model")
    if args.gt_label is not None and not 0 <= args.gt_label < len(classnames):
        raise ValueError("--gt-label is outside the loaded class names")
    depth = len(adapter.model.vision_model.encoder.layers)
    if not -1 <= args.game_start_layer < depth:
        raise ValueError(f"--game-start-layer must be -1 or in [0, {depth - 1}]")
    args.game_start_layer = depth - 1 if args.game_start_layer == -1 else args.game_start_layer
    classifier = build_zero_shot_siglip_classifier(adapter, classnames, args.prompt_template)
    metric_model = build_causal_metric_model(classifier)
    text_features = F.normalize(classifier.zero_shot_weights.detach(), dim=0)
    blur = build_blur_substrate(args.kernel_size, args.kernel_sigma)
    with args.sample_path.open(encoding="utf-8") as stream:
        samples = json.load(stream)
    if not isinstance(samples, dict) or not samples:
        raise ValueError("--sample-path must contain a nonempty {sample_id: relative_image_path} object")
    output_dir = args.output_dir / EXPLAINER_VERSION
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model_type="SigLIP", attack_type="xshift", explainer_version=EXPLAINER_VERSION,
                  image_size=adapter.image_size, prompt_template=args.prompt_template)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for sample_id, relative_path in tqdm(samples.items()):
        with Image.open(args.img_dir / relative_path) as source:
            image = source.convert("RGB").resize((adapter.image_size, adapter.image_size), Image.Resampling.BICUBIC)
        raw = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1).unsqueeze(0).div_(255).to(device)
        normalized = normalize_ImageNet1k(raw)
        _, _, clean_label, clean_confidence = predict_zero_shot_clip(classifier, normalized, device)
        target_label = clean_label if args.target_source == "pred" else args.gt_label
        adversarial, details = run_xshift_attack(
            adapter, classifier, raw, clean_label, target_label, text_features,
            eps=args.eps / 255, alpha=args.alpha / 255,
            xshift_steps=args.xshift_steps, xai_topk_ratio=args.xai_topk_ratio,
            xai_alpha=args.xai_alpha, lambda_pred=args.lambda_pred,
            lambda_patch=args.lambda_patch, lambda_ent=args.lambda_ent, margin=args.margin,
        )
        adv_normalized = normalize_ImageNet1k(adversarial)
        _, _, adv_label, adv_confidence = predict_zero_shot_clip(classifier, adv_normalized, device)
        target_texts = [classnames[adv_label]]
        with torch.no_grad():
            text_embedding = F.normalize(adapter.encode_text(target_texts), dim=-1)
        logging.info(
            "%s: clean=%s (%.4f), adv=%s (%.4f), selected_step=%d map_change=%.6f",
            sample_id, classnames[clean_label], clean_confidence, classnames[adv_label],
            adv_confidence, details["selected_step"], details["selected_map_change"],
        )
        for method in args.hm_types:
            sample_dir = output_dir / method / "xshift" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_image(adversarial.detach().cpu(), str(sample_dir / "adversarial_image_xshift.png"))
            heatmap = adapter.generate_hm(
                method, adv_normalized, text_embedding, target_texts, Resize((224, 224)),
            ).cpu()
            save_saliency_outputs(heatmap, image, str(sample_dir), stem=f"xshift_{method}_saliency")
            rerun = {}
            for mode in ("del", "ins"):
                substrate = torch.zeros_like if mode == "del" else blur
                metric = CausalMetric(metric_model, mode, args.step, substrate)
                process_dir = sample_dir / f"steps_{mode}"
                curve = metric.single_run(
                    adv_normalized.cpu(), heatmap.numpy(), verbose=args.verbose,
                    save_to=str(process_dir) if args.save_process else None,
                )
                save_causal_metric_summary(
                    image_tensor=adv_normalized.cpu(),
                    final_tensor=torch.zeros_like(adv_normalized.cpu()) if mode == "del" else adv_normalized.cpu(),
                    scores=curve, output_path=str(sample_dir / f"{mode}_summary.png"),
                    mode=mode, class_name=classnames[adv_label], preprocess=adapter.preprocess,
                )
                rerun[mode] = {"curve": curve.tolist(), "auc": float(auc(curve))}
            payload = {
                "model_type": "SigLIP", "attack_type": "xshift", "eval_hm_type": method,
                "target_source": args.target_source, "target_label": int(target_label),
                "target_classname": classnames[target_label], "pred_label_clean": clean_label,
                "pred_classname_clean": classnames[clean_label], "pred_confidence_clean": clean_confidence,
                "pred_label_adv": adv_label, "pred_classname_adv": classnames[adv_label],
                "pred_confidence_adv": adv_confidence, **details, "rerun": rerun,
            }
            (sample_dir / "curve_information.json").write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )


if __name__ == "__main__":
    main()
