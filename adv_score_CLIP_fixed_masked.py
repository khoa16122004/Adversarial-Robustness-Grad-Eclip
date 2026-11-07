"""CLIP fixed-mask attacks; final AUC always uses fresh adversarial saliency."""

import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-model", default="ViT-B/16")
    parser.add_argument("--clip-checkpoint", default=None)
    parser.add_argument(
        "--hm-type", default="eclip",
        choices=["selfattn", "gradcam", "maskclip", "eclip", "eclip-wo-ksim",
                 "game", "rollout", "surgery", "m2ib", "rise"],
    )
    parser.add_argument("--mask-source", choices=["clean", "random"], default="clean")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the fixed random pixel ranking")
    parser.add_argument("--target-source", choices=["pred", "gt"], default="pred")
    parser.add_argument("--gt-label", type=int, default=None)
    parser.add_argument("--step", type=int, default=224)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=float, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="fixed_mask_eval_outputs")
    parser.add_argument("--img-dir", default="Imagenet/val")
    parser.add_argument("--sample-path", default="vit_b_16_1k.json")
    parser.add_argument("--classnames-path", default=None)
    parser.add_argument("--prompt-template", default="a photo of {}")
    parser.add_argument("--save-process", action="store_true")
    parser.add_argument("--eps", type=float, default=16.0, help="L-infinity budget in pixel values")
    parser.add_argument("--alpha", type=float, default=4.0, help="PGD step size in pixel values")
    parser.add_argument("--pgd-steps", type=int, default=50)
    parser.add_argument(
        "--process-batch-size", "--deletion-batch-size",
        dest="process_batch_size", type=int, default=32,
    )
    parser.add_argument("--mode", choices=["del", "ins", "del+ins"], default="del")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.step < 1 or args.pgd_steps < 0 or args.eps < 0 or args.alpha < 0:
        parser.error("--step must be positive; --pgd-steps, --eps and --alpha must be non-negative")
    if args.mask_source == "random" and args.seed < 0:
        parser.error("--seed must be non-negative for random masks")
    if args.target_source == "gt" and args.gt_label is None:
        parser.error("--gt-label is required when --target-source gt")
    if args.clip_checkpoint and not Path(args.clip_checkpoint).is_file():
        raise FileNotFoundError(f"--clip-checkpoint not found: {args.clip_checkpoint}")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms import Resize
    from torchvision.utils import save_image
    from tqdm import tqdm
    try:
        from CLIP import clip
    except ImportError:
        import clip
    from RISE.evaluation import (
        AdversarialCausalMetricFixedMask, JointAdversarialCausalMetricFixedMask,
        CausalMetric, auc,
    )
    from util import (
        build_blur_substrate, build_causal_metric_model, build_zero_shot_clip_classifier,
        denorm_ImageNet1k, generate_hm, load_classnames, normalize_ImageNet1k,
        predict_zero_shot_clip, save_causal_metric_summary, save_saliency_outputs,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    classnames = load_classnames(args.classnames_path)
    if args.gt_label is not None and not 0 <= args.gt_label < len(classnames):
        raise ValueError("--gt-label must be a valid class index")
    blur_fn = build_blur_substrate(args.kernel_size, args.kernel_sigma)
    clip_model, preprocess = clip.load(args.clip_checkpoint or args.clip_model, device=device)
    clip_model.eval()
    classifier, _ = build_zero_shot_clip_classifier(
        clip_model, device=device, classnames=classnames,
        prompt_template=args.prompt_template, num_classes_per_batch=10, use_tqdm=True,
    )
    metric_model = build_causal_metric_model(classifier)
    strategy = "clean" if args.mask_source == "clean" else f"random_seed_{args.seed}"
    output_dir = Path(args.output_dir) / strategy / args.hm_type / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.sample_path).open(encoding="utf-8") as handle:
        samples = json.load(handle)

    for folder_name, image_name in tqdm(samples.items()):
        sample_dir = output_dir / folder_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        img_path = Path(args.img_dir) / image_name
        with Image.open(img_path) as image:
            resolution = clip_model.visual.input_resolution
            resized_image = image.convert("RGB").resize((resolution, resolution), Image.BICUBIC)
        image_normalized = preprocess(resized_image).unsqueeze(0).to(device)
        image_raw = denorm_ImageNet1k(image_normalized)
        metric_resize = Resize(tuple(image_normalized.shape[-2:]))
        _, _, pred_label, pred_confidence = predict_zero_shot_clip(
            classifier, image_normalized, device
        )
        target_label = pred_label if args.target_source == "pred" else args.gt_label
        target_texts = [classnames[target_label]]
        with torch.no_grad():
            text_embedding = F.normalize(
                clip_model.encode_text(clip.tokenize(target_texts).to(device)), dim=-1
            )

        if args.mode == "del+ins":
            metric = JointAdversarialCausalMetricFixedMask(
                metric_model, clip_model, args.step, torch.zeros_like, blur_fn,
                args.hm_type, text_embedding, target_texts, metric_resize, preprocess,
            )
            image_filename = "adversarial_image_del_ins.png"
            stem = "joint"
        else:
            metric = AdversarialCausalMetricFixedMask(
                metric_model, clip_model, args.mode, args.step,
                torch.zeros_like if args.mode == "del" else blur_fn,
                args.hm_type, text_embedding, target_texts, metric_resize, preprocess,
            )
            image_filename = f"adversarial_image_{args.mode}.png"
            stem = args.mode
        x_adv, details = metric.single_run(
            image_raw, generate_hm, target_class=target_label,
            eps=args.eps / 255.0, alpha=args.alpha / 255.0,
            pgd_steps=args.pgd_steps, process_batch_size=args.process_batch_size,
            verbose=args.verbose, mask_source=args.mask_source, seed=args.seed,
        )
        if not torch.isfinite(x_adv).all():
            raise ValueError("Attack returned a non-finite image")
        if (x_adv - image_raw).abs().max().item() > args.eps / 255.0 + 1e-6:
            raise ValueError("Attack exceeded the L-infinity perturbation budget")
        np.save(sample_dir / "fixed_salient_order.npy", metric.fixed_salient_order)
        x_adv = x_adv.detach().cpu()
        save_image(x_adv, str(sample_dir / image_filename))
        x_adv_normalized = normalize_ImageNet1k(x_adv)
        _, _, adv_label, adv_confidence = predict_zero_shot_clip(
            classifier, x_adv_normalized, device
        )

        # Evaluation is intentionally independent of the ranking used by the attack.
        heatmap = generate_hm(
            clip_model, args.hm_type, x_adv_normalized, text_embedding,
            target_texts, metric_resize, preprocess,
        )
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.detach().cpu().numpy()
        heatmap = np.asarray(heatmap)
        if heatmap.size != image_raw.shape[-2] * image_raw.shape[-1] or not np.isfinite(heatmap).all():
            raise ValueError("Adversarial saliency must contain one finite value per image pixel")
        rerun_results = {}
        for rerun_mode in ("del", "ins"):
            substrate_fn = torch.zeros_like if rerun_mode == "del" else blur_fn
            causal_metric = CausalMetric(metric_model, rerun_mode, args.step, substrate_fn)
            process_dir = sample_dir / f"steps_{stem}_{rerun_mode}"
            if args.save_process:
                process_dir.mkdir(parents=True, exist_ok=True)
            save_saliency_outputs(
                heatmap, resized_image, str(sample_dir),
                stem=f"{stem}_{rerun_mode}_adv_{args.hm_type}_saliency",
            )
            curve = causal_metric.single_run(
                x_adv_normalized, heatmap, verbose=args.verbose,
                save_to=str(process_dir) if args.save_process else None,
            )
            save_causal_metric_summary(
                image_tensor=x_adv_normalized,
                final_tensor=torch.zeros_like(x_adv) if rerun_mode == "del" else x_adv_normalized,
                scores=curve, output_path=str(sample_dir / f"{stem}_{rerun_mode}_summary.png"),
                mode=rerun_mode, class_name=classnames[pred_label], preprocess=preprocess,
            )
            rerun_results[rerun_mode] = {"curve": curve.tolist(), "auc": float(auc(curve))}

        payload = {
            "image_path": str(img_path), "clip_model": args.clip_model,
            "hm_type": args.hm_type, "attack_mode": args.mode,
            "mask_source": details["mask_source"], "seed": details["seed"],
            "fixed_salient_order_file": "fixed_salient_order.npy",
            "evaluation_map_source": "adversarial",
            "target_source": args.target_source,
            "retain_class_label": int(target_label), "retain_classname": classnames[target_label],
            "pred_label": pred_label, "pred_confidence": pred_confidence,
            "clean_prob": details["clean_prob"], "adv_prob": details["adv_prob"],
            "clean_score": details["clean_prob"], "adv_score": details["adv_prob"],
            "adv_pred_label": adv_label, "adv_pred_classname": classnames[adv_label],
            "adv_pred_confidence": adv_confidence, "adv_pred_retain": bool(adv_label == target_label),
            "pgd_loss": details["loss"], "pgd_trace": details["pgd_trace"],
            "rerun": rerun_results,
        }
        if args.mode == "del+ins":
            payload["combined_score"] = rerun_results["del"]["auc"] + rerun_results["ins"]["auc"]
            payload["score_formula"] = "joint_del_auc + joint_ins_auc"
        with (sample_dir / "curve_information.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
