"""BLIP adaptation of adv_score_CLIP.py using the same causal attack engines."""

import argparse
import json
import logging
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize
from torchvision.utils import save_image
from tqdm import tqdm

from blip_attack_adapter import (
    BLIPAttackAdapter, HM_METHODS, build_zero_shot_blip_classifier, generate_hm,
)
from RISE.evaluation import AdversarialCausalMetric, CausalMetric, JointAdversarialCausalMetric, auc
from test_blip_explainers import DATA_ROOT, REPO_ROOT
from util import (
    DEFAULT_PROMPT_TEMPLATE, build_blur_substrate, build_causal_metric_model,
    denorm_ImageNet1k, load_classnames, normalize_ImageNet1k, predict_zero_shot_clip,
    save_causal_metric_summary, save_saliency_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blip-root", type=Path, default=REPO_ROOT / "BLIP")
    parser.add_argument("--blip-checkpoint", "--checkpoint", dest="checkpoint",
                        type=Path, default=DATA_ROOT / "BLIP.pth")
    parser.add_argument("--vit", choices=("base", "large"), default="base")
    # The shared causal engines use HW=224*224 and .cuda() in clean reruns.
    parser.add_argument("--image-size", type=int, choices=(224,), default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--last-heads", choices=("1", "native"), default="1")
    parser.add_argument("--hm-type", choices=tuple(HM_METHODS), default="eclip")
    parser.add_argument("--target-source", choices=("pred", "gt"), default="pred")
    parser.add_argument("--gt-label", type=int)
    parser.add_argument("--step", type=int, default=224)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=float, default=5)
    parser.add_argument("--output-dir", default=str(DATA_ROOT / "attack_results"))
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--sample-path", required=True)
    parser.add_argument("--classnames-path")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--save-process", action="store_true")
    parser.add_argument("--eps", type=float, default=32.0)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--pgd-steps", type=int, default=50)
    parser.add_argument("--process-batch-size", "--deletion-batch-size",
                        dest="process_batch_size", type=int, default=32)
    parser.add_argument("--mode", choices=("del", "ins", "del+ins"), default="del")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--game-start-layer", type=int, default=-1)
    parser.add_argument("--rise-masks", type=int, default=2000)
    parser.add_argument("--rise-grid", type=int, default=8)
    parser.add_argument("--rise-prob", type=float, default=0.5)
    parser.add_argument("--rise-batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for option, path in (("--blip-checkpoint", args.checkpoint),
                         ("--sample-path", Path(args.sample_path)),
                         ("--blip-root", args.blip_root / "models" / "blip_itm.py")):
        if not path.is_file():
            parser.error(f"{option}: file not found: {path}")
    if not Path(args.img_dir).is_dir():
        parser.error(f"--img-dir: directory not found: {args.img_dir}")
    try:
        device = torch.device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    if device.type != "cuda":
        parser.error("Shared RISE causal reruns require CUDA; use --device cuda or cuda:N")
    for name in ("step", "pgd_steps", "process_batch_size", "rise_masks", "rise_grid", "rise_batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.eps < 0 or args.alpha < 0:
        parser.error("--eps and --alpha must be nonnegative")
    if args.kernel_size <= 0 or args.kernel_size % 2 != 1 or args.kernel_sigma <= 0:
        parser.error("Blur kernel size must be positive and odd; sigma must be positive")
    if not 0 < args.rise_prob < 1 or args.seed < 0:
        parser.error("--rise-prob must be in (0, 1); --seed must be nonnegative")
    depth = 12 if args.vit == "base" else 24
    if not -1 <= args.game_start_layer < depth:
        parser.error("--game-start-layer is outside the BLIP visual block range")
    if args.target_source == "gt" and args.gt_label is None:
        parser.error("--gt-label is required with --target-source gt (same contract as CLIP)")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("The shared causal metric pipeline requires a CUDA-enabled PyTorch runtime")
    device = torch.device(args.device)
    torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.blip_root.resolve()))
    classnames = load_classnames(args.classnames_path)
    if len(classnames) < 2:
        raise ValueError("At least two class names are required for zero-shot attack evaluation")
    if args.gt_label is not None and not 0 <= args.gt_label < len(classnames):
        raise ValueError("--gt-label is outside the loaded class names")
    if args.last_heads == "1":
        logging.warning("Single-head notebook adaptation is used for BOTH classifier and explainer.")
    adapter = BLIPAttackAdapter(args, device)
    preprocess = adapter.preprocess
    classifier = build_zero_shot_blip_classifier(adapter, classnames, args.prompt_template)
    metric_model = build_causal_metric_model(classifier)
    blur_fn = build_blur_substrate(args.kernel_size, args.kernel_sigma)
    output_dir = Path(args.output_dir) / args.hm_type / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.sample_path, encoding="utf-8") as stream:
        samples = json.load(stream)
    if not isinstance(samples, dict) or not samples:
        raise ValueError("--sample-path must contain a nonempty {sample_id: relative_image_path} object")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "model": "BLIP_ITC",
        "logit_scale": classifier.logit_scale,
        "torch_version": torch.__version__,
        "attack_target": "clean predicted class (unchanged shared PGD behavior)",
        "explainer_text": "bare class name (same as adv_score_CLIP.py)",
    })
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for folder_name, image_name in tqdm(samples.items()):
        sample_dir = output_dir / folder_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(Path(args.img_dir) / image_name) as source:
            image = source.convert("RGB")
        resized_image = image.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        image_normalized = preprocess(resized_image).unsqueeze(0)
        image_raw = denorm_ImageNet1k(image_normalized).to(device)
        metric_resize = Resize(tuple(image_normalized.shape[-2:]))
        _, _, pred_label, pred_confidence = predict_zero_shot_clip(classifier, image_normalized, device)
        target_label = pred_label if args.target_source == "pred" else args.gt_label
        target_texts = [classnames[target_label]]
        with torch.no_grad():
            text_embedding = F.normalize(adapter.encode_text(target_texts), dim=-1)
        metric_args = (args.hm_type, text_embedding, target_texts, metric_resize, preprocess)
        if args.mode == "del+ins":
            attack = JointAdversarialCausalMetric(
                metric_model, adapter, args.step, torch.zeros_like, blur_fn, *metric_args,
            )
            prefix = "joint"
            image_filename = "adversarial_image_del_ins.png"
        else:
            substrate = torch.zeros_like if args.mode == "del" else blur_fn
            attack = AdversarialCausalMetric(
                metric_model, adapter, args.mode, args.step, substrate, *metric_args,
            )
            prefix = args.mode
            image_filename = f"adversarial_image_{args.mode}.png"
        x_adv, details = attack.single_run(
            image_raw, generate_hm, eps=args.eps / 255.0, alpha=args.alpha / 255.0,
            pgd_steps=args.pgd_steps, process_batch_size=args.process_batch_size,
        )
        x_adv = x_adv.detach().cpu()
        if not torch.isfinite(x_adv).all():
            raise ValueError("Attack returned a nonfinite image")
        if (x_adv - image_raw.cpu()).abs().max().item() > args.eps / 255.0 + 1e-6:
            raise ValueError("Attack exceeded the requested L-infinity perturbation bound")
        save_image(x_adv, str(sample_dir / image_filename))
        x_adv_normalized = normalize_ImageNet1k(x_adv)
        _, _, adv_label, adv_confidence = predict_zero_shot_clip(classifier, x_adv_normalized, device)
        heatmap = generate_hm(
            adapter, args.hm_type, x_adv_normalized, text_embedding, target_texts,
            metric_resize, preprocess,
        ).cpu()
        rerun_results = {}
        for mode in ("del", "ins"):
            substrate = torch.zeros_like if mode == "del" else blur_fn
            metric = CausalMetric(metric_model, mode, args.step, substrate)
            process_dir = sample_dir / f"steps_{prefix}_{mode}"
            if args.save_process:
                process_dir.mkdir(parents=True, exist_ok=True)
            save_saliency_outputs(
                heatmap.numpy(), resized_image, str(sample_dir),
                stem=f"{prefix}_{mode}_adv_{args.hm_type}_saliency",
            )
            with torch.no_grad():
                curve = metric.single_run(
                    x_adv_normalized, heatmap.numpy(), verbose=args.verbose,
                    save_to=str(process_dir) if args.save_process else None,
                )
            save_causal_metric_summary(
                image_tensor=x_adv_normalized,
                final_tensor=torch.zeros_like(x_adv) if mode == "del" else x_adv_normalized,
                scores=curve, output_path=str(sample_dir / f"{prefix}_{mode}_summary.png"),
                mode=mode, class_name=classnames[pred_label], preprocess=preprocess,
            )
            rerun_results[mode] = {"curve": curve.tolist(), "auc": float(auc(curve))}
        payload = {
            "attack_mode": args.mode,
            "retain_class_label": pred_label,
            "retain_classname": classnames[pred_label],
            "clean_prob": details["clean_prob"],
            "adv_prob": details["adv_prob"],
            "clean_score": details["clean_prob"],
            "adv_score": details["adv_prob"],
            "adv_pred_label": adv_label,
            "adv_pred_classname": classnames[adv_label],
            "adv_pred_confidence": adv_confidence,
            "adv_pred_retain": bool(adv_label == pred_label),
            "pgd_loss": details.get("loss", []),
            "pgd_trace": details.get("pgd_trace", []),
            "rerun": rerun_results,
            "model": "BLIP_ITC",
            "pred_label": pred_label,
            "pred_classname": classnames[pred_label],
            "pred_confidence": pred_confidence,
            "target_source": args.target_source,
            "target_label": target_label,
            "target_classname": classnames[target_label],
            "selected_step": details["selected_step"],
            "selected_class_preserved": details["selected_class_preserved"],
            "selection_fallback_last_step": details["selection_fallback_last_step"],
            "selected_loss": details["selected_loss"],
        }
        if args.mode == "del+ins":
            payload["combined_score"] = rerun_results["del"]["auc"] + rerun_results["ins"]["auc"]
            payload["score_formula"] = "joint_del_auc + joint_ins_auc"
        (sample_dir / "curve_information.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )


if __name__ == "__main__":
    main()
