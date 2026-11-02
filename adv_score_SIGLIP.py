"""SigLIP adversarial faithfulness evaluation using the shared causal engines."""

import argparse
import json
import logging
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize
from torchvision.utils import save_image
from tqdm import tqdm

from RISE.evaluation import (
    AdversarialCausalMetric, CausalMetric, JointAdversarialCausalMetric, auc,
)
from siglip_attack_adapter import (
    HM_METHODS, SigLIPAttackAdapter, build_zero_shot_siglip_classifier, generate_hm,
)
from test_siglip_explainers import EXPLAINER_VERSION, REPO_ROOT
from util import (
    build_blur_substrate, build_causal_metric_model, denorm_ImageNet1k,
    load_classnames, normalize_ImageNet1k, predict_zero_shot_clip,
    save_causal_metric_summary, save_saliency_outputs,
)


def _sample_outputs_are_complete(sample_dir: Path, args) -> bool:
    if args.mode == "del+ins":
        prefix = "joint"
        image_filename = "adversarial_image_del_ins.png"
    else:
        prefix = args.mode
        image_filename = f"adversarial_image_{args.mode}.png"

    required_outputs = [sample_dir / image_filename]
    for mode in ("del", "ins"):
        stem = f"{prefix}_{mode}_adv_{args.hm_type}_saliency"
        required_outputs.extend(
            sample_dir / f"{stem}{suffix}"
            for suffix in (".npy", ".png", "_overlay.png")
        )
        required_outputs.append(sample_dir / f"{prefix}_{mode}_summary.png")
        if args.save_process:
            process_dir = sample_dir / f"steps_{prefix}_{mode}"
            n_steps = (args.image_size * args.image_size + args.step - 1) // args.step
            required_outputs.extend(
                process_dir / f"{step:03d}.png" for step in range(n_steps + 1)
            )
    return all(path.is_file() for path in required_outputs)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/siglip-base-patch16-224",
                        help="Hugging Face model ID or local SigLIP model directory.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hm-type", choices=tuple(HM_METHODS), default="eclip")
    parser.add_argument("--target-source", choices=("pred", "gt"), default="pred")
    parser.add_argument("--gt-label", type=int)
    parser.add_argument("--step", type=int, default=224)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=float, default=5)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "SIGLIP_result"))
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--sample-path", required=True)
    parser.add_argument("--classnames-path")
    parser.add_argument("--prompt-template", default="a photo of {}")
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
    if not Path(args.img_dir).is_dir():
        parser.error(f"--img-dir: directory not found: {args.img_dir}")
    sample_path = Path(args.sample_path)
    if not sample_path.is_file():
        parser.error(f"--sample-path: file not found: {sample_path}")
    try:
        device = torch.device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    if device.type != "cuda":
        parser.error("Shared RISE causal reruns require CUDA; use --device cuda or cuda:N")
    for name in (
        "step", "pgd_steps", "process_batch_size", "rise_masks",
        "rise_grid", "rise_batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.eps < 0 or args.alpha < 0:
        parser.error("--eps and --alpha must be nonnegative")
    if args.kernel_size <= 0 or args.kernel_size % 2 != 1 or args.kernel_sigma <= 0:
        parser.error("Blur kernel size must be positive and odd; sigma must be positive")
    if not 0 < args.rise_prob < 1 or args.seed < 0:
        parser.error("--rise-prob must be in (0, 1); --seed must be nonnegative")
    if args.target_source == "gt" and args.gt_label is None:
        parser.error("--gt-label is required with --target-source gt")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("The shared causal metric pipeline requires a CUDA-enabled PyTorch runtime")
    device = torch.device(args.device)
    torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    torch.manual_seed(args.seed)

    classnames = load_classnames(args.classnames_path)
    if len(classnames) < 2:
        raise ValueError("At least two class names are required for zero-shot attack evaluation")
    if args.gt_label is not None and not 0 <= args.gt_label < len(classnames):
        raise ValueError("--gt-label is outside the loaded class names")
    adapter = SigLIPAttackAdapter(args, device)
    if adapter.image_size != 224:
        raise ValueError(
            f"The shared causal attack engines require 224x224 input, "
            f"but {args.model} uses {adapter.image_size}x{adapter.image_size}. "
            "Use google/siglip-base-patch16-224 or another 224px checkpoint."
        )
    depth = len(adapter.model.vision_model.encoder.layers)
    if not -1 <= args.game_start_layer < depth:
        raise ValueError(f"--game-start-layer must be -1 or in [0, {depth - 1}]")
    if args.game_start_layer == -1:
        args.game_start_layer = depth - 1

    classifier = build_zero_shot_siglip_classifier(
        adapter, classnames, args.prompt_template,
    )
    metric_model = build_causal_metric_model(classifier)
    blur_fn = build_blur_substrate(args.kernel_size, args.kernel_sigma)
    output_dir = Path(args.output_dir) / args.hm_type / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.sample_path, encoding="utf-8") as stream:
        samples = json.load(stream)
    if not isinstance(samples, dict) or not samples:
        raise ValueError("--sample-path must contain a nonempty {sample_id: relative_image_path} object")
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update({
        "model_type": "SigLIP",
        "explainer_version": EXPLAINER_VERSION,
        "image_size": adapter.image_size,
        "patch_size": adapter.patch_size,
        "logit_scale": classifier.logit_scale,
        "logit_bias": float(adapter.model.logit_bias.detach().item()),
        "prompt_template": args.prompt_template,
        "attack_target": "clean predicted class (shared PGD behavior)",
        "score_target": "SigLIP scaled cosine similarity plus logit bias",
    })
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    skipped_samples = 0
    for folder_name, image_name in tqdm(samples.items()):
        sample_dir = output_dir / folder_name
        if _sample_outputs_are_complete(sample_dir, args):
            skipped_samples += 1
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(Path(args.img_dir) / image_name) as source:
            image = source.convert("RGB")
        resized_image = image.resize(
            (adapter.image_size, adapter.image_size), Image.Resampling.BICUBIC,
        )
        image_normalized = adapter.preprocess(resized_image).unsqueeze(0)
        image_raw = denorm_ImageNet1k(image_normalized).to(device)
        metric_resize = Resize(tuple(image_normalized.shape[-2:]))
        _, _, pred_label, pred_confidence = predict_zero_shot_clip(
            classifier, image_normalized, device,
        )
        target_label = pred_label if args.target_source == "pred" else args.gt_label
        target_texts = [classnames[target_label]]
        with torch.no_grad():
            text_embedding = F.normalize(adapter.encode_text(target_texts), dim=-1)
        metric_args = (
            args.hm_type, text_embedding, target_texts, metric_resize, adapter.preprocess,
        )
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
        _, _, adv_label, adv_confidence = predict_zero_shot_clip(
            classifier, x_adv_normalized, device,
        )
        heatmap = adapter.generate_hm(
            args.hm_type, x_adv_normalized, text_embedding, target_texts,
            metric_resize,
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
                scores=curve,
                output_path=str(sample_dir / f"{prefix}_{mode}_summary.png"),
                mode=mode, class_name=classnames[pred_label], preprocess=adapter.preprocess,
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
            "model": "SigLIP",
            "explainer_version": EXPLAINER_VERSION,
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
        _write_json_atomic(sample_dir / "curve_information.json", payload)
    logging.info("Skipped %d completed samples", skipped_samples)


if __name__ == "__main__":
    main()
