"""Adapt the existing X-Shift objective and selection rule to BLIP ITC."""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize
from torchvision.utils import save_image
from tqdm import tqdm

from adv_score_BLIP import _write_json_atomic
from blip_attack_adapter import BLIPAttackAdapter, build_zero_shot_blip_classifier
from RISE.evaluation import CausalMetric, auc
from util import (
    build_blur_substrate, build_causal_metric_model, denorm_ImageNet1k,
    load_classnames, normalize_ImageNet1k, predict_zero_shot_clip,
    save_causal_metric_summary, save_saliency_outputs,
)
from XShift_score import run_xshift_attack


METHODS = ("eclip", "game", "gradcam", "maskclip", "rise")


def encode_blip_tokens(adapter, images):
    return adapter.encode_tokens(images)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path,
                        help="Optional existing BLIP configuration; otherwise supply dataset inputs")
    parser.add_argument("--img-dir")
    parser.add_argument("--sample-path")
    parser.add_argument("--classnames-path")
    parser.add_argument("--vit", choices=("base", "large"), default="base")
    parser.add_argument("--last-heads", choices=("1", "native"), default="1")
    parser.add_argument("--image-size", type=int, choices=(224,), default=224)
    parser.add_argument("--step", type=int, default=224)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=float, default=5)
    parser.add_argument("--prompt-template", default="a photo of {}")
    parser.add_argument("--game-start-layer", type=int, default=-1)
    parser.add_argument("--rise-masks", type=int, default=2000)
    parser.add_argument("--rise-grid", type=int, default=8)
    parser.add_argument("--rise-prob", type=float, default=0.5)
    parser.add_argument("--rise-batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Dataset epsilon root; appends <explainer>/xshift/<sample>")
    parser.add_argument("--blip-root", type=Path, default=Path("BLIP"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--eps", type=float, default=16)
    parser.add_argument("--alpha", type=float, default=4)
    parser.add_argument("--xshift-steps", type=int, default=50)
    parser.add_argument("--xai-topk-ratio", type=float, default=0.1)
    parser.add_argument("--xai-alpha", type=float, default=0.5)
    parser.add_argument("--lambda-pred", type=float, default=1)
    parser.add_argument("--lambda-patch", type=float, default=0.3)
    parser.add_argument("--lambda-ent", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--save-process", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.run_config is None:
        if not args.checkpoint or not args.img_dir or not args.sample_path:
            parser.error("Without --run-config, --checkpoint, --img-dir and --sample-path are required")
        if min(args.step, args.rise_masks, args.rise_grid, args.rise_batch_size) <= 0:
            parser.error("Step and RISE sizes must be positive")
        if not 0 < args.rise_prob < 1 or args.seed < 0:
            parser.error("RISE probability must be in (0, 1); seed must be nonnegative")
        if args.kernel_size <= 0 or args.kernel_size % 2 != 1 or not math.isfinite(args.kernel_sigma) or args.kernel_sigma <= 0:
            parser.error("Blur kernel must be positive and odd; sigma must be finite and positive")
        if not -1 <= args.game_start_layer < (12 if args.vit == "base" else 24):
            parser.error("GAME start layer is outside the visual block range")
    elif any(value is not None for value in (args.img_dir, args.sample_path, args.classnames_path)):
        parser.error("Dataset inputs cannot be combined with --run-config")
    for name in ("eps", "alpha", "xai_alpha", "lambda_pred", "lambda_patch", "lambda_ent", "margin"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not 0 < args.xai_topk_ratio <= 1 or args.xshift_steps <= 0:
        parser.error("--xai-topk-ratio must be in (0, 1]; --xshift-steps must be positive")
    args.methods = list(dict.fromkeys(args.methods))
    return args


def completed(folder, method, image_size, step, save_process):
    required = [
        folder / "curve_information.json", folder / "adversarial_image_xshift.png",
    ]
    for mode in ("del", "ins"):
        required.append(folder / f"{mode}_summary.png")
        for suffix in (".npy", ".png", "_overlay.png"):
            required.append(folder / f"{mode}_xshift_{method}_saliency{suffix}")
        if save_process:
            steps = (image_size * image_size + step - 1) // step
            required.extend(folder / f"steps_{mode}" / f"{index:03d}.png" for index in range(steps + 1))
    return all(path.is_file() for path in required)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.run_config is not None:
        config = json.loads(args.run_config.read_text(encoding="utf-8"))
    else:
        config = {
            key: getattr(args, key) for key in (
                "img_dir", "sample_path", "classnames_path", "vit", "last_heads",
                "image_size", "step", "kernel_size", "kernel_sigma",
                "prompt_template", "game_start_layer", "rise_masks", "rise_grid",
                "rise_prob", "rise_batch_size", "seed",
            )
        }
        config.update(model="BLIP_ITC", target_source="pred", checkpoint=str(args.checkpoint))
    if config.get("model") != "BLIP_ITC":
        raise ValueError(f"Expected a BLIP attack configuration: {args.run_config}")
    if config.get("target_source", "pred") != "pred":
        raise ValueError("This runner requires predicted-class attack/clean targets")
    if config.get("image_size") != 224 or config.get("step", 0) <= 0:
        raise ValueError("Shared causal reruns require image_size=224 and a positive step")
    config["checkpoint"] = str((args.checkpoint or Path(config["checkpoint"])).resolve())
    config["blip_root"] = str(args.blip_root.resolve())
    for path in (
        Path(config["checkpoint"]), args.blip_root / "models" / "blip_itm.py",
        Path(config["sample_path"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    image_root = Path(config["img_dir"])
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    if config.get("classnames_path") and not Path(config["classnames_path"]).is_file():
        raise FileNotFoundError(f"Class names file not found: {config['classnames_path']}")
    samples = json.loads(Path(config["sample_path"]).read_text(encoding="utf-8"))
    if not isinstance(samples, dict) or not samples:
        raise ValueError("Sample JSON must contain a nonempty sample-id to image-path mapping")
    for sample_id, relative_path in samples.items():
        if not isinstance(sample_id, str) or Path(sample_id).name != sample_id or sample_id in (".", ".."):
            raise ValueError(f"Invalid sample ID: {sample_id!r}")
        if not isinstance(relative_path, str) or not (image_root / relative_path).is_file():
            raise FileNotFoundError(f"Sample image not found: {relative_path}")

    attack_options = {
        key: getattr(args, key)
        for key in ("eps", "alpha", "xshift_steps", "xai_topk_ratio", "xai_alpha",
                    "lambda_pred", "lambda_patch", "lambda_ent", "margin")
    }
    recorded = {
        key: config.get(key) for key in (
            "model", "checkpoint", "blip_root", "vit", "last_heads", "image_size", "step",
            "kernel_size", "kernel_sigma", "img_dir", "sample_path", "classnames_path",
            "prompt_template", "game_start_layer", "rise_masks", "rise_grid",
            "rise_prob", "rise_batch_size", "seed", "target_source",
        )
    }
    recorded.update(attack_options, mode="xshift", save_process=args.save_process)
    for method in args.methods:
        folder = args.output_dir / method / "xshift"
        config_path = folder / "run_config.json"
        expected = dict(recorded, hm_type=method)
        if config_path.exists() and not args.overwrite:
            if json.loads(config_path.read_text(encoding="utf-8")) != expected:
                raise ValueError(f"Existing X-Shift settings differ: {config_path}; use --overwrite")
        elif folder.exists() and any(folder.iterdir()) and not args.overwrite:
            raise ValueError(f"Unconfigured existing X-Shift outputs: {folder}; use --overwrite")
        folder.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(config_path, expected)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Shared CausalMetric reruns require CUDA")
    torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    torch.manual_seed(config["seed"])
    sys.path.insert(0, config["blip_root"])
    adapter_options = dict(config, checkpoint=Path(config["checkpoint"]),
                           blip_root=Path(config["blip_root"]))
    adapter = BLIPAttackAdapter(SimpleNamespace(**adapter_options), device)
    classnames = load_classnames(config.get("classnames_path"))
    if len(classnames) < 2:
        raise ValueError("X-Shift requires at least two classifier classes")
    classifier = build_zero_shot_blip_classifier(adapter, classnames, config["prompt_template"])
    model = build_causal_metric_model(classifier)
    blur = build_blur_substrate(config["kernel_size"], config["kernel_sigma"])
    text_features = F.normalize(classifier.zero_shot_weights.detach(), dim=0)
    metric_resize = Resize((224, 224))
    if config["last_heads"] == "1":
        logging.warning("Using single-head BLIP adaptation for both X-Shift and explanations")
    for sample_id, relative_path in tqdm(samples.items(), desc="BLIP X-Shift"):
        pending = [
            method for method in args.methods
            if args.overwrite or not completed(
                args.output_dir / method / "xshift" / sample_id, method,
                224, config["step"], args.save_process,
            )
        ]
        if not pending:
            continue
        with Image.open(image_root / relative_path) as source:
            image = source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        normalized = adapter.preprocess(image).unsqueeze(0).to(device)
        raw = denorm_ImageNet1k(normalized).detach()
        _, _, label, confidence = predict_zero_shot_clip(classifier, normalized, device)
        attack_kwargs = dict(attack_options, eps=args.eps / 255, alpha=args.alpha / 255)
        adversarial, details = run_xshift_attack(
            adapter, classifier, raw, label, label, text_features,
            **attack_kwargs, token_encoder=encode_blip_tokens,
        )
        if not torch.isfinite(adversarial).all() or (
            (adversarial - raw).abs().max().item() > args.eps / 255 + 1e-6
        ):
            raise ValueError(f"Invalid or out-of-budget X-Shift image: {sample_id}")
        adv_norm = normalize_ImageNet1k(adversarial)
        _, _, adv_label, adv_confidence = predict_zero_shot_clip(classifier, adv_norm, device)
        if adv_label != label:
            raise ValueError(f"X-Shift selected a class-changing candidate: {sample_id}")
        texts = [classnames[adv_label]]
        with torch.no_grad():
            embedding = F.normalize(adapter.encode_text(texts), dim=-1)
        logging.info("%s: selected_step=%d map_change=%.6f", sample_id,
                     details["selected_step"], details["selected_map_change"])
        for method in pending:
            folder = args.output_dir / method / "xshift" / sample_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "road_deletion_information.json").unlink(missing_ok=True)
            save_image(adversarial.cpu(), str(folder / "adversarial_image_xshift.png"))
            heatmap = adapter.generate_hm(method, adv_norm, embedding, texts, metric_resize)
            saliency = heatmap.cpu().numpy()
            rerun = {}
            for mode in ("del", "ins"):
                metric = CausalMetric(model, mode, config["step"], torch.zeros_like if mode == "del" else blur)
                steps_dir = folder / f"steps_{mode}"
                if args.save_process:
                    steps_dir.mkdir(exist_ok=True)
                save_saliency_outputs(saliency, image, str(folder), stem=f"{mode}_xshift_{method}_saliency")
                with torch.no_grad():
                    curve = metric.single_run(
                        adv_norm.cpu(), saliency, verbose=0,
                        save_to=str(steps_dir) if args.save_process else None,
                    )
                save_causal_metric_summary(
                    image_tensor=adv_norm.cpu(),
                    final_tensor=torch.zeros_like(adv_norm.cpu()) if mode == "del" else adv_norm.cpu(),
                    scores=curve, output_path=str(folder / f"{mode}_summary.png"),
                    mode=mode, class_name=classnames[adv_label], preprocess=adapter.preprocess,
                )
                rerun[mode] = {"curve": curve.tolist(), "auc": float(auc(curve))}
            _write_json_atomic(folder / "curve_information.json", {
                "model": "BLIP_ITC", "attack_type": "xshift", "attack_mode": "xshift",
                "eval_hm_type": method, "target_source": "pred", "target_label": label,
                "target_classname": classnames[label], "pred_label_clean": label,
                "pred_confidence_clean": confidence, "adv_pred_label": adv_label,
                "adv_pred_confidence": adv_confidence, "adv_pred_retain": True,
                "clean_similarity": details["clean_similarity"],
                "adv_similarity": details["adv_similarity"],
                "selected_step": details["selected_step"],
                "selected_map_change": details["selected_map_change"],
                "xshift_trace": details["trace"], "config": attack_options, "rerun": rerun,
            })
    logging.info("BLIP X-Shift complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
