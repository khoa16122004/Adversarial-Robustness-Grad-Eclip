"""Evaluate SigLIP explainers and clean-image deletion/insertion baselines."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm

from RISE.evaluation import CausalMetric, auc
from siglip_attack_adapter import HM_METHODS, SigLIPAttackAdapter, build_zero_shot_siglip_classifier
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
    parser.add_argument("--output-dir", type=Path, default=Path("SIGLIP_result_clean"))
    parser.add_argument("--classnames-path", type=Path)
    parser.add_argument("--prompt-template", default="a photo of {}")
    parser.add_argument("--hm-types", nargs="+", choices=tuple(HM_METHODS), default=["eclip", "game", "gradcam", "maskclip", "rise"])
    parser.add_argument("--target-source", choices=("pred", "gt"), default="pred")
    parser.add_argument("--gt-label", type=int)
    parser.add_argument("--step", type=int, default=224)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=float, default=5)
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
    if min(args.step, args.rise_masks, args.rise_grid, args.rise_batch_size) <= 0:
        parser.error("--step, --rise-masks, --rise-grid and --rise-batch-size must be positive")
    if not 0 < args.rise_prob < 1 or args.seed < 0:
        parser.error("--rise-prob must be in (0, 1) and --seed must be nonnegative")
    args.hm_types = list(dict.fromkeys(args.hm_types))
    return args


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
    if args.gt_label is not None and not 0 <= args.gt_label < len(classnames):
        raise ValueError("--gt-label is outside the loaded class names")
    adapter = SigLIPAttackAdapter(args, device)
    if adapter.image_size != 224:
        raise ValueError("The shared causal metric currently requires a 224x224 SigLIP model")
    depth = len(adapter.model.vision_model.encoder.layers)
    if not -1 <= args.game_start_layer < depth:
        raise ValueError(f"--game-start-layer must be -1 or in [0, {depth - 1}]")
    args.game_start_layer = depth - 1 if args.game_start_layer == -1 else args.game_start_layer
    classifier = build_zero_shot_siglip_classifier(adapter, classnames, args.prompt_template)
    metric_model = build_causal_metric_model(classifier)
    blur = build_blur_substrate(args.kernel_size, args.kernel_sigma)
    with args.sample_path.open(encoding="utf-8") as stream:
        samples = json.load(stream)
    if not isinstance(samples, dict) or not samples:
        raise ValueError("--sample-path must contain a nonempty {sample_id: relative_image_path} object")

    root = args.output_dir / EXPLAINER_VERSION
    root.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model_type="SigLIP", evaluation_type="clean", explainer_version=EXPLAINER_VERSION,
                  image_size=adapter.image_size, prompt_template=args.prompt_template)
    (root / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for sample_id, relative_path in tqdm(samples.items()):
        with Image.open(args.img_dir / relative_path) as source:
            image = source.convert("RGB").resize((adapter.image_size, adapter.image_size), Image.Resampling.BICUBIC)
        raw = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1).unsqueeze(0).div_(255)
        normalized = normalize_ImageNet1k(raw).to(device)
        _, _, pred_label, confidence = predict_zero_shot_clip(classifier, normalized, device)
        target_label = pred_label if args.target_source == "pred" else args.gt_label
        target_texts = [classnames[target_label]]
        with torch.no_grad():
            text_embedding = F.normalize(adapter.encode_text(target_texts), dim=-1)
        for method in args.hm_types:
            sample_dir = root / method / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            image_path = sample_dir / "clean_image_resized.png"
            image.save(image_path)
            heatmap = adapter.generate_hm(
                method, normalized, text_embedding, target_texts, Resize((224, 224)),
            ).cpu()
            save_saliency_outputs(heatmap, image, str(sample_dir), stem=f"clean_{method}_saliency")
            results = {}
            for mode in ("del", "ins"):
                substrate = torch.zeros_like if mode == "del" else blur
                metric = CausalMetric(metric_model, mode, args.step, substrate)
                process_dir = sample_dir / f"{mode}_steps"
                curve = metric.single_run(
                    normalized.cpu(), heatmap.numpy(), verbose=args.verbose,
                    save_to=str(process_dir) if args.save_process else None,
                )
                save_causal_metric_summary(
                    image_tensor=normalized.cpu(),
                    final_tensor=torch.zeros_like(normalized.cpu()) if mode == "del" else normalized.cpu(),
                    scores=curve, output_path=str(sample_dir / f"{mode}_summary.png"),
                    mode=mode, class_name=classnames[pred_label], preprocess=adapter.preprocess,
                )
                results[mode] = {"curve": curve.tolist(), "auc": float(auc(curve))}
            payload = {
                "model_type": "SigLIP", "evaluation_type": "clean", "hm_type": method,
                "target_source": args.target_source, "target_label": int(target_label),
                "target_classname": classnames[target_label], "pred_label": pred_label,
                "pred_classname": classnames[pred_label], "pred_confidence": confidence,
                "image_path": str(args.img_dir / relative_path), "rerun": results,
            }
            (sample_dir / "curve_information.json").write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
            logging.info("%s %s: clean deletion/insertion complete", sample_id, method)


if __name__ == "__main__":
    main()
