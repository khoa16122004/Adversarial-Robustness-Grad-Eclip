"""Generate BLIP clean saliency, causal curves, and ROAD baselines."""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from adv_score_BLIP import _write_json_atomic
from blip_attack_adapter import BLIPAttackAdapter, HM_METHODS, build_zero_shot_blip_classifier
from RISE.evaluation import CausalMetric, auc
from road_evaluation.deletion_metrics import METRIC_FILES, evaluate_both
from util import (
    DEFAULT_PROMPT_TEMPLATE, build_blur_substrate, build_causal_metric_model,
    load_classnames, predict_zero_shot_clip, save_causal_metric_summary,
    save_saliency_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blip-root", type=Path, default=Path("BLIP"))
    parser.add_argument("--checkpoint", "--blip-checkpoint", type=Path, required=True)
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--sample-path", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="Dataset folder name under the output root")
    parser.add_argument("--output-root", type=Path, default=Path("BLIP_result"))
    parser.add_argument("--classnames-path")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--methods", nargs="+", choices=tuple(HM_METHODS), default=["eclip", "game", "gradcam", "maskclip", "rise"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vit", choices=("base", "large"), default="base")
    parser.add_argument("--last-heads", choices=("1", "native"), default="1")
    parser.add_argument("--image-size", type=int, choices=(224,), default=224)
    parser.add_argument("--step", type=int, default=224)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--kernel-sigma", type=float, default=5)
    parser.add_argument("--game-start-layer", type=int, default=-1)
    parser.add_argument("--rise-masks", type=int, default=2000)
    parser.add_argument("--rise-grid", type=int, default=8)
    parser.add_argument("--rise-prob", type=float, default=0.5)
    parser.add_argument("--rise-batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--road-noise", type=float, default=0.01)
    parser.add_argument("--save-process", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.dataset or Path(args.dataset).name != args.dataset or args.dataset in (".", ".."):
        parser.error("--dataset must be a single folder name")
    for path in (args.checkpoint, args.sample_path, args.blip_root / "models" / "blip_itm.py"):
        if not path.is_file():
            parser.error(f"Required file not found: {path}")
    if not args.img_dir.is_dir():
        parser.error(f"Image directory not found: {args.img_dir}")
    if min(args.step, args.rise_masks, args.rise_grid, args.rise_batch_size) <= 0:
        parser.error("Step and RISE sizes must be positive")
    if not 0 < args.rise_prob < 1 or args.seed < 0:
        parser.error("RISE probability must be in (0, 1); seed must be nonnegative")
    if args.kernel_size <= 0 or args.kernel_size % 2 != 1 or not math.isfinite(args.kernel_sigma) or args.kernel_sigma <= 0:
        parser.error("Blur kernel must be positive and odd; sigma must be finite and positive")
    if not math.isfinite(args.road_noise) or args.road_noise < 0:
        parser.error("ROAD noise must be finite and nonnegative")
    if not -1 <= args.game_start_layer < (12 if args.vit == "base" else 24):
        parser.error("GAME start layer is outside the visual block range")
    args.methods = list(dict.fromkeys("eclip" if method == "gradeclip" else method for method in args.methods))
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Shared CausalMetric reruns require CUDA")
    torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.blip_root.resolve()))
    samples = json.loads(args.sample_path.read_text(encoding="utf-8"))
    if not isinstance(samples, dict) or not samples:
        raise ValueError("Sample JSON must contain a nonempty sample-id to image-path mapping")
    for sample_id, relative_path in samples.items():
        if not isinstance(sample_id, str) or Path(sample_id).name != sample_id or sample_id in (".", ".."):
            raise ValueError(f"Invalid sample ID: {sample_id!r}")
        if not isinstance(relative_path, str) or not (args.img_dir / relative_path).is_file():
            raise FileNotFoundError(f"Sample image not found: {relative_path}")
    classnames = load_classnames(args.classnames_path)
    if len(classnames) < 2:
        raise ValueError("At least two classifier class names are required")
    adapter = BLIPAttackAdapter(args, device)
    classifier = build_zero_shot_blip_classifier(adapter, classnames, args.prompt_template)
    model = build_causal_metric_model(classifier)
    blur = build_blur_substrate(args.kernel_size, args.kernel_sigma)
    root = args.output_root / args.dataset / "clean"
    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items() if key != "overwrite"
    }
    config.update(model="BLIP_ITC", target_source="pred")
    for method in args.methods:
        method_root = root / method
        method_root.mkdir(parents=True, exist_ok=True)
        config_path = method_root / "run_config.json"
        if config_path.exists() and not args.overwrite:
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous != config:
                raise ValueError(f"Clean settings differ: {config_path}; use another output root or --overwrite")
        _write_json_atomic(config_path, config)
        for sample_id, relative_path in tqdm(samples.items(), desc=f"BLIP clean {method}"):
            folder = method_root / sample_id
            required = (
                "curve_information.json", "insertion_information.json",
                "deletion_information.json", "road_deletion_information.json",
                "del_avg_channel_information.json",
                "clean_image_resized.png", f"saliency/{method}_saliency.npy",
                "ins_summary.png", "del_summary.png",
            )
            if not args.overwrite and all((folder / name).is_file() for name in required):
                continue
            folder.mkdir(parents=True, exist_ok=True)
            with Image.open(args.img_dir / relative_path) as source:
                image = source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            normalized = adapter.preprocess(image).unsqueeze(0).to(device)
            _, _, label, confidence = predict_zero_shot_clip(classifier, normalized, device)
            texts = [classnames[label]]
            with torch.no_grad():
                embedding = F.normalize(adapter.encode_text(texts), dim=-1)
            heatmap = adapter.generate_hm(method, normalized, embedding, texts, Resize((224, 224)))
            saliency = heatmap.cpu().numpy()
            image.save(folder / "clean_image_resized.png")
            save_saliency_outputs(saliency, image, str(folder / "saliency"), stem=f"{method}_saliency")
            rerun = {}
            for mode in ("ins", "del"):
                metric = CausalMetric(model, mode, args.step, blur if mode == "ins" else torch.zeros_like)
                steps_dir = folder / f"steps_{mode}"
                if args.save_process:
                    steps_dir.mkdir(exist_ok=True)
                with torch.no_grad():
                    curve = metric.single_run(
                        normalized, saliency, verbose=0,
                        save_to=str(steps_dir) if args.save_process else None,
                    )
                rerun[mode] = {"curve": curve.tolist(), "auc": float(auc(curve))}
                prefix = "insertion" if mode == "ins" else "deletion"
                _write_json_atomic(folder / f"{prefix}_information.json", {
                    f"{prefix}_curve": curve.tolist(), f"{prefix}_auc": float(auc(curve)),
                })
                save_causal_metric_summary(
                    image_tensor=normalized,
                    final_tensor=normalized if mode == "ins" else torch.zeros_like(normalized),
                    scores=curve, output_path=str(folder / f"{mode}_summary.png"),
                    mode=mode, class_name=classnames[label], preprocess=adapter.preprocess,
                )
            metrics = evaluate_both(model, to_tensor(image), saliency.squeeze(), label,
                                    device, args.road_noise, args.seed)
            for name, result in metrics.items():
                result.update(model="BLIP_ITC", target_label=label, map_source="clean",
                              checkpoint=str(args.checkpoint.resolve()), last_heads=args.last_heads)
                _write_json_atomic(folder / METRIC_FILES[name], result)
            _write_json_atomic(folder / "curve_information.json", {
                "model": "BLIP_ITC", "attack_mode": "clean",
                "pred_label": label, "pred_classname": classnames[label],
                "pred_confidence": confidence, "target_label": label,
                "image_path": str(args.img_dir / relative_path), "rerun": rerun,
            })
    logging.info("Clean BLIP results saved under %s", root)


if __name__ == "__main__":
    main()
