"""Evaluate ROAD and DEL_avg_channel from recorded SigLIP attack configurations."""

import argparse
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from road_evaluation.deletion_metrics import METRIC_FILES, evaluate_both
from siglip_attack_adapter import SigLIPAttackAdapter, build_zero_shot_siglip_classifier
from util import build_causal_metric_model, load_classnames


def attack_artifacts(sample_dir, mode, method):
    if mode not in ("del", "ins", "del+ins"):
        raise ValueError(f"Unsupported SigLIP attack mode: {mode}")
    suffix = "del_ins" if mode == "del+ins" else mode
    prefix = "joint" if mode == "del+ins" else mode
    return (
        sample_dir / f"adversarial_image_{suffix}.png",
        sample_dir / f"{prefix}_del_adv_{method}_saliency.npy",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Root containing SigLIP attack run_config.json files")
    parser.add_argument("--modes", nargs="+", choices=("del", "ins", "del+ins"),
                        default=["del", "ins", "del+ins"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.noise) or args.noise < 0 or args.seed < 0:
        parser.error("--noise must be finite and nonnegative; --seed must be nonnegative")
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"SigLIP result root not found: {args.input_root}")
    configs = sorted(args.input_root.rglob("run_config.json"))
    if not configs:
        raise FileNotFoundError(f"No SigLIP run_config.json under {args.input_root}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or the SigLIP PyTorch environment")
    processed, skipped = 0, 0
    selected = False
    for config_path in configs:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_type") != "SigLIP":
            raise ValueError(f"Not a SigLIP configuration: {config_path}")
        if config["mode"] not in args.modes:
            continue
        selected = True
        info_paths = sorted(config_path.parent.glob("*/curve_information.json"))
        if not info_paths:
            logging.warning("No completed attack samples under %s", config_path.parent)
            continue
        pending = []
        for info_path in info_paths:
            complete = True
            for name in ("road_deletion", "DEL_avg_channel"):
                output = info_path.parent / METRIC_FILES[name]
                if args.overwrite or not output.is_file():
                    complete = False
                    continue
                previous = json.loads(output.read_text(encoding="utf-8"))
                if (
                    previous.get("metric") != name or previous.get("seed") != args.seed
                    or previous.get("road_noise") != args.noise
                    or previous.get("model") != config["model"]
                    or previous.get("explainer_version") != config["explainer_version"]
                    or previous.get("prompt_template") != config["prompt_template"]
                    or previous.get("classnames_path") != config.get("classnames_path")
                ):
                    raise ValueError(f"Existing deletion evaluation settings differ: {output}; use --overwrite")
            if complete:
                skipped += 1
            else:
                pending.append(info_path)
        if not pending:
            continue
        adapter = SigLIPAttackAdapter(SimpleNamespace(**config), device)
        if adapter.image_size != config["image_size"]:
            raise ValueError(f"SigLIP image size differs from recorded configuration: {config_path}")
        classnames = load_classnames(config.get("classnames_path"))
        classifier = build_zero_shot_siglip_classifier(adapter, classnames, config["prompt_template"])
        model = build_causal_metric_model(classifier)
        for info_path in pending:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            label = info["adv_pred_label"]
            if not isinstance(label, int) or not 0 <= label < len(classnames):
                raise ValueError(f"Invalid adv_pred_label: {info_path}")
            image_path, saliency_path = attack_artifacts(
                info_path.parent, config["mode"], config["hm_type"],
            )
            with Image.open(image_path) as source:
                image = to_tensor(source.convert("RGB"))
            if tuple(image.shape) != (3, adapter.image_size, adapter.image_size):
                raise ValueError(f"Invalid SigLIP image dimensions: {image_path}")
            saliency = np.load(saliency_path, allow_pickle=False).squeeze()
            results = evaluate_both(model, image, saliency, label, device, args.noise, args.seed)
            for name, result in results.items():
                result.update({
                    "model_type": "SigLIP", "model": config["model"],
                    "explainer_version": config["explainer_version"],
                    "prompt_template": config["prompt_template"],
                    "classnames_path": config.get("classnames_path"),
                    "attack_mode": config["mode"], "map_source": "post_attack",
                    "target_label": label, "target_source": "adv_pred_label",
                    "road_noise": args.noise, "road_seed": args.seed,
                    "image_path_used": str(image_path), "saliency_path_used": str(saliency_path),
                })
                output = info_path.parent / METRIC_FILES[name]
                temporary = output.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
                temporary.replace(output)
                logging.info("Saved %s", output)
            processed += 1
        del model, classifier, adapter
    if not selected:
        raise FileNotFoundError(f"No SigLIP configurations for requested modes: {args.modes}")
    logging.info("Deletion evaluation complete: processed=%d skipped=%d", processed, skipped)


if __name__ == "__main__":
    main()
