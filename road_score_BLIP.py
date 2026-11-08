"""Evaluate ROAD and DEL_avg_channel on existing BLIP attack artifacts."""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from blip_attack_adapter import BLIPAttackAdapter, build_zero_shot_blip_classifier
from road_evaluation.deletion_metrics import METRIC_FILES, evaluate_both, evaluate_deletion
from road_evaluation.road.imputations import NoisyLinearImputer
from util import build_causal_metric_model, load_classnames


def attack_artifacts(sample_dir, mode, method):
    if mode == "xshift":
        image_name = "adversarial_image_xshift.png"
        saliency_name = f"del_xshift_{method}_saliency.npy"
    elif mode in ("ins", "del", "del+ins"):
        prefix = "joint" if mode == "del+ins" else mode
        image_suffix = "del_ins" if mode == "del+ins" else mode
        image_name = f"adversarial_image_{image_suffix}.png"
        saliency_name = f"{prefix}_del_adv_{method}_saliency.npy"
    else:
        raise ValueError(f"Unsupported attack mode: {mode}")
    return sample_dir / image_name, sample_dir / saliency_name


def evaluate(model, image, saliency, label, device, noise, seed):
    return evaluate_deletion(
        model, image, saliency, label, device, NoisyLinearImputer(noise=noise), seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("BLIP_result"))
    parser.add_argument("--epsilon", required=True, help="Single epsilon, e.g. 2")
    parser.add_argument("--clean-only", action="store_true",
                        help="Evaluate saved clean images/maps under dataset/clean/explainer/sample")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset folder names")
    parser.add_argument(
        "--modes", nargs="+", choices=("ins", "del", "del+ins", "xshift"),
        default=["ins", "del", "del+ins", "xshift"],
        help="Evaluate saved post-attack maps for these attack modes",
    )
    parser.add_argument("--blip-root", type=Path, default=Path("BLIP"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override recorded checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.noise) or args.noise < 0 or args.seed < 0:
        parser.error("--noise must be finite and nonnegative; --seed must be nonnegative")
    epsilon = args.epsilon.removeprefix("eps_")
    if not epsilon.isdigit():
        parser.error("--epsilon must be a nonnegative integer")
    root = args.input_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {root}")
    configs = sorted(
        path for dataset in root.iterdir()
        if dataset.is_dir() and (not args.datasets or dataset.name in args.datasets)
        for path in (dataset / ("clean" if args.clean_only else f"eps_{epsilon}")).glob("**/run_config.json")
    )
    if not configs:
        raise FileNotFoundError(f"No BLIP run_config.json for eps_{epsilon} under {root}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate the BLIP PyTorch environment")
    sys.path.insert(0, str(args.blip_root.resolve()))
    processed = 0
    skipped = 0
    discovered_modes = set()
    for config_path in configs:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model") != "BLIP_ITC":
            raise ValueError(f"Not a BLIP attack configuration: {config_path}")
        mode = "clean" if args.clean_only else config["mode"]
        if mode not in ("clean", "ins", "del", "del+ins", "xshift"):
            raise ValueError(f"Unsupported attack mode {mode}: {config_path}")
        if mode != "clean" and mode not in args.modes:
            continue
        discovered_modes.add(mode)
        method = config_path.parent.name if args.clean_only else config["hm_type"]
        config["checkpoint"] = args.checkpoint or Path(config["checkpoint"])
        config["blip_root"] = args.blip_root.resolve()
        if not config["checkpoint"].is_file():
            raise FileNotFoundError(f"Checkpoint not found: {config['checkpoint']}")
        pending = []
        sample_infos = sorted(config_path.parent.glob("*/curve_information.json"))
        if not sample_infos:
            raise FileNotFoundError(f"No completed attack sample JSON under {config_path.parent}")
        logging.info("ROAD group: %s | mode=%s | samples=%d",
                     config_path.parent, mode, len(sample_infos))
        for info_path in sample_infos:
            complete = True
            for name in ("road_deletion", "DEL_avg_channel"):
                output = info_path.parent / METRIC_FILES[name]
                if args.overwrite or not output.exists():
                    complete = False
                    continue
                previous = json.loads(output.read_text(encoding="utf-8"))
                if (
                    previous.get("model") == "BLIP_ITC"
                    and previous.get("road_noise", previous.get("linear_noise") if name == "road_deletion" else args.noise) == args.noise
                    and previous.get("road_seed", previous.get("seed")) == args.seed
                    and previous.get("checkpoint") == str(config["checkpoint"])
                    and previous.get("last_heads") == config["last_heads"]
                ):
                    if previous.get("metric") != name or previous.get("seed") != args.seed:
                        complete = False
                    continue
                raise ValueError(f"Existing ROAD settings differ: {output}; use --overwrite")
            if complete:
                skipped += 1
                continue
            pending.append(info_path)
        if not pending:
            continue
        logging.info("Loading BLIP for %s; samples=%d", config_path.parent, len(pending))
        adapter = BLIPAttackAdapter(SimpleNamespace(**config), device)
        classnames = load_classnames(config.get("classnames_path"))
        classifier = build_zero_shot_blip_classifier(adapter, classnames, config["prompt_template"])
        model = build_causal_metric_model(classifier)
        for info_path in pending:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if args.clean_only:
                image_path = info_path.parent / "clean_image_resized.png"
                saliency_path = info_path.parent / "saliency" / f"{method}_saliency.npy"
            else:
                image_path, saliency_path = attack_artifacts(info_path.parent, mode, method)
            with Image.open(image_path) as source:
                image = to_tensor(source.convert("RGB"))
            saliency = np.load(saliency_path, allow_pickle=False).squeeze()
            if (
                tuple(image.shape) != (3, config["image_size"], config["image_size"])
                or saliency.shape != tuple(image.shape[-2:])
                or not np.isfinite(saliency).all()
            ):
                raise ValueError(f"Invalid image/saliency shape or values: {info_path.parent}")
            # Match CausalMetric reruns: score the unmasked adversarial predicted class.
            label = info["pred_label"] if args.clean_only else info["adv_pred_label"]
            if not isinstance(label, int) or not 0 <= label < len(classnames):
                raise ValueError(f"Invalid adv_pred_label: {info_path}")
            results = evaluate_both(model, image, saliency, label, device, args.noise, args.seed)
            metadata = {
                "model": "BLIP_ITC",
                "attack_mode": mode, "map_source": "clean" if args.clean_only else "post_attack",
                "target_label": label, "target_source": "pred_label" if args.clean_only else "adv_pred_label",
                "road_noise": args.noise, "road_seed": args.seed,
                "checkpoint": str(config["checkpoint"]), "last_heads": config["last_heads"],
                "image_path_used": str(image_path), "saliency_path_used": str(saliency_path),
            }
            for name, payload in results.items():
                payload.update(metadata)
                output = info_path.parent / METRIC_FILES[name]
                temporary = output.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
                temporary.replace(output)
                logging.info("Saved %s", output)
            processed += 1
        del model, classifier, adapter
    for mode in (["clean"] if args.clean_only else args.modes):
        if mode not in discovered_modes:
            logging.warning("No attack configurations for mode=%s at eps_%s; not evaluated",
                            mode, epsilon)
    if not discovered_modes:
        raise FileNotFoundError(f"No configurations for requested modes: {args.modes}")
    logging.info("ROAD complete: processed=%d skipped=%d", processed, skipped)


if __name__ == "__main__":
    main()
