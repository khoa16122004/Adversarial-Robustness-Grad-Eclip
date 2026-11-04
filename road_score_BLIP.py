"""Evaluate ROAD on existing BLIP attack images and saliency maps."""

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
from road_evaluation.road.imputations import NoisyLinearImputer
from road_evaluation.road.imputed_dataset import ImputedDataset
from util import build_causal_metric_model, load_classnames, normalize_ImageNet1k


def evaluate(model, image, saliency, label, device, noise, seed):
    percentages = np.arange(1, 10, dtype=np.float64) / 10
    generator_state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    curve = []
    accuracy = []
    try:
        for percentage in percentages:
            dataset = ImputedDataset(
                base_dataset=[(image, label)],
                mask=[np.repeat(saliency[:, :, None], 3, axis=2)],
                th_p=float(percentage), remove=True,
                imputation=NoisyLinearImputer(noise=noise),
                prediction=[label], use_cache=False,
            )
            imputed, _, _ = dataset[0]
            with torch.no_grad():
                probabilities = model(normalize_ImageNet1k(imputed.unsqueeze(0).to(device)))
            if not torch.isfinite(probabilities).all():
                raise ValueError("Nonfinite BLIP probabilities during ROAD evaluation")
            curve.append(float(probabilities[0, label].item()))
            accuracy.append(float(probabilities.argmax(dim=1).item() == label))
    finally:
        np.random.set_state(generator_state)
    # Match the existing clean_score ROAD integration over percentages 0.1..0.9.
    def integrate(values):
        return sum(
            (left + right) * (x_right - x_left) / 2
            for left, right, x_left, x_right in zip(
                values, values[1:], percentages, percentages[1:]
            )
        )
    maximum = max(curve)
    return {
        "percentages": percentages.tolist(),
        "deletion_curve": curve,
        "deletion_accuracy_curve": accuracy,
        "deletion_auc": integrate(curve),
        "deletion_fair_auc": integrate(curve) / maximum if maximum > 0 else None,
        "deletion_accuracy_auc": integrate(accuracy),
        "morf": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("BLIP_result"))
    parser.add_argument("--epsilon", required=True, help="Single epsilon, e.g. 2")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset folder names")
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
    configs = sorted(root.rglob("run_config.json"))
    configs = [
        path for path in configs
        if f"eps_{epsilon}" in path.relative_to(root).parts
        and (not args.datasets or path.relative_to(root).parts[0] in args.datasets)
    ]
    if not configs:
        raise FileNotFoundError(f"No BLIP run_config.json for eps_{epsilon} under {root}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate the BLIP PyTorch environment")
    sys.path.insert(0, str(args.blip_root.resolve()))
    processed = 0
    skipped = 0
    for config_path in configs:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model") != "BLIP_ITC":
            raise ValueError(f"Not a BLIP attack configuration: {config_path}")
        mode = config["mode"]
        if mode not in ("ins", "del", "del+ins"):
            raise ValueError(f"Unsupported attack mode {mode}: {config_path}")
        config["checkpoint"] = args.checkpoint or Path(config["checkpoint"])
        config["blip_root"] = args.blip_root.resolve()
        if not config["checkpoint"].is_file():
            raise FileNotFoundError(f"Checkpoint not found: {config['checkpoint']}")
        pending = []
        for info_path in sorted(config_path.parent.glob("*/curve_information.json")):
            output = info_path.parent / "road_deletion_information.json"
            if output.exists() and not args.overwrite:
                previous = json.loads(output.read_text(encoding="utf-8"))
                if (
                    previous.get("model") == "BLIP_ITC"
                    and previous.get("road_noise") == args.noise
                    and previous.get("road_seed") == args.seed
                    and previous.get("checkpoint") == str(config["checkpoint"])
                    and previous.get("last_heads") == config["last_heads"]
                ):
                    skipped += 1
                    continue
                raise ValueError(f"Existing ROAD settings differ: {output}; use --overwrite")
            pending.append((info_path, output))
        if not pending:
            continue
        logging.info("Loading BLIP for %s; samples=%d", config_path.parent, len(pending))
        adapter = BLIPAttackAdapter(SimpleNamespace(**config), device)
        classnames = load_classnames(config.get("classnames_path"))
        classifier = build_zero_shot_blip_classifier(adapter, classnames, config["prompt_template"])
        model = build_causal_metric_model(classifier)
        for info_path, output in pending:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            prefix = "joint" if mode == "del+ins" else mode
            image_suffix = "del_ins" if mode == "del+ins" else mode
            image_path = info_path.parent / f"adversarial_image_{image_suffix}.png"
            saliency_path = info_path.parent / f"{prefix}_del_adv_{config['hm_type']}_saliency.npy"
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
            label = info["adv_pred_label"]
            if not isinstance(label, int) or not 0 <= label < len(classnames):
                raise ValueError(f"Invalid adv_pred_label: {info_path}")
            payload = evaluate(model, image, saliency, label, device, args.noise, args.seed)
            payload.update({
                "model": "BLIP_ITC", "metric": "road_deletion",
                "target_label": label, "target_source": "adv_pred_label",
                "road_noise": args.noise, "road_seed": args.seed,
                "checkpoint": str(config["checkpoint"]), "last_heads": config["last_heads"],
                "image_path_used": str(image_path), "saliency_path_used": str(saliency_path),
            })
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(output)
            processed += 1
            logging.info("Saved %s", output)
        del model, classifier, adapter
    logging.info("ROAD complete: processed=%d skipped=%d", processed, skipped)


if __name__ == "__main__":
    main()
