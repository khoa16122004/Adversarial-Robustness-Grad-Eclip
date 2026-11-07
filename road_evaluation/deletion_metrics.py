"""Shared ROAD and channel-mean deletion evaluation for CLIP-shaped adapters."""

import math

import numpy as np
import torch

from .road.imputations import ChannelMeanImputer, NoisyLinearImputer, ZeroImputer
from .road.imputed_dataset import ImputedDataset
from util import normalize_ImageNet1k


DEFAULT_PERCENTAGES = [i / 10 for i in range(1, 10)]
METRIC_FILES = {
    "road_deletion": "road_deletion_information.json",
    "DEL_avg_channel": "del_avg_channel_information.json",
    "DEL_zero": "del_zero_information.json",
}


def metric_name(imputer):
    if isinstance(imputer, NoisyLinearImputer):
        return "road_deletion"
    if isinstance(imputer, ChannelMeanImputer):
        return "DEL_avg_channel"
    if isinstance(imputer, ZeroImputer):
        return "DEL_zero"
    raise ValueError(f"Unsupported deletion imputer: {type(imputer).__name__}")


def choose_imputer(kind, noise):
    if kind == "linear":
        return NoisyLinearImputer(noise=noise)
    if kind == "fixed":
        return ChannelMeanImputer()
    if kind == "zero":
        return ZeroImputer()
    raise ValueError(f"Unknown imputer: {kind}")


def evaluate_deletion(model, image, saliency, label, device, imputer,
                      percentages=DEFAULT_PERCENTAGES, seed=42, accuracy_label=None):
    """Use one nested ranking, an untouched image per state, and isolated RNGs."""
    name = metric_name(imputer)
    x = np.asarray(percentages, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Deletion percentages must be at least two finite, strictly increasing values.")
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("Deletion percentages must be in [0, 1].")
    if name == "road_deletion" and np.any(x == 1):
        raise ValueError("ROAD linear imputation requires remaining pixels; percentages must be below 1.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Deletion seed must be a nonnegative integer.")
    if name == "road_deletion" and (not math.isfinite(imputer.noise) or imputer.noise < 0):
        raise ValueError("ROAD noise must be finite and nonnegative.")
    image = image.detach().cpu()
    if image.ndim != 3 or image.shape[0] != 3 or not torch.isfinite(image).all():
        raise ValueError("Deletion evaluation expects a finite RGB tensor with shape (3, H, W).")
    saliency = np.asarray(saliency, dtype=np.float32)
    if saliency.ndim == 3 and saliency.shape[0] in (1, 3) and saliency.shape[1:] == tuple(image.shape[-2:]):
        saliency = saliency.transpose(1, 2, 0)
    if saliency.ndim == 2:
        saliency = saliency[:, :, None]
    if saliency.ndim != 3 or saliency.shape[:2] != tuple(image.shape[-2:]) or not np.isfinite(saliency).all():
        raise ValueError("Deletion saliency must be finite and match the image dimensions.")
    accuracy_label = label if accuracy_label is None else accuracy_label
    model.eval()
    model.to(device)
    curve, accuracy = [], []
    # ImputedDataset adds noise to break attribution ties. Reuse it at every percentage
    # and in both metrics so only the imputation method differs.
    tie_noise = 1e-4 * np.random.default_rng(seed).standard_normal(image.shape[-2:])
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            for percentage in x:
                if int(image.shape[-2] * image.shape[-1] * percentage) == 0:
                    imputed = image.clone()
                else:
                    dataset = ImputedDataset(
                        base_dataset=[(image.clone(), accuracy_label)],
                        mask=[saliency], th_p=float(percentage), remove=True,
                        imputation=imputer, prediction=[label], use_cache=False,
                    )
                    dataset.random_v = tie_noise.copy()
                    imputed, _, _ = dataset[0]
                if not torch.isfinite(imputed).all():
                    raise ValueError(f"Nonfinite imputed pixels during {name} evaluation.")
                with torch.no_grad():
                    probabilities = model(normalize_ImageNet1k(imputed.unsqueeze(0).to(device)))
                if probabilities.ndim != 2 or probabilities.shape[0] != 1 or not torch.isfinite(probabilities).all():
                    raise ValueError(f"Invalid classifier probabilities during {name} evaluation.")
                if not isinstance(label, int) or not 0 <= label < probabilities.shape[1]:
                    raise ValueError(f"Invalid deletion target class: {label}")
                curve.append(float(probabilities[0, label].item()))
                accuracy.append(float(probabilities.argmax(dim=1).item() == accuracy_label))
    finally:
        np.random.set_state(numpy_state)

    def integrate(values):
        return float(sum(
            (left + right) * (x_right - x_left) / 2
            for left, right, x_left, x_right in zip(values, values[1:], x, x[1:])
        ))

    area = integrate(curve)
    maximum = max(curve)
    return {
        "metric": name,
        "imputer": {"road_deletion": "linear", "DEL_avg_channel": "fixed", "DEL_zero": "zero"}[name],
        "seed": seed,
        "linear_noise": imputer.noise if name == "road_deletion" else None,
        "percentages": x.tolist(),
        "deletion_curve": curve,
        "deletion_accuracy_curve": accuracy,
        "deletion_auc": area,
        "deletion_fair_auc": area / maximum if maximum > 0 else None,
        "deletion_accuracy_auc": integrate(accuracy),
        "morf": True,
    }


def evaluate_both(model, image, saliency, label, device, noise=0.01, seed=42,
                  percentages=DEFAULT_PERCENTAGES, accuracy_label=None):
    return {
        metric_name(imputer): evaluate_deletion(
            model, image, saliency, label, device, imputer, percentages, seed, accuracy_label,
        )
        for imputer in (NoisyLinearImputer(noise=noise), ChannelMeanImputer())
    }
