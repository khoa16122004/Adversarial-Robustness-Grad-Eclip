"""SigLIP interfaces for the shared classifier/explainer causal attack pipeline."""

from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from test_siglip_explainers import (
    encode_image, gradient_maps, normalize_map, rise_map,
)

HM_METHODS = {
    "eclip": "gradeclip",
    "gradeclip": "gradeclip",
    "game": "game",
    "gradcam": "gradcam",
    "maskclip": "maskclip",
    "rise": "rise",
}

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class DegenerateSigLIPHeatmapError(ValueError):
    pass


def load_model(args, device):
    try:
        from transformers import SiglipModel, SiglipProcessor
    except ImportError as exc:
        raise RuntimeError(
            "SigLIP support requires Transformers with SiglipModel and SiglipProcessor "
            "(install requirements_siglip.txt in a separate environment)."
        ) from exc
    processor = SiglipProcessor.from_pretrained(args.model)
    model = SiglipModel.from_pretrained(args.model).to(device).eval()
    return model, processor


def _configured_image_size(model):
    size = model.config.vision_config.image_size
    if isinstance(size, (list, tuple)):
        if len(size) != 2 or size[0] != size[1]:
            raise ValueError(f"Expected a square SigLIP image size, got {size}")
        return int(size[0])
    return int(size)


class SigLIPAttackAdapter(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.model, self.processor = load_model(args, device)
        self.model.requires_grad_(False)
        self.image_size = _configured_image_size(self.model)
        self.patch_size = int(self.model.config.vision_config.patch_size)
        self.options = SimpleNamespace(
            game_start_layer=args.game_start_layer,
            rise_masks=args.rise_masks,
            rise_grid=args.rise_grid,
            rise_prob=args.rise_prob,
            batch_size=args.rise_batch_size,
            seed=args.seed,
        )
        image_processor = self.processor.image_processor
        image_mean = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1)
        image_std = torch.tensor(image_processor.image_std).view(1, 3, 1, 1)
        self.register_buffer("siglip_mean", image_mean)
        self.register_buffer("siglip_std", image_std)
        self.preprocess = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
        self.to(device)
        self.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to_siglip_pixels(self, clip_normalized_images):
        mean = clip_normalized_images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = clip_normalized_images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        pixels = (clip_normalized_images * std + mean).clamp(0, 1)
        return (pixels - self.siglip_mean) / self.siglip_std

    def encode_image(self, clip_normalized_images):
        pixels = self.to_siglip_pixels(clip_normalized_images)
        features, _, _ = encode_image(self.model, pixels)
        return features

    def encode_text(self, texts):
        tokens = self.processor(
            text=texts, padding="max_length", return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        return self.model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask,
        )

    def generate_hm(self, hm_type, image, text_embedding, texts, resize):
        if hm_type not in HM_METHODS:
            raise ValueError(f"Unsupported SigLIP explainer: {hm_type}")
        if image.shape != (1, 3, self.image_size, self.image_size):
            raise ValueError(
                f"Expected one normalized SigLIP image, got {tuple(image.shape)}",
            )
        if text_embedding.ndim != 2 or text_embedding.shape[0] != 1:
            raise ValueError("SigLIP attack explanations require one target text embedding")
        method = HM_METHODS[hm_type]
        options = SimpleNamespace(
            **vars(self.options), methods=[method],
        )
        diagnostics = {}
        with torch.enable_grad():
            image = image.detach().to(self.device).requires_grad_(True)
            pixels = self.to_siglip_pixels(image)
            text_embedding = text_embedding.detach().to(self.device)
            if method == "rise":
                saliency = rise_map(self.model, pixels, text_embedding, options)
            else:
                maps, _ = gradient_maps(
                    self.model, pixels, text_embedding, options, diagnostics=diagnostics,
                )
                saliency = maps[method]
        saliency = saliency.detach().float()
        if method == "gradeclip" and torch.isfinite(saliency).all():
            if saliency.max().item() == saliency.min().item():
                raise DegenerateSigLIPHeatmapError(
                    f"SigLIP {hm_type} produced a constant heatmap for target {texts!r}. "
                    "A constant map gives an arbitrary pixel ranking. "
                    f"Diagnostics: {diagnostics}",
                )
        saliency = normalize_map(saliency)
        saliency = resize(saliency.unsqueeze(0))[0].clamp(0, 1)
        if not torch.isfinite(saliency).all():
            raise ValueError(f"{hm_type}: nonfinite resized SigLIP heatmap")
        return saliency.detach()


class ZeroShotSigLIPClassifier(nn.Module):
    def __init__(self, adapter, weights):
        super().__init__()
        self.adapter = adapter
        self.register_buffer("zero_shot_weights", weights.detach())

    @property
    def logit_scale(self):
        return float(self.adapter.model.logit_scale.detach().exp().item())

    def forward(self, images):
        image_features = F.normalize(self.adapter.encode_image(images), dim=-1)
        text_weights = F.normalize(self.zero_shot_weights, dim=0)
        return (
            self.adapter.model.logit_scale.exp() * (image_features @ text_weights)
            + self.adapter.model.logit_bias
        )


def build_zero_shot_siglip_classifier(
    adapter, classnames, prompt_template, batch_size=10,
):
    from tqdm import tqdm

    weights = []
    with torch.no_grad():
        for start in tqdm(range(0, len(classnames), batch_size), desc="SigLIP text classifier"):
            texts = [
                prompt_template.format(name)
                for name in classnames[start:start + batch_size]
            ]
            weights.append(F.normalize(adapter.encode_text(texts), dim=-1))
    classifier = ZeroShotSigLIPClassifier(adapter, torch.cat(weights, dim=0).T.contiguous()).eval()
    return classifier


def generate_hm(raw_model, hm_type, image, text_embedding, texts, resize, preprocess):
    """Keep the shared RISE attack callback signature without model-specific branching."""
    return raw_model.generate_hm(hm_type, image, text_embedding, texts, resize)
