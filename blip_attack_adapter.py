"""BLIP interfaces for the existing classifier/explainer causal attack pipeline."""

from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from test_blip_explainers import encode_image, gradient_maps, load_model, normalize_map, rise_map

HM_METHODS = {
    "eclip": "gradeclip",
    "gradeclip": "gradeclip",
    "game": "game",
    "gradcam": "gradcam",
    "maskclip": "maskclip",
    "rise": "rise",
}


class BLIPAttackAdapter(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.model = load_model(args, device)
        self.model.requires_grad_(False)
        self.options = SimpleNamespace(
            image_size=args.image_size,
            last_heads=args.last_heads,
            game_start_layer=args.game_start_layer,
            rise_masks=args.rise_masks,
            rise_grid=args.rise_grid,
            rise_prob=args.rise_prob,
            batch_size=args.rise_batch_size,
            seed=args.seed,
        )
        self.preprocess = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def encode_image(self, images):
        features, _, _ = encode_image(self.model, images, self.options.last_heads)
        return features

    def encode_text(self, texts):
        tokens = self.model.tokenizer(
            texts, padding="max_length", truncation=True, max_length=35,
            return_tensors="pt",
        ).to(self.device)
        output = self.model.text_encoder(
            tokens.input_ids, attention_mask=tokens.attention_mask,
            return_dict=True, mode="text",
        )
        return self.model.text_proj(output.last_hidden_state[:, 0])

    def generate_hm(self, hm_type, image, text_embedding, texts, resize):
        if hm_type not in HM_METHODS:
            raise ValueError(f"Unsupported BLIP explainer: {hm_type}")
        if image.shape != (1, 3, self.options.image_size, self.options.image_size):
            raise ValueError(f"Expected one normalized BLIP image, got {tuple(image.shape)}")
        if text_embedding.ndim != 2 or text_embedding.shape[0] != 1:
            raise ValueError("BLIP attack explanations require one target text embedding")
        method = HM_METHODS[hm_type]
        options = SimpleNamespace(**vars(self.options), methods=[method])
        # Ranking is fixed within each PGD step. Do not consume the attack's
        # graph when autograd computes gradients for the explanation itself.
        with torch.enable_grad():
            image = image.detach().to(self.device).requires_grad_(True)
            text_embedding = text_embedding.detach().to(self.device)
            if method == "rise":
                saliency = rise_map(self.model, image, text_embedding, options)
            else:
                maps, _ = gradient_maps(self.model, image, text_embedding, options)
                saliency = maps[method]
        saliency = normalize_map(saliency.detach().float())
        saliency = resize(saliency.unsqueeze(0))[0].clamp(0, 1)
        if not torch.isfinite(saliency).all():
            raise ValueError(f"{hm_type}: nonfinite resized BLIP heatmap")
        return saliency.detach()


class ZeroShotBLIPClassifier(nn.Module):
    def __init__(self, adapter, weights, logit_scale=100.0):
        super().__init__()
        self.adapter = adapter
        self.register_buffer("zero_shot_weights", weights.detach())
        self.logit_scale = logit_scale

    def forward(self, images):
        features = F.normalize(self.adapter.encode_image(images), dim=-1)
        return self.logit_scale * features @ self.zero_shot_weights


def build_zero_shot_blip_classifier(adapter, classnames, prompt_template, batch_size=10):
    from tqdm import tqdm

    weights = []
    with torch.no_grad():
        for start in tqdm(range(0, len(classnames), batch_size), desc="BLIP text classifier"):
            texts = [prompt_template.format(name) for name in classnames[start:start + batch_size]]
            weights.append(F.normalize(adapter.encode_text(texts), dim=-1))
    classifier = ZeroShotBLIPClassifier(adapter, torch.cat(weights).T.contiguous()).eval()
    return classifier


def generate_hm(raw_model, hm_type, image, text_embedding, texts, resize, preprocess):
    """Keep the existing RISE attack callback signature without CLIP imports."""
    return raw_model.generate_hm(hm_type, image, text_embedding, texts, resize)
