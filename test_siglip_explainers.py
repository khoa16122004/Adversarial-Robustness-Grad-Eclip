"""SigLIP explainers and a single-image explainer smoke runner."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

METHODS = ("gradeclip", "game", "gradcam", "maskclip", "rise")
LOG = logging.getLogger("siglip-explainers")
REPO_ROOT = Path(__file__).resolve().parent
EXPLAINER_VERSION = "siglip-pooling-v2"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", required=True, help="One target caption per run.")
    parser.add_argument("--model", default="google/siglip-base-patch16-224",
                        help="Hugging Face model ID or local SigLIP model directory.")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "siglip_explainer_results")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--game-start-layer", type=int, default=-1,
                        help="-1: final block only; 0: all vision blocks.")
    parser.add_argument("--rise-masks", type=int, default=2000)
    parser.add_argument("--rise-grid", type=int, default=8)
    parser.add_argument("--rise-prob", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=50, help="RISE batch size.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"--image: file does not exist: {args.image}")
    if not args.text.strip():
        parser.error("--text cannot be empty")
    if min(args.rise_masks, args.rise_grid, args.batch_size) <= 0:
        parser.error("--rise-masks, --rise-grid and --batch-size must be positive")
    if not 0 < args.rise_prob < 1:
        parser.error("--rise-prob must be between 0 and 1 (exclusive)")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    args.methods = list(dict.fromkeys(args.methods))
    return args


def normalize_map(values):
    import torch

    if not torch.isfinite(values).all():
        raise ValueError("Explainer returned NaN or infinity")
    low, high = values.amin(), values.amax()
    if (high - low).item() == 0:
        LOG.warning("Constant heatmap (value=%g); normalized map is zero.", low.item())
    return (values - low) / (high - low).clamp_min(torch.finfo(values.dtype).eps)


def _pool_image(head, hidden):
    """Expose the learned probe's attention without changing SigLIP pooling."""
    import torch
    import torch.nn.functional as F

    attention = head.attention
    batch, count, width = hidden.shape
    probe = head.probe.expand(batch, -1, -1)
    weights = attention.in_proj_weight.chunk(3)
    biases = (
        attention.in_proj_bias.chunk(3)
        if attention.in_proj_bias is not None else (None, None, None)
    )
    heads = attention.num_heads
    head_dim = width // heads
    query = F.linear(probe, weights[0], biases[0])
    key = F.linear(hidden, weights[1], biases[1])
    value = F.linear(hidden, weights[2], biases[2])
    query = query.reshape(batch, 1, heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, count, heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, count, heads, head_dim).transpose(1, 2)
    probabilities = ((query * head_dim ** -0.5) @ key.transpose(-2, -1)).softmax(-1)
    attended = (probabilities @ value).transpose(1, 2).reshape(batch, 1, width)
    pooled = attention.out_proj(attended)
    features = (pooled + head.mlp(head.layernorm(pooled)))[:, 0]
    return features, {
        "q": query, "k": key, "v": value,
        "attention": probabilities, "attended": attended,
    }


def encode_image(model, pixel_values, game_start=None):
    """Run the native vision computation, exposing backbone and pooling attention."""
    import torch
    import torch.nn.functional as F

    vision = model.vision_model
    hidden = vision.embeddings(pixel_values)
    cache = {}
    attentions = []
    layers = vision.encoder.layers
    for index, layer in enumerate(layers):
        final = index == len(layers) - 1
        capture_game = game_start is not None and index >= game_start
        residual = hidden
        normalized = layer.layer_norm1(hidden)
        batch, token_count, width = normalized.shape
        attention = layer.self_attn
        heads = attention.num_heads
        head_dim = width // heads

        def project(projection):
            return projection(normalized).view(
                batch, token_count, heads, head_dim,
            ).transpose(1, 2)

        query, key, value = (
            project(projection)
            for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
        )
        weights = F.softmax(
            (query @ key.transpose(-2, -1)) * head_dim ** -0.5,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        attended = (weights @ value).transpose(1, 2).reshape(batch, token_count, width)
        hidden = residual + attention.out_proj(attended)
        hidden = hidden + layer.mlp(layer.layer_norm2(hidden))
        if capture_game:
            attentions.append(weights)
        if final:
            cache = {
                "input": residual,
                "attended": attended,
                "q": query,
                "k": key,
                "v": value,
            }

    hidden = vision.post_layernorm(hidden)
    features, pooling = _pool_image(vision.head, hidden)
    cache["pooling"] = pooling
    return features, cache, attentions


def image_text_logits(model, image_features, text_features):
    import torch.nn.functional as F

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    return model.logit_scale.exp() * (image_features @ text_features.T) + model.logit_bias


def gradient_maps(model, pixel_values, text_feature, args, *, diagnostics=None):
    import torch
    import torch.nn.functional as F

    layers = model.vision_model.encoder.layers
    start = len(layers) - 1 if args.game_start_layer == -1 else args.game_start_layer
    if not 0 <= start < len(layers):
        raise ValueError(f"game_start_layer must be in [0, {len(layers) - 1}] or -1")
    features, cache, attentions = encode_image(
        model, pixel_values, start if "game" in args.methods else None,
    )
    # Match the BLIP/CLIP gradient explainers' cosine objective, without sigmoid saturation.
    score = (F.normalize(features, dim=-1) @ F.normalize(text_feature, dim=-1).T).reshape(())
    pooling = cache["pooling"]
    patch_count = cache["input"].shape[1]
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise ValueError(f"Expected a square SigLIP patch grid, got {patch_count} patches")
    maps = {}

    if "gradeclip" in args.methods:
        gradient = torch.autograd.grad(score, pooling["attended"], retain_graph=True)[0][0, 0]
        query, key, value = (
            pooling[name].detach().transpose(1, 2).flatten(2)[0]
            for name in ("q", "k", "v")
        )
        similarity = (F.normalize(query, dim=-1) * F.normalize(key, dim=-1)).sum(-1)
        locality = normalize_map(similarity)
        contributions = (gradient * value * locality[:, None]).sum(-1)
        maps["gradeclip"] = contributions.relu()
        if maps["gradeclip"].max().item() == maps["gradeclip"].min().item():
            diagnostic = (
                "Degenerate SigLIP gradeclip: cosine=%.6g, gradient_norm=%.6g, "
                "query_norm=%.6g, key_norm=%.6g, value_norm=%.6g, "
                "qk_range=[%.6g, %.6g], contribution_range=[%.6g, %.6g], "
                "positive_patches=%d/%d. No attribution fallback was applied."
            ) % (
                score.detach().item(), gradient.norm().item(),
                query.norm().item(), key.norm().item(), value.norm().item(),
                similarity.min().item(), similarity.max().item(),
                contributions.min().item(), contributions.max().item(),
                (contributions > 0).sum().item(), contributions.numel(),
            )
            LOG.warning("%s", diagnostic)
            if diagnostics is not None:
                diagnostics["gradeclip"] = diagnostic

    if "gradcam" in args.methods:
        layer_input = cache["input"]
        gradient = torch.autograd.grad(score, layer_input, retain_graph=True)[0]
        channel_weights = gradient.mean(dim=1, keepdim=True)
        maps["gradcam"] = (channel_weights * layer_input).sum(-1)[0].relu()

    if "game" in args.methods:
        gradients = torch.autograd.grad(
            score, [*attentions, pooling["attention"]], retain_graph=True,
        )
        token_count = attentions[0].shape[-1]
        relevance = torch.eye(token_count, device=pixel_values.device).unsqueeze(0)
        for attention, gradient in zip(attentions, gradients[:-1]):
            cam = (attention.detach() * gradient.detach()).clamp_min(0).mean(dim=1)
            relevance = relevance + cam.bmm(relevance)
        probe_relevance = (
            pooling["attention"].detach() * gradients[-1].detach()
        ).clamp_min(0).mean(dim=1)
        maps["game"] = probe_relevance.bmm(relevance)[0, 0]

    if "maskclip" in args.methods:
        with torch.no_grad():
            query, key, value = (
                pooling[name].detach().transpose(1, 2).flatten(2)
                for name in ("q", "k", "v")
            )
            locality = (
                F.normalize(query, dim=-1) * F.normalize(key, dim=-1)
            ).sum(-1)[0]
            head = model.vision_model.head
            dense = head.attention.out_proj(value)
            dense_features = F.normalize(dense + head.mlp(head.layernorm(dense)), dim=-1)
            text = F.normalize(text_feature, dim=-1)
            maps["maskclip"] = (dense_features[0] @ text[0]) * locality

    return {
        name: value.detach().reshape(side, side)
        for name, value in maps.items()
    }, float(image_text_logits(model, features.detach(), text_feature.detach()).item())


def rise_map(model, pixel_values, text_feature, args):
    import torch
    import torch.nn.functional as F

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    height, width = pixel_values.shape[-2:]
    grid = args.rise_grid
    cell_y, cell_x = math.ceil(height / grid), math.ceil(width / grid)
    up_height, up_width = (grid + 1) * cell_y, (grid + 1) * cell_x
    saliency = torch.zeros((height, width), device=pixel_values.device)
    with torch.no_grad():
        for offset in range(0, args.rise_masks, args.batch_size):
            count = min(args.batch_size, args.rise_masks - offset)
            masks = []
            for _ in range(count):
                coarse = (
                    torch.rand((1, 1, grid, grid), generator=generator) < args.rise_prob
                ).float()
                upsampled = F.interpolate(
                    coarse, size=(up_height, up_width), mode="bilinear",
                    align_corners=False,
                )
                y, x = torch.randint(cell_y, (1,), generator=generator).item(), \
                    torch.randint(cell_x, (1,), generator=generator).item()
                masks.append(upsampled[:, :, y:y + height, x:x + width])
            masks = torch.cat(masks).to(pixel_values.device)
            features, _, _ = encode_image(model, pixel_values * masks)
            scores = image_text_logits(model, features, text_feature).sigmoid().reshape(-1, 1, 1)
            saliency += (scores * masks[:, 0]).sum(dim=0)
            LOG.info("RISE: %d/%d masks", offset + count, args.rise_masks)
    return saliency / (args.rise_masks * args.rise_prob)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import SiglipModel, SiglipProcessor
    from util import save_saliency_outputs

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device,
    )
    processor = SiglipProcessor.from_pretrained(args.model)
    model = SiglipModel.from_pretrained(args.model).to(device).eval()
    model.requires_grad_(False)
    with Image.open(args.image) as source:
        raw_image = source.convert("RGB")
    inputs = processor(images=raw_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    text_inputs = processor(
        text=[args.text], padding="max_length", return_tensors="pt",
    )
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        text_feature = F.normalize(
            model.get_text_features(input_ids=input_ids, attention_mask=attention_mask),
            dim=-1,
        )
        encoded_features, _, _ = encode_image(model, pixel_values)
        reference_features = model.get_image_features(pixel_values=pixel_values)
        torch.testing.assert_close(encoded_features, reference_features, rtol=1e-4, atol=1e-5)
        reference_score = image_text_logits(model, reference_features, text_feature).item()

    depth = len(model.vision_model.encoder.layers)
    start = depth - 1 if args.game_start_layer == -1 else args.game_start_layer
    if not 0 <= start < depth:
        raise ValueError(f"--game-start-layer must be -1 or in [0, {depth - 1}]")
    image_size = pixel_values.shape[-1]
    patch_size = model.config.vision_config.patch_size
    if image_size % patch_size:
        raise ValueError(f"Processor image size {image_size} is not divisible by patch size {patch_size}")
    args.game_start_layer = start
    started = time.perf_counter()
    maps = {}
    if any(name != "rise" for name in args.methods):
        maps, explained_score = gradient_maps(
            model, pixel_values.detach().requires_grad_(True), text_feature, args,
        )
        if not math.isclose(explained_score, reference_score, rel_tol=1e-4, abs_tol=1e-5):
            raise RuntimeError("Explainer forward does not match the SigLIP image-text logit")
    if "rise" in args.methods:
        maps["rise"] = rise_map(model, pixel_values, text_feature, args)

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "image": str(args.image.resolve()),
        "text": args.text,
        "model": args.model,
        "device": str(device),
        "methods": args.methods,
        "explainer_version": EXPLAINER_VERSION,
        "gradient_target": "image-text cosine similarity (same as BLIP/CLIP)",
        "rise_target": "sigmoid of the native SigLIP image-text logit",
        "score_target": "SigLIP image-text logit (scaled cosine similarity plus logit bias)",
        "score": reference_score,
        "game_start_layer": args.game_start_layer,
        "seed": args.seed,
        "rise": {
            "masks": args.rise_masks,
            "grid": args.rise_grid,
            "probability": args.rise_prob,
            "batch_size": args.batch_size,
            "baseline": "zero in SigLIP-normalized pixel space",
        },
        "torch_version": torch.__version__,
        "maps": {},
    }
    for method in args.methods:
        raw_map = maps[method].float().cpu()
        if raw_map.ndim != 2 or not torch.isfinite(raw_map).all():
            raise ValueError(f"{method}: expected a finite 2D heatmap")
        heatmap = F.interpolate(
            normalize_map(raw_map)[None, None],
            size=(raw_image.height, raw_image.width),
            mode="bilinear", align_corners=False,
        )[0, 0].clamp(0, 1)
        save_saliency_outputs(
            heatmap.numpy(), raw_image, str(args.output), stem=method,
        )
        metadata["maps"][method] = {
            "shape": list(raw_map.shape),
            "heatmap_shape": list(heatmap.shape),
            "min": float(raw_map.min().item()),
            "max": float(raw_map.max().item()),
            "constant": bool(raw_map.max().item() == raw_map.min().item()),
        }
    metadata["elapsed_seconds"] = time.perf_counter() - started
    (args.output / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    LOG.info("Saved SigLIP explanations to %s", args.output)


if __name__ == "__main__":
    main()
