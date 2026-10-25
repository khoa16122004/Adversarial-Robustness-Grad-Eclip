"""BLIP ITC explainers only; no attack dependencies. See README_BLIP_explainers.md."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

METHODS = ("gradeclip", "game", "gradcam", "maskclip", "rise")
LOG = logging.getLogger("blip-explainers")
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(r"E:\Khoatn_ICLR_MMM_Neurips\iclr_neurips_result\BLIP_model")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", required=True, help="One target caption per run.")
    parser.add_argument("--checkpoint", type=Path,
                        default=DATA_ROOT / "BLIP.pth")
    parser.add_argument("--blip-root", type=Path,
                        default=REPO_ROOT / "BLIP",
                        help="Salesforce BLIP source directory containing models and configs.")
    parser.add_argument("--output", type=Path,
                        default=DATA_ROOT / "explainer_results")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--vit", choices=("base", "large"), default="base")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--last-heads", choices=("1", "native"), default="1",
                        help="1 reproduces the notebook's modified final attention.")
    parser.add_argument("--game-start-layer", type=int, default=-1,
                        help="-1: final block only (repository default); 0: all blocks.")
    parser.add_argument("--rise-masks", type=int, default=2000)
    parser.add_argument("--rise-grid", type=int, default=8)
    parser.add_argument("--rise-prob", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=50, help="RISE batch size.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("image", "checkpoint"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name}: file does not exist: {getattr(args, name)}")
    if not (args.blip_root / "models" / "blip_itm.py").is_file():
        parser.error("--blip-root must contain models\\blip_itm.py")
    if not args.text.strip():
        parser.error("--text cannot be empty")
    if args.image_size <= 0 or args.image_size % 16:
        parser.error("--image-size must be a positive multiple of 16")
    if min(args.rise_masks, args.rise_grid, args.batch_size) <= 0:
        parser.error("--rise-masks, --rise-grid and --batch-size must be positive")
    if not 0 < args.rise_prob < 1:
        parser.error("--rise-prob must be between 0 and 1 (exclusive)")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    depth = 12 if args.vit == "base" else 24
    if not -1 <= args.game_start_layer < depth:
        parser.error(f"--game-start-layer must be -1 or in [0, {depth - 1}]")
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


def load_model(args, device):
    import torch
    from models.blip_itm import BLIP_ITM
    from models.vit import interpolate_pos_embed

    model = BLIP_ITM(
        med_config=str(args.blip_root / "configs" / "med_config.json"),
        image_size=args.image_size, vit=args.vit,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a BLIP checkpoint dictionary")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or "visual_encoder.pos_embed" not in state:
        raise ValueError("Checkpoint must contain BLIP visual_encoder weights")
    state = dict(state)
    state["visual_encoder.pos_embed"] = interpolate_pos_embed(
        state["visual_encoder.pos_embed"], model.visual_encoder,
    )
    incompatible = model.load_state_dict(state, strict=False)
    required_prefixes = ("visual_encoder.", "vision_proj.", "text_encoder.", "text_proj.")
    missing = [key for key in incompatible.missing_keys if key.startswith(required_prefixes)]
    if missing:
        raise RuntimeError(f"Missing weights required for ITC explainers: {missing}")
    LOG.info("Checkpoint loaded: missing=%s, unexpected=%d (e.g. momentum/queue weights)",
             incompatible.missing_keys, len(incompatible.unexpected_keys))
    if incompatible.unexpected_keys:
        LOG.info("Unexpected keys (first 20): %s", incompatible.unexpected_keys[:20])
    return model.to(device).eval()


def encode_image(model, image, last_heads, game_start=None):
    import torch

    vit = model.visual_encoder
    x = vit.patch_embed(image)
    x = torch.cat((vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = vit.pos_drop(x + vit.pos_embed)
    captured = {}
    game_attentions = []
    for index, block in enumerate(vit.blocks):
        final = index == len(vit.blocks) - 1
        capture_game = game_start is not None and index >= game_start
        if not final and not capture_game:
            x = block(x)
            continue
        residual = x
        tokens = block.norm1(x)
        batch, count, width = tokens.shape
        heads = 1 if final and last_heads == "1" else block.attn.num_heads
        qkv = block.attn.qkv(tokens).reshape(
            batch, count, 3, heads, width // heads,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = block.attn.attn_drop(
            ((q @ k.transpose(-2, -1)) * block.attn.scale).softmax(dim=-1)
        )
        attended = (attention @ v).transpose(1, 2).reshape(batch, count, width)
        x = residual + block.drop_path(block.attn.proj_drop(block.attn.proj(attended)))
        x = x + block.drop_path(block.mlp(block.norm2(x)))
        if capture_game:
            game_attentions.append(attention)
        if final:
            captured = {
                "input": residual, "attended": attended,
                "q": q, "k": k, "v": v,
            }
    features = model.vision_proj(vit.norm(x)[:, 0])
    return features, captured, game_attentions


def gradient_maps(model, image, text_feature, args):
    import torch
    import torch.nn.functional as F

    start = len(model.visual_encoder.blocks) - 1 if args.game_start_layer == -1 else args.game_start_layer
    features, cache, attentions = encode_image(
        model, image, args.last_heads, start if "game" in args.methods else None,
    )
    score = (F.normalize(features, dim=-1) @ text_feature.T).squeeze()
    side = args.image_size // 16
    maps = {}
    if "gradeclip" in args.methods:
        gradient = torch.autograd.grad(score, cache["attended"], retain_graph=True)[0][0, 0]
        q, k, v = (cache[key].detach() for key in ("q", "k", "v"))
        q, k, v = (value.transpose(1, 2).flatten(2)[0] for value in (q, k, v))
        locality = normalize_map((F.normalize(q[:1], dim=-1) *
                                  F.normalize(k[1:], dim=-1)).sum(-1))
        maps["gradeclip"] = (gradient * v[1:] * locality[:, None]).sum(-1).relu()
    if "gradcam" in args.methods:
        layer = cache["input"]
        gradient = torch.autograd.grad(score, layer, retain_graph=True)[0]
        weights = gradient.mean(dim=1, keepdim=True)
        maps["gradcam"] = (weights * layer).sum(-1)[0, 1:].relu()
    if "game" in args.methods:
        gradients = torch.autograd.grad(score, attentions, retain_graph=True)
        count = attentions[0].shape[-1]
        relevance = torch.eye(count, device=image.device).unsqueeze(0)
        for attention, gradient in zip(attentions, gradients):
            cam = (attention.detach() * gradient.detach()).clamp_min(0).mean(dim=1)
            relevance = relevance + cam.bmm(relevance)
        maps["game"] = relevance[0, 0, 1:]
    if "maskclip" in args.methods:
        with torch.no_grad():
            block = model.visual_encoder.blocks[-1]
            k, v = (cache[key].detach().transpose(1, 2).flatten(2) for key in ("k", "v"))
            keys = F.normalize(block.attn.proj(k), dim=-1)
            locality = (keys[:, :1] * keys[:, 1:]).sum(-1)
            dense = block.attn.proj_drop(block.attn.proj(v)) + cache["input"].detach()
            dense = dense + block.drop_path(block.mlp(block.norm2(dense)))
            dense = F.normalize(model.vision_proj(model.visual_encoder.norm(dense)), dim=-1)
            maps["maskclip"] = ((dense[:, 1:] @ text_feature.T).squeeze(-1) * locality)[0]
    return {name: value.detach().reshape(side, side) for name, value in maps.items()}, score.item()


def rise_map(model, image, text_feature, args):
    import torch
    import torch.nn.functional as F

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    size, grid = args.image_size, args.rise_grid
    cell = math.ceil(size / grid)
    up_size = (grid + 1) * cell
    saliency = torch.zeros((size, size), device=image.device)
    with torch.no_grad():
        for offset in range(0, args.rise_masks, args.batch_size):
            count = min(args.batch_size, args.rise_masks - offset)
            masks = []
            for _ in range(count):
                coarse = (torch.rand((1, 1, grid, grid), generator=generator) <
                          args.rise_prob).float()
                upsampled = F.interpolate(coarse, size=(up_size, up_size),
                                          mode="bilinear", align_corners=False)
                y, x = torch.randint(cell, (2,), generator=generator).tolist()
                masks.append(upsampled[:, :, y:y + size, x:x + size])
            masks = torch.cat(masks).to(image.device)
            features, _, _ = encode_image(model, image * masks, args.last_heads)
            scores = (F.normalize(features, dim=-1) @ text_feature.T).reshape(-1, 1, 1)
            saliency += (scores * masks[:, 0]).sum(dim=0)
            LOG.info("RISE: %d/%d masks", offset + count, args.rise_masks)
    return saliency / (args.rise_masks * args.rise_prob)


def save_outputs(raw_image, maps, args, metadata):
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image, ImageDraw

    args.output.mkdir(parents=True, exist_ok=True)
    raw_image.save(args.output / "input.png")
    panels = [raw_image.copy()]
    labels = ["input"]
    metadata["maps"] = {}
    width, height = raw_image.size
    for name in args.methods:
        raw = maps[name].float().cpu()
        if raw.ndim != 2 or not raw.isfinite().all():
            raise ValueError(f"{name}: expected a finite 2D heatmap")
        np.save(args.output / f"{name}_raw.npy", raw.numpy())
        normalized = normalize_map(raw)
        normalized = F.interpolate(normalized[None, None], size=(height, width),
                                   mode="bilinear", align_corners=False)[0, 0].clamp(0, 1)
        values = normalized.numpy()
        np.save(args.output / f"{name}_heatmap.npy", values)
        Image.fromarray((values * 255).astype(np.uint8)).save(args.output / f"{name}_heatmap.png")
        color = np.stack([np.clip(1.5 - np.abs(4 * values - shift), 0, 1)
                          for shift in (3, 2, 1)], axis=-1)
        overlay = (0.5 * np.asarray(raw_image, dtype=np.float32) +
                   0.5 * color * 255).clip(0, 255).astype(np.uint8)
        overlay = Image.fromarray(overlay)
        overlay.save(args.output / f"{name}_overlay.png")
        panels.append(overlay)
        labels.append(name)
        metadata["maps"][name] = {
            "raw_shape": list(raw.shape), "heatmap_shape": list(values.shape),
            "min": raw.min().item(), "max": raw.max().item(),
            "constant": bool(raw.max().item() == raw.min().item()),
        }
    thumb_width = min(width, 384)
    thumb_height = max(1, round(height * thumb_width / width))
    sheet = Image.new("RGB", (thumb_width * len(panels), thumb_height + 28), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        sheet.paste(panel.resize((thumb_width, thumb_height)), (index * thumb_width, 28))
        draw.text((index * thumb_width + 6, 7), label, fill="black")
    sheet.save(args.output / "comparison.png")
    (args.output / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8",
    )


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.path.insert(0, str(args.blip_root.resolve()))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device,
    )
    LOG.info("Device: %s; final attention heads: %s", device, args.last_heads)
    if args.last_heads == "1":
        LOG.warning("Notebook single-head adaptation changes the original BLIP ITC score.")
    model = load_model(args, device)
    with Image.open(args.image) as source:
        raw_image = source.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    image = preprocess(raw_image).unsqueeze(0).to(device)
    with torch.no_grad():
        text = model.tokenizer([args.text], padding="max_length", truncation=True,
                               max_length=35, return_tensors="pt").to(device)
        text_output = model.text_encoder(
            text.input_ids, attention_mask=text.attention_mask,
            return_dict=True, mode="text",
        )
        text_feature = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0]), dim=-1)
        native_features = F.normalize(model.vision_proj(model.visual_encoder(image)[:, 0]), dim=-1)
        native_score = (native_features @ text_feature.T).item()
        adapted_features, _, _ = encode_image(model, image, args.last_heads)
        adapted_score = (F.normalize(adapted_features, dim=-1) @ text_feature.T).item()
        if args.last_heads == "native":
            torch.testing.assert_close(adapted_features,
                                       model.vision_proj(model.visual_encoder(image)[:, 0]),
                                       rtol=1e-4, atol=1e-5)
    metadata = {
        "image": str(args.image.resolve()), "text": args.text,
        "checkpoint": str(args.checkpoint.resolve()), "blip_root": str(args.blip_root.resolve()),
        "vit": args.vit, "image_size": args.image_size, "device": str(device),
        "last_heads": args.last_heads, "methods": args.methods,
        "score_target": "BLIP image-text contrastive cosine (ITC), not ITM",
        "native_itc_score": native_score, "explained_itc_score": adapted_score,
        "game_start_layer": args.game_start_layer, "seed": args.seed,
        "rise": {"masks": args.rise_masks, "grid": args.rise_grid,
                 "probability": args.rise_prob, "batch_size": args.batch_size,
                 "baseline": "zero in normalized input space (RGB normalization mean)"},
        "torch_version": torch.__version__,
    }
    started = time.perf_counter()
    maps = {}
    if any(name != "rise" for name in args.methods):
        maps, score = gradient_maps(model, image, text_feature, args)
        if not math.isclose(score, adapted_score, rel_tol=1e-4, abs_tol=1e-5):
            raise RuntimeError("Explainer forward does not match the selected ITC score")
    if "rise" in args.methods:
        maps["rise"] = rise_map(model, image, text_feature, args)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metadata["explainer_seconds"] = time.perf_counter() - started
    save_outputs(raw_image, maps, args, metadata)
    LOG.info("Done: %s (ITC native=%.6f, explained=%.6f)",
             args.output.resolve(), native_score, adapted_score)


if __name__ == "__main__":
    main()
