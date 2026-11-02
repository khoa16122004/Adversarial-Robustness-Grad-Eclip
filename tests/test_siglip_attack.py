"""CPU contract tests for the SigLIP adapter and explainers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize

from siglip_attack_adapter import (
    CLIP_MEAN, CLIP_STD, SigLIPAttackAdapter, build_zero_shot_siglip_classifier,
)
from test_siglip_explainers import encode_image, gradient_maps


class TinySiglipAttention(nn.Module):
    def __init__(self, width=8, heads=2):
        super().__init__()
        self.num_heads = heads
        self.scale = (width // heads) ** -0.5
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)


class TinySiglipLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(8)
        self.self_attn = TinySiglipAttention()
        self.layer_norm2 = nn.LayerNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))

    def forward(self, hidden, attention_mask=None):
        normalized = self.layer_norm1(hidden)
        batch, tokens, width = normalized.shape
        heads = self.self_attn.num_heads
        dim = width // heads

        def project(layer):
            return layer(normalized).view(batch, tokens, heads, dim).transpose(1, 2)

        q, k, v = (
            project(layer)
            for layer in (
                self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj,
            )
        )
        weights = (q @ k.transpose(-2, -1) * self.self_attn.scale).softmax(-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, tokens, width)
        hidden = hidden + self.self_attn.out_proj(attended)
        return hidden + self.mlp(self.layer_norm2(hidden))


class TinySiglipEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, 8, 16, stride=16)
        self.position_embedding = nn.Embedding(196, 8)

    def forward(self, pixels):
        patches = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        positions = torch.arange(196, device=pixels.device).unsqueeze(0)
        return patches + self.position_embedding(positions)


class TinySiglipEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinySiglipLayer(), TinySiglipLayer()])


class TinySiglipVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = TinySiglipEmbeddings()
        self.encoder = TinySiglipEncoder()
        self.post_layernorm = nn.LayerNorm(8)
        self.head = TinySiglipPoolingHead()

    def forward(self, pixels):
        hidden = self.embeddings(pixels)
        for layer in self.encoder.layers:
            hidden = layer(hidden, attention_mask=None)
        return self.head(self.post_layernorm(hidden))


class TinySiglipPoolingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, 8))
        self.attention = nn.MultiheadAttention(8, 2, batch_first=True)
        self.layernorm = nn.LayerNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))

    def forward(self, hidden):
        probe = self.probe.expand(hidden.shape[0], -1, -1)
        pooled = self.attention(probe, hidden, hidden)[0]
        return (pooled + self.mlp(self.layernorm(pooled)))[:, 0]


class TinySiglip(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = TinySiglipVision()
        self.token_embedding = nn.Embedding(32, 8)
        self.text_projection = nn.Linear(8, 8)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(-2.0))
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(image_size=224, patch_size=16),
        )

    def get_image_features(self, pixel_values):
        return self.vision_model(pixel_values)

    def get_text_features(self, input_ids, attention_mask=None):
        return self.text_projection(self.token_embedding(input_ids).mean(dim=1))


class TinyImageProcessor:
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]


class TinyProcessor:
    image_processor = TinyImageProcessor()

    def __call__(self, text, padding, return_tensors):
        ids = torch.tensor([[sum(item.encode()) % 32, 1] for item in text])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def options():
    return SimpleNamespace(
        model="unused", image_size=224, game_start_layer=0,
        rise_masks=4, rise_grid=4, rise_prob=0.5, rise_batch_size=2, seed=7,
    )


class SiglipAttackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.args = options()
        self.model = TinySiglip().eval()
        self.processor = TinyProcessor()

    def make_adapter(self):
        with patch(
            "siglip_attack_adapter.load_model",
            return_value=(self.model, self.processor),
        ):
            return SigLIPAttackAdapter(self.args, "cpu")

    def test_manual_vision_features_match_model_and_classifier_has_gradients(self):
        adapter = self.make_adapter()
        pixels = torch.rand(1, 3, 224, 224)
        features, _, _ = encode_image(self.model, pixels)
        torch.testing.assert_close(features, self.model.get_image_features(pixels))

        clip_mean = pixels.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        clip_std = pixels.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        clip_normalized = (pixels * 0.5 + 0.5 - clip_mean) / clip_std
        converted_pixels = adapter.to_siglip_pixels(clip_normalized)
        expected_pixels = pixels * 0.5 + 0.5
        torch.testing.assert_close(converted_pixels, (expected_pixels - 0.5) / 0.5)

        classifier = build_zero_shot_siglip_classifier(
            adapter, ["bird", "dog"], "This is a photo of {}.",
        )
        input_image = torch.randn(1, 3, 224, 224, requires_grad=True)
        logits = classifier(input_image)
        self.assertEqual(logits.shape, (1, 2))
        gradient = torch.autograd.grad(logits[0, 0], input_image)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))

    def test_all_explainers_are_finite_and_do_not_consume_outer_graph(self):
        adapter = self.make_adapter()
        text = F.normalize(adapter.encode_text(["bird"]), dim=-1)
        image = torch.randn(1, 3, 224, 224, requires_grad=True)
        objective = adapter.encode_image(image).sum()
        for method in ("eclip", "game", "gradcam", "maskclip", "rise"):
            with self.subTest(method=method):
                saliency = adapter.generate_hm(
                    method, image, text, ["bird"], Resize((224, 224)),
                )
                self.assertEqual(saliency.shape, (224, 224))
                self.assertFalse(saliency.requires_grad)
                self.assertTrue(torch.isfinite(saliency).all())
                self.assertGreaterEqual(saliency.min().item(), 0)
                self.assertLessEqual(saliency.max().item(), 1)
                self.assertTrue(torch.isfinite(torch.autograd.grad(
                    objective, image, retain_graph=True,
                )[0]).all())

    def test_gradient_map_score_matches_siglip_logit(self):
        model = self.model
        image = torch.randn(1, 3, 224, 224)
        text = F.normalize(model.get_text_features(torch.tensor([[3, 1]])), dim=-1)
        args = SimpleNamespace(game_start_layer=0, methods=["gradeclip", "game", "gradcam", "maskclip"])
        maps, score = gradient_maps(model, image, text, args)
        image_features = F.normalize(model.get_image_features(image), dim=-1)
        expected = (
            model.logit_scale.exp() * (image_features @ text.T) + model.logit_bias
        ).item()
        self.assertAlmostEqual(score, expected, places=5)
        self.assertEqual(set(maps), {"gradeclip", "game", "gradcam", "maskclip"})
        for values in maps.values():
            self.assertEqual(values.shape, (14, 14))
            self.assertTrue(torch.isfinite(values).all())

    def test_real_transformers_siglip_forward_and_explainers(self):
        from transformers import SiglipConfig, SiglipModel

        config = SiglipConfig(
            text_config={
                "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 2, "num_attention_heads": 2,
                "max_position_embeddings": 4,
            },
            vision_config={
                "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 2, "num_attention_heads": 2,
                "image_size": 32, "patch_size": 16,
            },
        )
        model = SiglipModel(config).eval().requires_grad_(False)
        image = torch.randn(1, 3, 32, 32, requires_grad=True)
        features, _, _ = encode_image(model, image)
        expected = model.vision_model(pixel_values=image, return_dict=True).pooler_output
        torch.testing.assert_close(features, expected, rtol=1e-4, atol=1e-5)
        text = F.normalize(model.text_model(
            input_ids=torch.tensor([[3, 1, 1, 1]]), return_dict=True,
        ).pooler_output, dim=-1)
        args = SimpleNamespace(
            game_start_layer=0, methods=["gradeclip", "game", "gradcam", "maskclip"],
        )
        maps, _ = gradient_maps(model, image, text, args)
        for values in maps.values():
            self.assertEqual(values.shape, (2, 2))
            self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.isfinite(torch.autograd.grad(features.sum(), image)[0]).all())


if __name__ == "__main__":
    unittest.main()
