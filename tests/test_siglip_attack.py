"""CPU contract tests for the SigLIP adapter and explainers."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize

from siglip_attack_adapter import (
    CLIP_MEAN, CLIP_STD, SigLIPAttackAdapter, build_zero_shot_siglip_classifier,
)
from test_siglip_explainers import (
    _pool_image, _pooling_gradeclip, audit_pooling_gradeclip,
    encode_image, gradient_maps, rise_map,
)


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

    def forward(self, pixel_values, return_dict=False):
        hidden = self.embeddings(pixel_values)
        for layer in self.encoder.layers:
            hidden = layer(hidden, attention_mask=None)
        features = self.head(self.post_layernorm(hidden))
        return SimpleNamespace(pooler_output=features) if return_dict else features


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

    def test_siglip_xshift_preserves_class_and_perturbation_bound(self):
        from XShift_score_SIGLIP import encode_siglip_tokens, run_xshift_attack

        adapter = self.make_adapter()
        classifier = build_zero_shot_siglip_classifier(adapter, ["bird", "dog"], "a photo of {}")
        image = torch.full((1, 3, 224, 224), 0.5)
        normalized = (image - image.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)) / image.new_tensor(
            CLIP_STD,
        ).view(1, 3, 1, 1)
        features = encode_siglip_tokens(adapter, normalized)
        self.assertEqual(features.shape, (1, 196, 8))
        self.assertTrue(torch.isfinite(features).all())
        with torch.no_grad():
            label = int(classifier(normalized).argmax(dim=1).item())
        adversarial, details = run_xshift_attack(
            adapter, classifier, image, label, label,
            classifier.zero_shot_weights.detach(), eps=4 / 255, alpha=1 / 255,
            xshift_steps=2,
        )
        self.assertTrue(torch.isfinite(adversarial).all())
        self.assertLessEqual((adversarial - image).abs().max().item(), 4 / 255 + 1e-6)
        self.assertGreaterEqual(adversarial.min().item(), 0)
        self.assertLessEqual(adversarial.max().item(), 1)
        self.assertEqual(details["selected_pred_label"], label)
        self.assertEqual(len(details["trace"]), 2)

    def test_siglip_launchers_are_regular_shell_scripts(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("run_siglip_xshift.sh", "run_siglip_clean.sh"):
            source = (root / "script_sh" / name).read_text(encoding="utf-8")
            with self.subTest(name=name):
                self.assertNotIn("#SBATCH", source)
                expected = "XSHIFT_STEPS" if "xshift" in name else "clean_score_SIGLIP.py"
                self.assertIn(expected, source)

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

    def test_pooling_forward_and_input_gradients_match_native_head(self):
        head = self.model.vision_model.head
        hidden = torch.randn(2, 4, 8, requires_grad=True)
        actual, cache = _pool_image(head, hidden)
        expected = head(hidden)
        torch.testing.assert_close(actual, expected)
        actual_gradient = torch.autograd.grad(actual.square().sum(), hidden)[0]
        expected_gradient = torch.autograd.grad(expected.square().sum(), hidden)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        probe = head.probe.expand(2, -1, -1)
        _, native_attention = head.attention(
            probe, hidden, hidden, need_weights=True, average_attn_weights=False,
        )
        torch.testing.assert_close(cache["attention"], native_attention)

    def test_attack_rejects_constant_gradeclip_instead_of_ranking_zero_map(self):
        adapter = self.make_adapter()
        image = torch.zeros(1, 3, 224, 224)
        text = torch.ones(1, 8)
        with patch(
            "siglip_attack_adapter.gradient_maps",
            return_value=({"gradeclip": torch.zeros(14, 14)}, 0.0),
        ):
            with self.assertRaisesRegex(ValueError, "arbitrary pixel ranking"):
                adapter.generate_hm("eclip", image, text, ["bird"], Resize((224, 224)))

    def test_gradeclip_uses_pooling_probe_gradient_and_values(self):
        model = self.model
        image = torch.randn(1, 3, 224, 224)
        text = F.normalize(model.get_text_features(torch.tensor([[3, 1]])), dim=-1)
        features, cache, _ = encode_image(model, image)
        pool = cache["pooling"]
        score = (F.normalize(features, dim=-1) @ text.T).sum()
        gradient = torch.autograd.grad(score, pool["attended"])[0][0, 0]
        expected = torch.zeros(196)
        head_dim = pool["v"].shape[-1]
        for head in range(pool["v"].shape[1]):
            query = pool["q"][0, head, 0].detach()
            keys = pool["k"][0, head].detach()
            values = pool["v"][0, head].detach()
            similarity = F.cosine_similarity(query[None], keys, dim=-1)
            locality = (similarity - similarity.min()) / (
                similarity.max() - similarity.min()
            ).clamp_min(torch.finfo(similarity.dtype).eps)
            channels = gradient[head * head_dim:(head + 1) * head_dim]
            expected += (channels * values).sum(-1) * locality
        expected = expected.relu().reshape(14, 14)
        args = SimpleNamespace(game_start_layer=-1, methods=["gradeclip"])
        maps, _ = gradient_maps(model, image, text, args)
        torch.testing.assert_close(maps["gradeclip"], expected)

    def test_gradeclip_keeps_head_localities_separate_and_relu_after_summing(self):
        pooling = {
            "q": torch.tensor([[[[1., 0.]], [[1., 0.]]]]),
            "k": torch.tensor([[[[1., 0.], [-1., 0.]],
                                [[-1., 0.], [1., 0.]]]]),
            "v": torch.tensor([[[[2., 0.], [100., 0.]],
                                [[100., 0.], [-3., 0.]]]]),
        }
        gradient = torch.tensor([[[1., 0., 1., 0.]]])
        saliency, signed, _ = _pooling_gradeclip(pooling, gradient)
        torch.testing.assert_close(signed, torch.tensor([[2., -3.]]))
        torch.testing.assert_close(saliency, torch.tensor([[2., 0.]]))

        # A head's positive Q/K scale must not change its cosine locality.
        scaled = {name: value.clone() for name, value in pooling.items()}
        scaled["q"][:, 0] *= 100
        scaled["k"][:, 0] *= 100
        torch.testing.assert_close(_pooling_gradeclip(scaled, gradient)[0], saliency)

    def test_pooling_attention_gradient_matches_value_channel_contributions(self):
        hidden = torch.randn(1, 4, 8, requires_grad=True)
        features, pool = _pool_image(self.model.vision_model.head, hidden)
        score = features.square().sum()
        gradient, attention_gradient = torch.autograd.grad(
            score, (pool["attended"], pool["attention"]),
        )
        heads, head_dim = pool["v"].shape[1], pool["v"].shape[-1]
        per_head_gradient = gradient.reshape(1, 1, heads, head_dim).transpose(1, 2)
        expected = (per_head_gradient * pool["v"]).sum(-1).unsqueeze(2)
        torch.testing.assert_close(attention_gradient, expected)

    def test_failure_audit_compares_native_computation_on_same_input(self):
        image = torch.randn(1, 3, 224, 224)
        text = torch.randn(1, 8)
        report, tensors = audit_pooling_gradeclip(self.model, image, text)
        for key in ("features", "input_gradient", "pooling_attention_gradient"):
            self.assertTrue(report[key]["matches"])
        torch.testing.assert_close(tensors["pixel_values"], image)
        self.assertEqual(tensors["signed_per_head"].shape, (1, 2, 196))

    def test_game_propagates_pooling_relevance_through_backbone(self):
        model = self.model
        image = torch.randn(1, 3, 224, 224)
        text = F.normalize(model.get_text_features(torch.tensor([[3, 1]])), dim=-1)
        features, cache, attentions = encode_image(model, image, game_start=0)
        score = (F.normalize(features, dim=-1) @ text.T).sum()
        all_attention = [*attentions, cache["pooling"]["attention"]]
        gradients = torch.autograd.grad(score, all_attention)
        relevance = torch.eye(196).unsqueeze(0)
        for attention, gradient in zip(attentions, gradients[:-1]):
            cam = (attention * gradient).clamp_min(0).mean(dim=1)
            relevance = (torch.eye(196).unsqueeze(0) + cam).bmm(relevance)
        seed = (all_attention[-1] * gradients[-1]).clamp_min(0).mean(dim=1)
        expected = seed.bmm(relevance)[0, 0].reshape(14, 14)
        args = SimpleNamespace(game_start_layer=0, methods=["game"])
        maps, _ = gradient_maps(model, image, text, args)
        torch.testing.assert_close(maps["game"], expected)

    def test_rise_weights_masks_with_sigmoid_probability(self):
        args = SimpleNamespace(
            seed=7, rise_masks=4, rise_grid=2, rise_prob=0.5, batch_size=2,
        )
        image = torch.ones(1, 3, 8, 8)
        text = torch.ones(1, 8)
        with patch("test_siglip_explainers.encode_image", return_value=(text, {}, [])), \
                patch("test_siglip_explainers.image_text_logits") as logits:
            logits.return_value = torch.full((2, 1), -2.0)
            negative = rise_map(self.model, image, text, args)
            logits.return_value = torch.zeros(2, 1)
            zero = rise_map(self.model, image, text, args)
        self.assertGreater(negative.sum().item(), 0)
        torch.testing.assert_close(
            negative, zero * (torch.tensor(-2.0).sigmoid() / 0.5),
        )

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
        actual_gradient = torch.autograd.grad(features.square().sum(), image, retain_graph=True)[0]
        native_gradient = torch.autograd.grad(expected.square().sum(), image)[0]
        torch.testing.assert_close(actual_gradient, native_gradient, rtol=1e-4, atol=1e-5)
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
