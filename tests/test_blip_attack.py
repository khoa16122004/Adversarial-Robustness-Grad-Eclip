"""CPU contract tests with a tiny BLIP-shaped model; no checkpoints/downloads."""

import json
from itertools import product
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize

import adv_score_BLIP as runner
from blip_attack_adapter import BLIPAttackAdapter, HM_METHODS, build_zero_shot_blip_classifier
from test_blip_explainers import encode_image


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.scale = 4 ** -0.5
        self.qkv = nn.Linear(8, 24)
        self.attn_drop = nn.Identity()
        self.proj = nn.Linear(8, 8)
        self.proj_drop = nn.Identity()

    def forward(self, x):
        batch, count, _ = x.shape
        q, k, v = self.qkv(x).reshape(batch, count, 3, 2, 4).permute(2, 0, 3, 1, 4)
        attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        return self.proj((attention @ v).transpose(1, 2).reshape(batch, count, 8))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(8)
        self.attn = TinyAttention()
        self.norm2 = nn.LayerNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
        self.drop_path = nn.Identity()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyPatches(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 8, 16, stride=16)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = TinyPatches()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 8))
        self.pos_embed = nn.Parameter(torch.randn(1, 197, 8) * 0.01)
        self.pos_drop = nn.Identity()
        self.blocks = nn.Sequential(TinyBlock(), TinyBlock())
        self.norm = nn.LayerNorm(8)

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return self.norm(self.blocks(x + self.pos_embed))


class Tokens(SimpleNamespace):
    def to(self, device):
        return Tokens(input_ids=self.input_ids.to(device), attention_mask=self.attention_mask.to(device))


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyBLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = TinyVision()
        self.vision_proj = nn.Linear(8, 4)
        self.text_encoder = TinyText()
        self.text_proj = nn.Linear(8, 4)

    def tokenizer(self, texts, **kwargs):
        ids = torch.tensor([[sum(text.encode()) % 32] for text in texts])
        return Tokens(input_ids=ids, attention_mask=torch.ones_like(ids))


def options():
    return SimpleNamespace(
        image_size=224, last_heads="1", game_start_layer=0,
        rise_masks=4, rise_grid=4, rise_prob=0.5, rise_batch_size=2, seed=7,
        blip_root=Path(__file__).resolve().parents[1] / "BLIP", checkpoint=Path("unused.pth"),
        vit="base", device="cpu", prompt_template="a photo of {}",
        hm_type="eclip", target_source="pred", gt_label=None,
        step=25088, kernel_size=3, kernel_sigma=1, save_process=False,
        eps=4.0, alpha=1.0, pgd_steps=2, process_batch_size=2, verbose=0,
        classnames_path=None,
    )


class BLIPAttackTests(unittest.TestCase):
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
        self.model = TinyBLIP().eval()

    def make_adapter(self):
        with patch("blip_attack_adapter.load_model", return_value=self.model):
            return BLIPAttackAdapter(self.args, "cpu")

    def test_native_forward_matches_and_classifier_input_gradients(self):
        self.args.last_heads = "native"
        adapter = self.make_adapter()
        image = torch.randn(1, 3, 224, 224, requires_grad=True)
        features, _, _ = encode_image(self.model, image, "native")
        expected = self.model.vision_proj(self.model.visual_encoder(image)[:, 0])
        torch.testing.assert_close(features, expected)
        classifier = build_zero_shot_blip_classifier(adapter, ["dog", "bird"], "a photo of {}")
        logits = classifier(image)
        self.assertEqual(logits.shape, (1, 2))
        gradient = torch.autograd.grad(logits[0, 0], image)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(not parameter.requires_grad for parameter in adapter.parameters()))

    def test_all_explainers_finite_shape_and_preserve_attack_graph(self):
        adapter = self.make_adapter()
        with torch.no_grad():
            text = F.normalize(adapter.encode_text(["bird"]), dim=-1)
        for heads in ("1", "native"):
            adapter.options.last_heads = heads
            for method in HM_METHODS:
                with self.subTest(heads=heads, method=method):
                    image = torch.randn(1, 3, 224, 224, requires_grad=True)
                    objective = adapter.encode_image(image).sum()
                    saliency = adapter.generate_hm(method, image, text, ["bird"], Resize((224, 224)))
                    self.assertEqual(saliency.shape, (224, 224))
                    self.assertFalse(saliency.requires_grad)
                    self.assertTrue(torch.isfinite(saliency).all())
                    self.assertGreaterEqual(saliency.min().item(), 0)
                    self.assertLessEqual(saliency.max().item(), 1)
                    self.assertTrue(torch.isfinite(torch.autograd.grad(objective, image)[0]).all())
                    if method == "rise":
                        torch.testing.assert_close(
                            saliency,
                            adapter.generate_hm(method, image, text, ["bird"], Resize((224, 224))),
                        )

    def test_output_parity_all_attack_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = (torch.rand(224, 224, 3).numpy() * 255).astype(np.uint8)
            Image.fromarray(image).save(root / "input.png")
            (root / "samples.json").write_text('{"sample": "input.png"}', encoding="utf-8")
            (root / "classes.json").write_text('["dog", "bird"]', encoding="utf-8")
            args = self.args
            args.img_dir = str(root)
            args.sample_path = str(root / "samples.json")
            args.classnames_path = str(root / "classes.json")
            args.output_dir = str(root / "out")
            for mode, method in product(("del", "ins", "del+ins"),
                                        ("eclip", "game", "gradcam", "maskclip", "rise")):
                with self.subTest(mode=mode, method=method):
                    args.mode = mode
                    args.hm_type = method
                    args.save_process = mode == "del" and method == "eclip"
                    # Only CUDA placement is replaced for this CPU contract test.
                    # The shared PGD engines and artifact writers run unchanged.
                    with patch.object(runner, "parse_args", return_value=args), \
                         patch("blip_attack_adapter.load_model", return_value=self.model), \
                         patch("torch.cuda.is_available", return_value=True), \
                         patch("torch.cuda.current_device", return_value=0), \
                         patch("torch.cuda.set_device"), \
                         patch.object(torch.Tensor, "cuda", lambda tensor: tensor):
                        runner.main()
                    folder = root / "out" / method / mode / "sample"
                    prefix = "joint" if mode == "del+ins" else mode
                    filename = "adversarial_image_del_ins.png" if mode == "del+ins" else f"adversarial_image_{mode}.png"
                    self.assertTrue((folder / filename).is_file())
                    payload = json.loads((folder / "curve_information.json").read_text())
                    self.assertEqual(payload["attack_mode"], mode)
                    self.assertEqual(len(payload["pgd_trace"]), 2)
                    self.assertIn("clean_prob", payload)
                    self.assertIn("adv_prob", payload)
                    for rerun_mode in ("del", "ins"):
                        self.assertEqual(len(payload["rerun"][rerun_mode]["curve"]), 3)
                        self.assertTrue(np.isfinite(payload["rerun"][rerun_mode]["auc"]))
                        stem = f"{prefix}_{rerun_mode}_adv_{method}_saliency"
                        for suffix in (".npy", ".png", "_overlay.png"):
                            self.assertTrue((folder / (stem + suffix)).is_file())
                        self.assertTrue((folder / f"{prefix}_{rerun_mode}_summary.png").is_file())
                        if args.save_process:
                            process_dir = folder / f"steps_{prefix}_{rerun_mode}"
                            self.assertEqual(len(list(process_dir.glob("*.png"))), 3)
                        saliency = np.load(folder / f"{stem}.npy")
                        self.assertEqual(saliency.shape, (224, 224))
                        self.assertTrue(np.isfinite(saliency).all())
                    if mode == "del+ins":
                        self.assertAlmostEqual(
                            payload["combined_score"],
                            sum(payload["rerun"][name]["auc"] for name in ("del", "ins")),
                        )
                    self.assertTrue((folder.parent / "run_config.json").is_file())

    def test_no_clip_model_modules_imported(self):
        import sys

        for name in ("clip_utils", "generate_emap", "CLIP", "Game_MM_CLIP.clip", "CLIP_Surgery.clip"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
