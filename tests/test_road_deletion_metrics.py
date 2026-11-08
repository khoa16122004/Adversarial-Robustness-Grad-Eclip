"""CPU tests for paired deletion metrics without pretrained checkpoints."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from road_evaluation.deletion_metrics import (
    METRIC_FILES, evaluate_both, evaluate_deletion,
)
from road_evaluation.road.imputations import ChannelMeanImputer, NoisyLinearImputer
from util import normalize_ImageNet1k


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, image):
        self.inputs.append(image.detach().cpu().clone())
        score = image[:, :, 0, 0].mean(dim=1)
        return torch.stack((score, -score), dim=1).softmax(dim=1)


class PairedDeletionTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) / 48
        self.saliency = np.arange(16, dtype=np.float32).reshape(4, 4)

    def test_names_auc_and_rng_isolation(self):
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        results = evaluate_both(
            TinyClassifier(), self.image, self.saliency, 0, "cpu",
            noise=.01, seed=7, percentages=[0, .25, .5],
        )
        self.assertEqual(set(results), {"road_deletion", "DEL_avg_channel"})
        for name, result in results.items():
            self.assertEqual(result["metric"], name)
            self.assertEqual(len(result["deletion_curve"]), 3)
            self.assertTrue(np.isfinite(result["deletion_curve"]).all())
            values = result["deletion_curve"]
            expected = sum((a + b) * .25 / 2 for a, b in zip(values, values[1:]))
            self.assertAlmostEqual(result["deletion_auc"], expected)
            json.dumps(result, allow_nan=False)
        np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
        self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
        torch.testing.assert_close(torch_state, torch.random.get_rng_state())

    def test_channel_mean_uses_original_image_for_every_percentage(self):
        model = TinyClassifier()
        original = self.image.clone()
        evaluate_deletion(model, self.image, self.saliency, 0, "cpu",
                          ChannelMeanImputer(), percentages=[.25, .5], seed=7)
        for inputs, count in zip(model.inputs, (4, 8)):
            expected = original.clone().reshape(3, 16)
            expected[:, -count:] = original.mean(dim=(1, 2))[:, None]
            torch.testing.assert_close(inputs, normalize_ImageNet1k(expected.reshape(1, 3, 4, 4)))
        torch.testing.assert_close(self.image, original)

    def test_both_methods_share_nested_masks(self):
        masks = []

        class RecordLinear(NoisyLinearImputer):
            def __call__(self, image, mask):
                masks.append(mask.clone())
                return image.clone()

        class RecordMean(ChannelMeanImputer):
            def __call__(self, image, mask):
                masks.append(mask.clone())
                return image.clone()

        for imputer in (RecordLinear(), RecordMean()):
            evaluate_deletion(TinyClassifier(), self.image, np.zeros((4, 4)), 0, "cpu",
                              imputer, percentages=[.25, .5, .75], seed=5)
        for left, right in zip(masks[:3], masks[3:]):
            torch.testing.assert_close(left, right)
        self.assertTrue(torch.all(masks[1] <= masks[0]))
        self.assertTrue(torch.all(masks[2] <= masks[1]))

    def test_reproducible_road_and_invalid_inputs(self):
        first = evaluate_both(TinyClassifier(), self.image, self.saliency, 0, "cpu", seed=7)
        second = evaluate_both(TinyClassifier(), self.image, self.saliency, 0, "cpu", seed=7)
        self.assertEqual(first, second)
        for percentages in ([.5, .25], [0, 1], [.2], [.1, float("nan")]):
            with self.subTest(percentages=percentages), self.assertRaises(ValueError):
                evaluate_both(TinyClassifier(), self.image, self.saliency, 0, "cpu",
                              percentages=percentages)
        with self.assertRaises(ValueError):
            evaluate_both(TinyClassifier(), self.image, np.full((4, 4), np.nan), 0, "cpu")

    def test_linear_zero_and_single_missing_pixel(self):
        imputer = NoisyLinearImputer(noise=0)
        mask = torch.ones(4, 4, dtype=torch.uint8)
        torch.testing.assert_close(imputer(self.image, mask), self.image)
        mask[1, 1] = 0
        filled = imputer(self.image, mask)
        self.assertEqual(filled.shape, self.image.shape)
        self.assertTrue(torch.isfinite(filled).all())
        torch.testing.assert_close(filled[:, mask.bool()], self.image[:, mask.bool()])

    def test_clip_runner_writes_both_and_migrates_legacy_road(self):
        import main_script.ROAD_evaluate_and_export as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample"
            sample.mkdir()
            Image.new("RGB", (4, 4), (100, 130, 160)).save(sample / "adversarial_image_del.png")
            np.save(sample / "del_del_adv_eclip_saliency.npy", self.saliency)
            (sample / "curve_information.json").write_text('{"retain_class_label": 0}')
            legacy = {"metric": "road_deletion", "deletion_auc": -1}
            (sample / METRIC_FILES["road_deletion"]).write_text(json.dumps(legacy))
            model = TinyClassifier()
            clip_model = SimpleNamespace(visual=SimpleNamespace(input_resolution=4), eval=lambda: None)
            argv = ["runner", "--input-root", str(root), "--device", "cpu",
                    "--percentages", ".25", ".5", "--seed", "7"]
            with patch.object(sys, "argv", argv), \
                    patch.object(runner.clip, "load", return_value=(clip_model, None)), \
                    patch.object(runner, "load_classnames", return_value=["dog", "cat"]), \
                    patch.object(runner, "build_zero_shot_clip_classifier", return_value=(model, None)), \
                    patch.object(runner, "build_causal_metric_model", return_value=model):
                runner.main()
                for name in ("road_deletion", "DEL_avg_channel"):
                    payload = json.loads((sample / METRIC_FILES[name]).read_text())
                    self.assertEqual(payload["metric"], name)
                    self.assertEqual(payload["seed"], 7)
                    self.assertEqual(len(payload["deletion_curve"]), 2)
                    self.assertGreaterEqual(payload["deletion_auc"], 0)
                calls = len(model.inputs)
                runner.main()
                self.assertEqual(len(model.inputs), calls)

    def test_siglip_runner_uses_recorded_config_and_writes_both(self):
        import road_score_SIGLIP as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample"
            sample.mkdir()
            config = {
                "model_type": "SigLIP", "model": "local-model", "mode": "del+ins",
                "hm_type": "eclip", "image_size": 4, "explainer_version": "test",
                "prompt_template": "a photo of {}", "classnames_path": None,
            }
            (root / "run_config.json").write_text(json.dumps(config))
            (sample / "curve_information.json").write_text('{"adv_pred_label": 1}')
            Image.new("RGB", (4, 4), (100, 130, 160)).save(sample / "adversarial_image_del_ins.png")
            np.save(sample / "joint_del_adv_eclip_saliency.npy", self.saliency)
            model = TinyClassifier()
            with patch.object(sys, "argv", ["runner", "--input-root", str(root), "--device", "cpu"]), \
                    patch.object(runner, "SigLIPAttackAdapter", return_value=SimpleNamespace(image_size=4)) as adapter, \
                    patch.object(runner, "load_classnames", return_value=["dog", "cat"]), \
                    patch.object(runner, "build_zero_shot_siglip_classifier", return_value=model), \
                    patch.object(runner, "build_causal_metric_model", return_value=model):
                runner.main()
                self.assertEqual(adapter.call_args.args[0].model, "local-model")
                for name in ("road_deletion", "DEL_avg_channel"):
                    payload = json.loads((sample / METRIC_FILES[name]).read_text())
                    self.assertEqual(payload["metric"], name)
                    self.assertEqual(payload["target_label"], 1)
                    self.assertEqual(payload["map_source"], "post_attack")
                runner.main()
                self.assertEqual(adapter.call_count, 1)

    def test_blip_compatibility_wrapper_is_true_road(self):
        from road_score_BLIP import evaluate
        result = evaluate(TinyClassifier(), self.image, self.saliency, 0, "cpu", .01, 42)
        self.assertEqual(result["metric"], "road_deletion")
        self.assertEqual(result["imputer"], "linear")


if __name__ == "__main__":
    unittest.main()
