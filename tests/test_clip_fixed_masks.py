"""Fixed-mask contracts with tiny CPU models and no downloaded checkpoints."""

import json
from itertools import product
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor

import adv_score_CLIP_fixed_masked as runner
import run_different_masking_strategies_CLIP as strategies
import RISE.evaluation as evaluation
import util


class TinyScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.inputs = []

    def forward(self, image):
        self.inputs.append(image.detach().clone())
        score = image.mean(dim=(1, 2, 3)) * self.scale
        return torch.stack((score, -score), dim=1).softmax(dim=1)


class FixedMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_metric(self, mode, fixed=True):
        model = TinyScore()
        if mode == "del+ins":
            cls = (evaluation.JointAdversarialCausalMetricFixedMask if fixed
                   else evaluation.JointAdversarialCausalMetric)
            metric = cls(model, model, 2, torch.zeros_like, torch.zeros_like,
                         "eclip", None, ["dog"], None, None)
        else:
            cls = (evaluation.AdversarialCausalMetricFixedMask if fixed
                   else evaluation.AdversarialCausalMetric)
            metric = cls(model, model, mode, 2, torch.zeros_like,
                         "eclip", None, ["dog"], None, None)
        return metric, model

    def test_masks_are_frozen_while_image_changes_in_all_modes(self):
        image = torch.tensor([.2, .4, .6, .8]).reshape(1, 1, 2, 2).repeat(1, 3, 1, 1)
        for mode, source in product(("del", "ins", "del+ins"), ("clean", "random")):
            with self.subTest(mode=mode, source=source), \
                    patch.object(evaluation, "HW", 4), \
                    patch.object(evaluation, "normalize_ImageNet1k", side_effect=lambda x: x):
                metric, model = self.make_metric(mode)
                explanation = Mock(return_value=np.arange(4).reshape(2, 2))
                if source == "random":
                    explanation.side_effect = AssertionError("Random optimization must not call saliency")
                adversarial, details = metric.single_run(
                    image, explanation, target_class=0, mask_source=source, seed=7,
                    eps=.1, alpha=.01, pgd_steps=3, process_batch_size=1,
                )
                self.assertEqual(explanation.call_count, 1 if source == "clean" else 0)
                if source == "clean":
                    torch.testing.assert_close(explanation.call_args.args[2], image)
                expected_order = (np.array([[3, 2, 1, 0]]) if source == "clean"
                                  else np.random.default_rng(7).permutation(4).reshape(1, 4))
                np.testing.assert_array_equal(metric.fixed_salient_order, expected_order)
                self.assertEqual(details["mask_source"], source)
                self.assertEqual(details["seed"], 7 if source == "random" else None)
                self.assertEqual(len(details["pgd_trace"]), 3)
                self.assertTrue(np.isfinite(details["loss"]).all())
                self.assertTrue(torch.isfinite(adversarial).all())
                self.assertLessEqual((adversarial - image).abs().max().item(), .1 + 1e-6)
                self.assertGreaterEqual(adversarial.min().item(), 0)
                self.assertLessEqual(adversarial.max().item(), 1)
                self.assertGreater((adversarial - image).abs().max().item(), 0)
                calls_per_step = 7 if mode == "del+ins" else 4
                selected = expected_order[0, :2]
                remaining = expected_order[0, 2:]
                for k in range(3):
                    offset = 1 + k * calls_per_step
                    current = model.inputs[offset].reshape(1, 3, 4)
                    paths = ("del", "ins") if mode == "del+ins" else (mode,)
                    for path_index, path in enumerate(paths):
                        partial = model.inputs[offset + 2 + 3 * path_index].reshape(1, 3, 4)
                        zeros, retained = (selected, remaining) if path == "del" else (remaining, selected)
                        torch.testing.assert_close(partial[:, :, zeros], torch.zeros(1, 3, 2))
                        torch.testing.assert_close(partial[:, :, retained], current[:, :, retained])
                self.assertFalse(torch.equal(model.inputs[1], model.inputs[1 + calls_per_step]))
                torch.testing.assert_close(image, torch.tensor([.2, .4, .6, .8]).reshape(1, 1, 2, 2).repeat(1, 3, 1, 1))

    def test_seed_reproducibility_reset_per_run_and_rng_isolation(self):
        image = torch.full((1, 3, 2, 2), .5)
        for mode in ("del", "ins", "del+ins"):
            with self.subTest(mode=mode), patch.object(evaluation, "HW", 4), \
                    patch.object(evaluation, "normalize_ImageNet1k", side_effect=lambda x: x):
                metric, _ = self.make_metric(mode)
                state = np.random.get_state()
                first = metric.single_run(image, None, mask_source="random", seed=7, pgd_steps=2)
                first_order = metric.fixed_salient_order.copy()
                second = metric.single_run(image, None, mask_source="random", seed=7, pgd_steps=2)
                torch.testing.assert_close(first[0], second[0])
                np.testing.assert_array_equal(first_order, metric.fixed_salient_order)
                after = np.random.get_state()
                self.assertEqual(state[0], after[0])
                np.testing.assert_array_equal(state[1], after[1])
                self.assertEqual(state[2:], after[2:])
                metric.single_run(image, None, mask_source="random", seed=8, pgd_steps=0)
                np.testing.assert_array_equal(
                    metric.fixed_salient_order, np.random.default_rng(8).permutation(4).reshape(1, 4)
                )
                clean = Mock(return_value=np.arange(4))
                returned = metric.single_run(image, clean, eps=0, pgd_steps=2, return_details=False)
                self.assertEqual(clean.call_count, 1)
                torch.testing.assert_close(returned, image)

    def test_dynamic_attack_explainers_still_run_each_pgd_step(self):
        for mode in ("del", "ins", "del+ins"):
            with self.subTest(mode=mode), patch.object(evaluation, "HW", 4), \
                    patch.object(evaluation, "normalize_ImageNet1k", side_effect=lambda x: x):
                metric, _ = self.make_metric(mode, fixed=False)
                explanation = Mock(return_value=np.arange(4))
                metric.single_run(torch.full((1, 3, 2, 2), .5), explanation, pgd_steps=3)
                self.assertEqual(explanation.call_count, 3)

    def test_invalid_sources_seeds_maps_and_steps_raise(self):
        image = torch.full((1, 3, 2, 2), .5)
        for mode in ("del", "del+ins"):
            with self.subTest(mode=mode), patch.object(evaluation, "HW", 4):
                metric, _ = self.make_metric(mode)
                for options in ({"mask_source": "unknown"}, {"mask_source": "random", "seed": -1},
                                {"mask_source": "random", "seed": None}):
                    with self.assertRaises(ValueError):
                        metric.single_run(image, None, **options)
                for saliency in (np.zeros(3), np.full(4, np.nan)):
                    with self.assertRaises(ValueError):
                        metric.single_run(image, Mock(return_value=saliency))
                metric.step = 0
                with self.assertRaises(ValueError):
                    metric.single_run(image, None, mask_source="random")

    def test_runner_evaluates_fresh_adversarial_map_and_persists_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (224, 224), (128, 128, 128)).save(root / "image.png")
            (root / "samples.json").write_text(json.dumps({"sample": "image.png"}), encoding="utf-8")
            fake_model = SimpleNamespace(
                visual=SimpleNamespace(input_resolution=224), eval=lambda: None,
                encode_text=lambda tokens: torch.ones(1, 2),
            )
            fake_clip = SimpleNamespace(
                load=lambda *a, **k: (fake_model, lambda img: util.normalize_ImageNet1k(ToTensor()(img))),
                tokenize=lambda texts: torch.zeros(1, 1, dtype=torch.long),
            )
            for mode, source in product(("del", "ins", "del+ins"), ("clean", "random")):
                with self.subTest(mode=mode, source=source):
                    model = TinyScore()
                    clean_map = torch.arange(224 * 224).reshape(224, 224)
                    adversarial_map = clean_map.flip((0, 1))
                    explanation = Mock(
                        side_effect=[clean_map, adversarial_map] if source == "clean" else [adversarial_map]
                    )
                    evaluated_maps = []
                    original_run = evaluation.CausalMetric.single_run

                    def evaluate(metric, image, saliency, **kwargs):
                        evaluated_maps.append(np.asarray(saliency).copy())
                        return original_run(metric, image, saliency, **kwargs)

                    argv = [
                        "runner", "--img-dir", str(root), "--sample-path", str(root / "samples.json"),
                        "--output-dir", str(root / "outputs"), "--mode", mode, "--mask-source", source,
                        "--seed", "7", "--device", "cpu", "--step", "25088", "--pgd-steps", "2",
                        "--process-batch-size", "1", "--target-source", "gt", "--gt-label", "1",
                    ]
                    with patch.object(sys, "argv", argv), \
                            patch.dict(sys.modules, {"CLIP": SimpleNamespace(clip=fake_clip)}), \
                            patch.object(util, "load_classnames", return_value=["dog", "cat"]), \
                            patch.object(util, "build_zero_shot_clip_classifier", return_value=(model, None)), \
                            patch.object(util, "build_causal_metric_model", return_value=model), \
                            patch.object(util, "build_blur_substrate", return_value=torch.zeros_like), \
                            patch.object(util, "generate_hm", explanation), \
                            patch.object(util, "save_saliency_outputs"), \
                            patch.object(util, "save_causal_metric_summary"), \
                            patch.object(evaluation.CausalMetric, "single_run", evaluate):
                        runner.main()
                    strategy = "clean" if source == "clean" else "random_seed_7"
                    sample_dir = root / "outputs" / strategy / "eclip" / mode / "sample"
                    payload = json.loads((sample_dir / "curve_information.json").read_text(encoding="utf-8"))
                    self.assertEqual(payload["mask_source"], source)
                    self.assertEqual(payload["seed"], 7 if source == "random" else None)
                    self.assertEqual(payload["evaluation_map_source"], "adversarial")
                    self.assertEqual(payload["retain_class_label"], 1)
                    self.assertAlmostEqual(payload["clean_prob"], float(model(util.normalize_ImageNet1k(
                        torch.full((1, 3, 224, 224), 128 / 255)
                    ))[0, 1]), places=6)
                    self.assertEqual(explanation.call_count, 2 if source == "clean" else 1)
                    self.assertEqual(len(evaluated_maps), 2)
                    for saliency in evaluated_maps:
                        np.testing.assert_array_equal(saliency, adversarial_map.numpy())
                    order = np.load(sample_dir / "fixed_salient_order.npy")
                    expected = (np.arange(224 * 224)[::-1] if source == "clean"
                                else np.random.default_rng(7).permutation(224 * 224))
                    np.testing.assert_array_equal(order[0], expected)
                    image_file = "adversarial_image_del_ins.png" if mode == "del+ins" else f"adversarial_image_{mode}.png"
                    self.assertTrue((sample_dir / image_file).is_file())
                    for result in payload["rerun"].values():
                        self.assertEqual(len(result["curve"]), 3)
                        self.assertTrue(np.isfinite(result["auc"]))


class StrategyLauncherTests(unittest.TestCase):
    def test_forwarding_seeds_and_deduplication(self):
        argv = ["strategies", "--mask-sources", "clean", "random", "clean",
                "--seeds", "7", "42", "7", "--mode", "ins", "--output-dir", "out"]
        with patch.object(sys, "argv", argv), patch.object(subprocess, "run") as run:
            strategies.main()
        self.assertEqual(run.call_count, 3)
        for call, source, seed in zip(run.call_args_list, ("clean", "random", "random"), (None, 7, 42)):
            command = call.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertIn("--mode", command)
            self.assertEqual(command[command.index("--mask-source") + 1], source)
            self.assertTrue(call.kwargs["check"])
            if seed is not None:
                self.assertEqual(command[command.index("--seed") + 1], str(seed))

    def test_child_failure_is_not_silenced(self):
        with patch.object(sys, "argv", ["strategies"]), \
                patch.object(subprocess, "run", side_effect=subprocess.CalledProcessError(1, "runner")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                strategies.main()
            self.assertEqual(run.call_count, 1)

    def test_invalid_options_do_not_launch(self):
        for options in (["--seeds", "-1"], ["--mask-source=random"], ["--seed", "2"]):
            with self.subTest(options=options), patch.object(sys, "argv", ["strategies", *options]), \
                    patch.object(subprocess, "run") as run:
                with self.assertRaises(SystemExit):
                    strategies.main()
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
