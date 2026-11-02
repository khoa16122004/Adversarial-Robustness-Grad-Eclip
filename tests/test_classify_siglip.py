import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

import classify


class FakeSigLIPClassifier(nn.Module):
    def forward(self, images):
        logits = torch.tensor([2.0, 0.0], device=images.device)
        return logits.expand(images.shape[0], -1)


class SigLIPClassificationTests(unittest.TestCase):
    def test_siglip_writes_attack_ready_samples_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            (data_dir / "n0").mkdir(parents=True)
            (data_dir / "n1").mkdir()
            Image.new("RGB", (8, 8)).save(data_dir / "n0" / "first.png")
            Image.new("RGB", (8, 8)).save(data_dir / "n1" / "second.png")
            index_path = root / "labels.json"
            index_path.write_text(
                json.dumps({"n0": [0, "cat"], "n1": [1, "dog"]}),
                encoding="utf-8",
            )
            summary_path = root / "summary.json"
            samples_path = root / "siglip_samples.json"
            adapter = type(
                "FakeAdapter",
                (),
                {"preprocess": staticmethod(lambda image: torch.zeros(3, 4, 4))},
            )()
            argv = [
                "classify.py",
                "--model", "siglip",
                "--siglip-model", "test/siglip",
                "--data-path", str(data_dir),
                "--index-json", str(index_path),
                "--device", "cpu",
                "--max-correct-samples", "1",
                "--output", str(summary_path),
                "--sample-output", str(samples_path),
            ]
            with patch.object(sys, "argv", argv), \
                    patch(
                        "siglip_attack_adapter.SigLIPAttackAdapter",
                        return_value=adapter,
                    ), \
                    patch(
                        "siglip_attack_adapter.build_zero_shot_siglip_classifier",
                        return_value=FakeSigLIPClassifier(),
                    ):
                classify.main()

            samples = json.loads(samples_path.read_text(encoding="utf-8"))
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples["n0"], "n0/first.png")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["model"], "SigLIP-test/siglip")
            self.assertEqual(summary["prompt_template"], "a photo of {}")
            self.assertEqual(summary["num_correct"], 1)
            self.assertEqual(summary["num_total_processed"], 2)


if __name__ == "__main__":
    unittest.main()
