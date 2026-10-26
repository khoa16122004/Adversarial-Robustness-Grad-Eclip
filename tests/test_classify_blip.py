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


class FakeBLIPClassifier(nn.Module):
    def forward(self, images):
        logits = torch.tensor([2.0, 0.0], device=images.device)
        return logits.expand(images.shape[0], -1)


class BLIPClassificationTests(unittest.TestCase):
    def test_blip_writes_attack_ready_correct_sample_json(self):
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
            blip_root = root / "BLIP"
            (blip_root / "models").mkdir(parents=True)
            (blip_root / "models" / "blip_itm.py").touch()
            checkpoint = root / "blip.pth"
            checkpoint.touch()
            summary_path = root / "summary.json"
            samples_path = root / "blip_100.json"

            adapter = type(
                "FakeAdapter",
                (),
                {"preprocess": staticmethod(lambda image: torch.zeros(3, 4, 4))},
            )()
            argv = [
                "classify.py",
                "--model", "blip",
                "--data-path", str(data_dir),
                "--index-json", str(index_path),
                "--blip-root", str(blip_root),
                "--blip-checkpoint", str(checkpoint),
                "--device", "cpu",
                "--max-correct-samples", "1",
                "--output", str(summary_path),
                "--sample-output", str(samples_path),
            ]
            with patch.object(sys, "argv", argv), \
                    patch("blip_attack_adapter.BLIPAttackAdapter", return_value=adapter), \
                    patch(
                        "blip_attack_adapter.build_zero_shot_blip_classifier",
                        return_value=FakeBLIPClassifier(),
                    ):
                classify.main()

            samples = json.loads(samples_path.read_text(encoding="utf-8"))
            self.assertEqual(samples, {"n0": "n0/first.png"})
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["model"], "BLIP-ViT-base")
            self.assertEqual(summary["num_correct"], 1)
            self.assertEqual(summary["num_total_processed"], 1)


if __name__ == "__main__":
    unittest.main()
