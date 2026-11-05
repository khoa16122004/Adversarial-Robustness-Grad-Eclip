import argparse
import ast
import contextlib
import io
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class StandaloneBLIPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "XShift_score_BLIP.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        parser_function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
        )
        namespace = {
            "argparse": argparse, "Path": Path, "math": math,
            "METHODS": ("eclip", "game", "gradcam", "maskclip", "rise"),
            "__doc__": "Test CLI without importing GPU dependencies",
        }
        exec(compile(ast.Module(body=[parser_function], type_ignores=[]), str(ROOT / "XShift_score_BLIP.py"), "exec"), namespace)
        cls.parse_args = staticmethod(namespace["parse_args"])

    def parse(self, *arguments):
        with patch.object(sys, "argv", ["XShift_score_BLIP.py", *arguments]):
            return self.parse_args()

    def test_standalone_inputs_without_existing_results(self):
        args = self.parse(
            "--output-dir", "new-output", "--checkpoint", "model.pth",
            "--img-dir", "images", "--sample-path", "samples.json",
            "--classnames-path", "cub.json",
        )
        self.assertIsNone(args.run_config)
        self.assertEqual(args.eps, 16)
        self.assertEqual(args.last_heads, "1")
        self.assertEqual(len(args.methods), 5)

    def test_existing_config_remains_supported(self):
        args = self.parse("--output-dir", "output", "--run-config", "run_config.json")
        self.assertEqual(args.run_config, Path("run_config.json"))

    def test_missing_standalone_inputs_fail(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse("--output-dir", "output")

    def test_launchers_have_no_attack_result_dependency(self):
        for name, output in (
            ("clean_blip_all.sh", "BLIP_result_clean"),
            ("xshift_blip_all.sh", "BLIP_result_xshift"),
        ):
            with self.subTest(name=name):
                source = (ROOT / "script_sh" / name).read_text(encoding="utf-8")
                self.assertNotIn("RESULT_ROOT", source)
                self.assertNotIn("--run-config", source)
                self.assertIn(f'output_root="${{OUTPUT_ROOT:-$repo_root/{output}}}"', source)
                self.assertIn('mkdir -p "$output_root"', source)
                self.assertIn("datasets=(ImageNet StandfordPet cub)", source)
                for value in (
                    "/datastore/quanphm/datasets/ImageNet/val",
                    "StandfordPet_Dataset", "cub_dataset",
                    "imgnet1k_label.json", "pet.json", "cub.json",
                ):
                    self.assertIn(value, source)


if __name__ == "__main__":
    unittest.main()
