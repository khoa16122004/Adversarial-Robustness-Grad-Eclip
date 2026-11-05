import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "export_blip_latex",
    Path(__file__).resolve().parents[1] / "script_eval" / "export_blip_latex.py",
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportBLIPLatexTests(unittest.TestCase):
    def test_clean_only_table_uses_clean_imd_and_road(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "ImageNet" / "clean" / "eclip" / "sample"
            sample.mkdir(parents=True)
            (sample / "curve_information.json").write_text(json.dumps({
                "attack_mode": "clean",
                "rerun": {
                    "ins": {"curve": [0., 1.]},
                    "del": {"curve": [1., 0.]},
                },
            }), encoding="utf-8")
            (sample / "road_deletion_information.json").write_text(json.dumps({
                "deletion_fair_auc": 0.625,
            }), encoding="utf-8")

            means = exporter.collect_metrics(
                root, "0", expected_attacks=("Without Attack",)
            )
            table = exporter.build_clean_table(means)

            self.assertAlmostEqual(
                means[("ImageNet", "GradECLIP", "Without Attack", "IMD")], 0.
            )
            self.assertAlmostEqual(
                means[("ImageNet", "GradECLIP", "Without Attack", "ROAD")], 0.625
            )
            self.assertIn("GradECLIP & Without Attack & 0.0000 & -- & 0.6250", table)
            self.assertEqual(table.count("Without Attack"), 5)

    def test_spearman_from_clean_and_attack_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "ImageNet" / "clean" / "eclip" / "sample" / "saliency"
            clean.mkdir(parents=True)
            clean_values = np.array([[1., 1.], [2., 3.]])
            np.save(clean / "eclip_saliency.npy", clean_values)
            for mode, prefix, values in (
                ("ins", "ins", clean_values),
                ("del", "del", -clean_values),
                ("del+ins", "joint", np.ones((2, 2))),
            ):
                sample = root / "ImageNet" / "eps_2" / "eclip" / mode / "sample"
                sample.mkdir(parents=True)
                (sample / "curve_information.json").write_text(
                    json.dumps({"attack_mode": mode}), encoding="utf-8"
                )
                np.save(sample / f"{prefix}_del_adv_eclip_saliency.npy", values)
            other = root / "ImageNet" / "eps_8"
            other.mkdir()
            (other / "curve_information.json").write_text("not JSON", encoding="utf-8")
            means = exporter.collect_metrics(root, "2")
            self.assertAlmostEqual(means[("ImageNet", "GradECLIP", "IOA", "SRC")], 1)
            self.assertAlmostEqual(means[("ImageNet", "GradECLIP", "DOA", "SRC")], -1)
            self.assertNotIn(("ImageNet", "GradECLIP", "IOA+DOA", "SRC"), means)

    def test_separate_clean_root_and_shape_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_root = root / "clean_results"
            sample = clean_root / "CUB" / "clean" / "game" / "sample"
            sample.mkdir(parents=True)
            clean_path = sample / "game_saliency.npy"
            np.save(clean_path, np.array([[1., 2.], [3., 4.]]))
            self.assertEqual(
                exporter.index_clean_maps(clean_root, "16")[("CUB", "GAME", "sample")],
                clean_path,
            )
            attack_path = root / "attack.npy"
            np.save(attack_path, np.arange(6).reshape(2, 3))
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                exporter.saliency_spearman(clean_path, attack_path)
            np.save(attack_path, np.array([[1., 2.], [3., float("nan")]]))
            with self.assertRaisesRegex(ValueError, "finite 2D"):
                exporter.saliency_spearman(clean_path, attack_path)

    def test_attack_separation_and_missing_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, insertion in (("ins", [0, 1, 1]), ("del", [0, 0, 1]), ("del+ins", [0, 1, 0])):
                path = root / "ImageNet" / "eps_2" / "eclip" / mode / "sample"
                path.mkdir(parents=True)
                (path / "curve_information.json").write_text(json.dumps({
                    "attack_mode": mode,
                    "rerun": {"ins": {"curve": insertion}, "del": {"curve": [1, 0, 0]}},
                }), encoding="utf-8")
            other = root / "ImageNet" / "eps_8" / "eclip" / "ins" / "sample"
            other.mkdir(parents=True)
            (other / "curve_information.json").write_text(json.dumps({
                "rerun": {"ins": {"curve": [1, 1]}, "del": {"curve": [0, 1]}}
            }), encoding="utf-8")
            means = exporter.collect_metrics(root, "2")
            self.assertAlmostEqual(means[("ImageNet", "GradECLIP", "IOA", "IMD")], 0.5)
            self.assertAlmostEqual(means[("ImageNet", "GradECLIP", "DOA", "IMD")], 0.0)
            self.assertAlmostEqual(means[("ImageNet", "GradECLIP", "IOA+DOA", "IMD")], 0.25)
            self.assertFalse(any(key[3] == "ROAD" for key in means))
            table = exporter.build_table(means, "2")
            self.assertEqual(table.count(r"\multirow[c]{5}"), 5)
            self.assertIn("StanfordPets", table)
            self.assertEqual(table.count(r"\textbf{IOA+DOA}"), 5)
            self.assertEqual(exporter.render_cell(means, "CUB", "GAME", "IOA", "IMD"), "")

    def test_highlight_and_delta(self):
        means = {
            ("ImageNet", "GAME", "Without Attack", "IMD"): 0.7,
            ("ImageNet", "GAME", "IOA", "IMD"): 0.2,
        }
        cell = exporter.render_cell(means, "ImageNet", "GAME", "IOA", "IMD")
        self.assertIn(r"\textcolor{blue!60}", cell)
        self.assertIn(r"\textcolor{red}{-.500}", cell)

    def test_invalid_curve_and_missing_results_fail(self):
        with self.assertRaises(ValueError):
            exporter.fair_auc([0, float("nan")])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                exporter.collect_metrics(Path(directory), "2")


if __name__ == "__main__":
    unittest.main()
