import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "export_blip_latex",
    Path(__file__).resolve().parents[1] / "script_eval" / "export_blip_latex.py",
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportBLIPLatexTests(unittest.TestCase):
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
