import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from script_eval import export_blip_average as exporter


class BLIPAverageTests(unittest.TestCase):
    def test_equal_explainer_weight_and_table_shape(self):
        means = {
            ("ImageNet", "GradECLIP", "IOA", "IMD"): .2,
            ("ImageNet", "GAME", "IOA", "IMD"): .8,
            ("ImageNet", "GAME", "IOA", "DEL_mean"): .7,
        }
        averages, counts = exporter.aggregate(means)
        self.assertAlmostEqual(averages[("ImageNet", "IOA", "IMD")], .5)
        self.assertEqual(len(counts[("ImageNet", "IOA", "IMD")]), 2)
        self.assertEqual(averages[("ImageNet", "IOA", "DEL_mean")], .7)
        table = exporter.build_table(averages, "16")
        self.assertIn(r"\multicolumn{13}{c}{\textbf{BLIP}}", table)
        for attack in exporter.ATTACKS:
            row = next(line for line in table.splitlines() if line.startswith(attack + " &"))
            self.assertEqual(row.count("&"), 12)
        self.assertNotIn("SigLIP", table)

    def test_distinct_roots_output_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            clean_root, attack_root, output = base / "clean_results", base / "attack_results", base / "tables"
            for root, parts, attack in (
                (clean_root, ("ImageNet", "clean", "eclip", "sample"), "clean"),
                (attack_root, ("ImageNet", "eps_16", "eclip", "ins", "sample"), "ins"),
            ):
                folder = root.joinpath(*parts)
                folder.mkdir(parents=True)
                (folder / "curve_information.json").write_text(json.dumps({
                    "attack_mode": attack,
                    "rerun": {"ins": {"curve": [0., 1., 1.]}, "del": {"curve": [1., 0., 0.]}},
                }))
                for name, imputer, filename, value in (
                    ("road_deletion", "linear", "road_deletion_information.json", .6),
                    ("DEL_avg_channel", "fixed", "del_avg_channel_information.json", .7),
                ):
                    (folder / filename).write_text(json.dumps({
                        "metric": name, "imputer": imputer, "deletion_fair_auc": value,
                    }))
            argv = ["exporter", "--clean-root", str(clean_root), "--attack-roots", str(attack_root),
                    "--epsilon", "16", "--output-dir", str(output)]
            with patch("sys.argv", argv):
                exporter.main()
            for suffix in ("tex", "csv", "json"):
                self.assertTrue((output / f"BLIP_result_avg_eps_16.{suffix}").is_file())
            data = json.loads((output / "BLIP_result_avg_eps_16.json").read_text())
            cell = next(row for row in data["cells"] if
                        (row["dataset"], row["attack"], row["metric"]) == ("ImageNet", "IOA", "DEL_mean"))
            self.assertAlmostEqual(cell["mean"], .7)
            self.assertEqual(cell["explainer_count"], 1)


if __name__ == "__main__":
    unittest.main()
