import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from generate_label_index import build_label_index


class GenerateLabelIndexTests(unittest.TestCase):
    def test_cli_writes_sorted_mapping_with_original_folder_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "dataset"
            data_path.mkdir()
            for name in ("n02090379-redbone", "n02088094-Afghan_hound"):
                (data_path / name).mkdir()
            (data_path / ".cache").mkdir()
            (data_path / "notes.txt").touch()
            output = root / "metadata" / "labels.json"
            script = Path(__file__).resolve().parents[1] / "generate_label_index.py"

            subprocess.run(
                [
                    sys.executable, str(script), "--data-path", str(data_path),
                    "--output", str(output),
                ],
                check=True, capture_output=True, text=True,
            )

            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {
                    "n02088094-Afghan_hound": [0, "Afghan hound"],
                    "n02090379-redbone": [1, "redbone"],
                },
            )

    def test_missing_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(NotADirectoryError):
                build_label_index(Path(directory) / "missing")

    def test_empty_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No class subfolders"):
                build_label_index(Path(directory))

    def test_empty_class_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "n02090379-").mkdir()
            with self.assertRaisesRegex(ValueError, "Cannot infer class name"):
                build_label_index(root)


if __name__ == "__main__":
    unittest.main()
