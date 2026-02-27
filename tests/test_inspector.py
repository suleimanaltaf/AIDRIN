from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aidrin.cli import main as cli_main
from aidrin.inspector import AIDRINInspector


class TestInspector(unittest.TestCase):
    def _write_sample_csv(self, tmp_dir: str) -> Path:
        path = Path(tmp_dir) / "dataset.csv"
        frame = pd.DataFrame(
            {
                "age": [10, 12, 14, 100, 13],
                "income": [100, 120, 130, 900, 110],
                "group": ["A", "A", "B", "B", "A"],
                "target": [0, 0, 1, 1, 1],
            }
        )
        frame.to_csv(path, index=False)
        return path

    def test_inspect_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = self._write_sample_csv(tmp_dir)
            inspector = AIDRINInspector(
                metrics=(
                    "completeness",
                    "duplicates",
                    "outliers",
                    "class_imbalance",
                    "fairness",
                )
            )
            result = inspector.inspect(
                source,
                target_column="target",
                sensitive_columns=["group"],
            )

            payload = result.to_dict()
            self.assertEqual(payload["dataset_format"], "csv")
            self.assertIn("summary", payload)
            self.assertIn("metrics", payload)
            self.assertIn("fairness", payload["metrics"])
            self.assertIn("group", payload["metrics"]["fairness"])

    def test_cli_generates_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = self._write_sample_csv(tmp_dir)
            output = Path(tmp_dir) / "report.json"

            code = cli_main(
                [
                    str(source),
                    "--target",
                    "target",
                    "--sensitive",
                    "group",
                    "--metric",
                    "completeness",
                    "--metric",
                    "class_imbalance",
                    "--metric",
                    "fairness",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(output.exists())

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("metrics", report)
            self.assertIn("class_imbalance", report["metrics"])


if __name__ == "__main__":
    unittest.main()
