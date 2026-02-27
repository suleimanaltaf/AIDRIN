from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aidrin.ingestion import DatasetLoaderRegistry, infer_dataset_format, load_dataset

try:
    import h5py  # noqa: F401

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import zarr  # noqa: F401

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

try:
    import uproot  # noqa: F401

    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False


class TestIngestion(unittest.TestCase):
    def test_load_csv_and_infer_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.csv"
            pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(path, index=False)

            self.assertEqual(infer_dataset_format(path), "csv")
            loaded = load_dataset(path)
            self.assertEqual(loaded.shape, (2, 2))
            self.assertListEqual(list(loaded.columns), ["a", "b"])

    @unittest.skipUnless(HAS_H5PY, "h5py not installed")
    def test_load_hdf5(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.h5"
            records = np.array([(1, 2.5), (2, 3.5)], dtype=[("id", "i4"), ("score", "f4")])
            with h5py.File(path, "w") as handle:
                handle.create_dataset("table", data=records)

            loaded = load_dataset(path, dataset_format="hdf5", key="table")
            self.assertEqual(loaded.shape[0], 2)
            self.assertIn("id", loaded.columns)
            self.assertIn("score", loaded.columns)

    @unittest.skipUnless(HAS_ZARR, "zarr not installed")
    def test_load_zarr(self) -> None:
        import zarr

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.zarr"
            array = np.array([[1, 10], [2, 20], [3, 30]])
            zarr.save(str(path), array)

            loaded = load_dataset(path, dataset_format="zarr")
            self.assertEqual(loaded.shape, (3, 2))

    @unittest.skipUnless(HAS_UPROOT, "uproot not installed")
    def test_load_root(self) -> None:
        import uproot

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.root"
            with uproot.recreate(path) as root_file:
                root_file["events"] = {
                    "feature_a": np.array([1, 2, 3]),
                    "feature_b": np.array([0.1, 0.2, 0.3]),
                }

            loaded = load_dataset(path, dataset_format="root", tree_name="events")
            self.assertEqual(loaded.shape[0], 3)
            self.assertIn("feature_a", loaded.columns)

    def test_custom_loader_and_registry(self) -> None:
        registry = DatasetLoaderRegistry()
        registry.register("mock", lambda _src, **_kwargs: pd.DataFrame({"x": [1], "y": [2]}))
        loaded = load_dataset("does-not-matter.mock", dataset_format="mock", registry=registry)
        self.assertEqual(loaded.shape, (1, 2))

        def custom_loader(_src: str, value: int = 1) -> pd.DataFrame:
            return pd.DataFrame({"value": [value]})

        loaded_custom = load_dataset("ignored", custom_loader=custom_loader, value=42)
        self.assertEqual(int(loaded_custom["value"].iloc[0]), 42)


if __name__ == "__main__":
    unittest.main()
