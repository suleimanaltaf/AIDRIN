"""Dataset ingestion utilities with extensible loader registration."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LoaderFn = Callable[..., pd.DataFrame]


def _to_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame()
    if isinstance(data, Mapping):
        return pd.DataFrame(data)
    if isinstance(data, np.ndarray):
        if data.dtype.names:
            return pd.DataFrame.from_records(data)
        if data.ndim == 1:
            return pd.DataFrame({"value": data})
        if data.ndim == 2:
            return pd.DataFrame(data)
        raise ValueError(f"Unsupported ndarray shape for tabular conversion: {data.shape}")
    if hasattr(data, "to_pandas") and callable(data.to_pandas):
        converted = data.to_pandas()
        if isinstance(converted, pd.DataFrame):
            return converted
    raise TypeError(f"Unsupported loader output type: {type(data)!r}")


def load_csv(source: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(source, **kwargs)


def load_json(source: str | Path, **kwargs: Any) -> pd.DataFrame:
    source_path = str(source).lower()
    if source_path.endswith(".jsonl") and "lines" not in kwargs:
        kwargs["lines"] = True
    return pd.read_json(source, **kwargs)


def load_parquet(source: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_parquet(source, **kwargs)


def _first_hdf5_dataset_name(h5_file: Any) -> str | None:
    dataset_name: str | None = None

    def _visit(name: str, obj: Any) -> None:
        nonlocal dataset_name
        if dataset_name is None and obj.__class__.__name__ == "Dataset":
            dataset_name = name

    h5_file.visititems(_visit)
    return dataset_name


def load_hdf5(source: str | Path, key: str | None = None, **_: Any) -> pd.DataFrame:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("HDF5 support requires h5py. Install with: pip install h5py") from exc

    with h5py.File(source, "r") as h5_file:
        dataset_key = key or _first_hdf5_dataset_name(h5_file)
        if not dataset_key:
            raise ValueError("No dataset found in HDF5 file; provide --loader-arg key=<dataset_path>")
        if dataset_key not in h5_file:
            raise KeyError(f"HDF5 dataset key not found: {dataset_key}")
        data = h5_file[dataset_key][()]

    return _to_dataframe(data)


def _find_first_zarr_array(node: Any) -> Any:
    if hasattr(node, "shape") and hasattr(node, "dtype"):
        return node
    if hasattr(node, "keys"):
        for child_name in node.keys():
            found = _find_first_zarr_array(node[child_name])
            if found is not None:
                return found
    return None


def load_zarr(source: str | Path, key: str | None = None, **_: Any) -> pd.DataFrame:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("Zarr support requires zarr. Install with: pip install zarr") from exc

    store = zarr.open(str(source), mode="r")
    if key:
        if key not in store:
            raise KeyError(f"Zarr key not found: {key}")
        node = store[key]
    else:
        node = _find_first_zarr_array(store)
        if node is None:
            raise ValueError("No array found in Zarr store; provide --loader-arg key=<array_path>")

    data = node[...]
    return _to_dataframe(data)


def load_root(source: str | Path, tree_name: str | None = None, **_: Any) -> pd.DataFrame:
    try:
        import uproot
    except ImportError as exc:
        raise ImportError("ROOT support requires uproot. Install with: pip install uproot") from exc

    root_file = uproot.open(source)
    tree = None

    if tree_name:
        tree = root_file[tree_name]
    else:
        class_names = root_file.classnames()
        for key, class_name in class_names.items():
            if "TTree" in class_name:
                tree = root_file[key]
                break

    if tree is None:
        raise ValueError("No TTree found in ROOT file; provide --loader-arg tree_name=<tree>")

    data = tree.arrays(library="np")
    return _to_dataframe(data)


class DatasetLoaderRegistry:
    """Registry for mapping format names to loader callables."""

    _EXTENSION_MAP = {
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "json",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".h5": "hdf5",
        ".hdf": "hdf5",
        ".hdf5": "hdf5",
        ".zarr": "zarr",
        ".root": "root",
    }

    def __init__(self) -> None:
        self._loaders: dict[str, LoaderFn] = {}

    def register(self, format_name: str, loader: LoaderFn, overwrite: bool = False) -> None:
        normalized = format_name.strip().lower()
        if not normalized:
            raise ValueError("Format name cannot be empty.")
        if normalized in self._loaders and not overwrite:
            raise ValueError(f"Loader already registered for format '{normalized}'.")
        self._loaders[normalized] = loader

    def get(self, format_name: str) -> LoaderFn:
        normalized = format_name.strip().lower()
        if normalized not in self._loaders:
            raise KeyError(f"No loader registered for format '{normalized}'.")
        return self._loaders[normalized]

    def detect_format(self, source: str | Path) -> str:
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix in self._EXTENSION_MAP:
            return self._EXTENSION_MAP[suffix]
        raise ValueError(f"Unable to infer dataset format from file extension: '{path.name}'")


DEFAULT_REGISTRY = DatasetLoaderRegistry()


def _register_builtin_loaders(registry: DatasetLoaderRegistry) -> None:
    registry.register("csv", load_csv)
    registry.register("json", load_json)
    registry.register("parquet", load_parquet)
    registry.register("hdf5", load_hdf5)
    registry.register("zarr", load_zarr)
    registry.register("root", load_root)


_register_builtin_loaders(DEFAULT_REGISTRY)


def register_loader(format_name: str, loader: LoaderFn, overwrite: bool = False) -> None:
    """Register a custom loader in the global registry."""
    DEFAULT_REGISTRY.register(format_name, loader, overwrite=overwrite)


def infer_dataset_format(source: str | Path) -> str:
    """Infer dataset format from source extension."""
    return DEFAULT_REGISTRY.detect_format(source)


def load_custom_loader(import_path: str) -> LoaderFn:
    """
    Load a custom loader callable from `module:function` import path.
    """
    if ":" not in import_path:
        raise ValueError("Custom loader path must be in 'module:function' format.")
    module_name, function_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    loader = getattr(module, function_name)
    if not callable(loader):
        raise TypeError(f"Custom loader '{import_path}' is not callable.")
    return loader


def load_dataset(
    source: str | Path,
    dataset_format: str | None = None,
    custom_loader: LoaderFn | None = None,
    registry: DatasetLoaderRegistry | None = None,
    **loader_kwargs: Any,
) -> pd.DataFrame:
    """
    Load a dataset as pandas DataFrame using built-in or custom loader.
    """
    active_registry = registry or DEFAULT_REGISTRY

    if custom_loader is not None:
        loaded = custom_loader(source, **loader_kwargs)
        return _to_dataframe(loaded)

    selected_format = dataset_format.lower() if dataset_format else active_registry.detect_format(source)
    loader = active_registry.get(selected_format)
    loaded = loader(source, **loader_kwargs)
    return _to_dataframe(loaded)
