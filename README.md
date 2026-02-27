# AIDRIN Prototype (AI Data Readiness Inspector)

This repository now contains a working AIDRIN prototype focused on the scope from `ProjectDescription.md`:

- support for additional file formats (**HDF5**, **Zarr**, **ROOT**), and
- support for **custom data ingestion** mechanisms.

It includes:

- a modular ingestion registry (`aidrin.ingestion`)
- core data-readiness metrics inspired by the AIDRIN papers (`aidrin.metrics`)
- an inspector orchestration API (`aidrin.inspector`)
- a CLI for generating JSON reports (`aidrin.cli`)

## Quick start

```bash
python -m pip install -e ".[io,ml]"
aidrin /path/to/data.csv --target label --sensitive sex --output report.json
```

## Supported data formats

Built-in loaders:

- CSV (`.csv`)
- JSON (`.json`, `.jsonl`)
- Parquet (`.parquet`, `.pq`)
- HDF5 (`.h5`, `.hdf`, `.hdf5`)
- Zarr (`.zarr`)
- ROOT (`.root`)

You can also force format selection with `--format`.

## Custom ingestion

Two options are available:

1. Programmatic custom loader:

```python
import pandas as pd
from aidrin.inspector import AIDRINInspector

def my_loader(source: str, **kwargs) -> pd.DataFrame:
    # Implement domain-specific loading logic here
    return pd.DataFrame({"feature": [1, 2], "target": [0, 1]})

result = AIDRINInspector().inspect(
    "ignored-by-custom-loader",
    custom_loader=my_loader,
    target_column="target",
)
print(result.to_json())
```

2. CLI custom loader import path (`module:function`):

```bash
aidrin ignored --custom-loader my_pkg.my_module:my_loader --target target
```

## CLI usage

```bash
aidrin SOURCE_PATH \
  [--format {csv,json,parquet,hdf5,zarr,root}] \
  [--metric METRIC ...] \
  [--target TARGET_COLUMN] \
  [--sensitive SENSITIVE_COLUMN ...] \
  [--custom-loader module:function] \
  [--loader-arg key=value ...] \
  [--output REPORT.json]
```

Example (HDF5):

```bash
aidrin data.h5 --format hdf5 --loader-arg key=dataset_name --output report.json
```

Example (ROOT):

```bash
aidrin events.root --format root --loader-arg tree_name=events --output report.json
```

## Implemented metrics

- completeness
- duplicates
- outliers (IQR-based)
- correlations (Pearson + Theil's U for categorical features)
- class imbalance
- fairness (representation rate, statistical rate, target standard deviation)
- feature importance (random forest importances, if scikit-learn is installed)

## Tests

Run:

```bash
python -m unittest discover -s tests -v
```
