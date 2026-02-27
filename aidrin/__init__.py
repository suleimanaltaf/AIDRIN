"""AIDRIN: AI Data Readiness Inspector prototype."""

from .ingestion import (
    DatasetLoaderRegistry,
    infer_dataset_format,
    load_custom_loader,
    load_dataset,
    register_loader,
)
from .inspector import AIDRINInspector, InspectionResult

__all__ = [
    "AIDRINInspector",
    "DatasetLoaderRegistry",
    "InspectionResult",
    "infer_dataset_format",
    "load_custom_loader",
    "load_dataset",
    "register_loader",
]
