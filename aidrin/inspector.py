"""High-level orchestration API for running AIDRIN inspections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ingestion import infer_dataset_format, load_dataset
from .metrics import DEFAULT_METRICS, compute_metrics, dataset_summary


@dataclass(slots=True)
class InspectionResult:
    source: str
    dataset_format: str
    summary: dict[str, Any]
    metrics: dict[str, Any]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dataset_format": self.dataset_format,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "metrics": self.metrics,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class AIDRINInspector:
    """Runs the end-to-end AIDRIN readiness assessment workflow."""

    def __init__(self, metrics: tuple[str, ...] | list[str] | None = None) -> None:
        self.metrics = tuple(metrics) if metrics else DEFAULT_METRICS

    def inspect(
        self,
        source: str | Path,
        dataset_format: str | None = None,
        target_column: str | None = None,
        sensitive_columns: list[str] | None = None,
        custom_loader: Any | None = None,
        loader_kwargs: dict[str, Any] | None = None,
    ) -> InspectionResult:
        kwargs = loader_kwargs or {}
        frame = load_dataset(
            source=source,
            dataset_format=dataset_format,
            custom_loader=custom_loader,
            **kwargs,
        )
        resolved_format = "custom" if custom_loader else (dataset_format or infer_dataset_format(source))

        summary = dataset_summary(frame)
        metric_report = compute_metrics(
            frame,
            metrics=self.metrics,
            target_column=target_column,
            sensitive_columns=sensitive_columns or [],
        )

        return InspectionResult(
            source=str(source),
            dataset_format=str(resolved_format),
            summary=summary,
            metrics=metric_report,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
