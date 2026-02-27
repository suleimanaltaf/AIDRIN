"""Command-line interface for AIDRIN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .ingestion import load_custom_loader
from .inspector import AIDRINInspector
from .metrics import DEFAULT_METRICS


def _parse_loader_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_loader_args(values: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    if not values:
        return parsed
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid loader arg '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"Invalid loader arg '{item}'. Key cannot be empty.")
        parsed[key] = _parse_loader_value(value)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aidrin",
        description="AIDRIN: AI Data Readiness Inspector",
    )
    parser.add_argument("source", help="Dataset path or source identifier.")
    parser.add_argument(
        "--format",
        dest="dataset_format",
        choices=["csv", "json", "parquet", "hdf5", "zarr", "root"],
        help="Dataset format override; otherwise inferred from extension.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=list(DEFAULT_METRICS),
        help="Metric to run. Can be repeated. Defaults to all metrics.",
    )
    parser.add_argument("--target", dest="target_column", help="Target column for imbalance/fairness/importance.")
    parser.add_argument(
        "--sensitive",
        action="append",
        dest="sensitive_columns",
        help="Sensitive column for fairness metrics. Can be repeated.",
    )
    parser.add_argument(
        "--custom-loader",
        help="Custom loader in module:function format.",
    )
    parser.add_argument(
        "--loader-arg",
        action="append",
        help="Loader argument as key=value. Can be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON report file path. Defaults to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        custom_loader = load_custom_loader(args.custom_loader) if args.custom_loader else None
        loader_kwargs = _parse_loader_args(args.loader_arg)

        inspector = AIDRINInspector(metrics=args.metric)
        result = inspector.inspect(
            source=args.source,
            dataset_format=args.dataset_format,
            target_column=args.target_column,
            sensitive_columns=args.sensitive_columns or [],
            custom_loader=custom_loader,
            loader_kwargs=loader_kwargs,
        )
        payload = result.to_json(indent=2)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload)

        return 0
    except (ValueError, KeyError, TypeError, ImportError, ModuleNotFoundError, OSError) as exc:
        print(f"aidrin: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
