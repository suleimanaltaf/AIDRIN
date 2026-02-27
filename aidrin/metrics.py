"""Core data readiness metrics used by AIDRIN."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_METRICS = (
    "completeness",
    "duplicates",
    "outliers",
    "correlations",
    "class_imbalance",
    "fairness",
    "feature_importance",
)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        if math.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _series_to_dict(series: pd.Series) -> dict[str, Any]:
    return {str(k): _normalize_value(v) for k, v in series.items()}


def _matrix_to_dict(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for row_label, row in frame.iterrows():
        matrix[str(row_label)] = {str(col): _normalize_value(val) for col, val in row.items()}
    return matrix


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
    categorical_columns = [c for c in df.columns if c not in numeric_columns]
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows_total": int(df.duplicated().sum()),
    }


def completeness(df: pd.DataFrame) -> dict[str, Any]:
    per_column = df.notna().mean()
    return {
        "overall_completeness": float(per_column.mean()),
        "per_column_completeness": _series_to_dict(per_column),
    }


def duplicates(df: pd.DataFrame) -> dict[str, Any]:
    duplicate_count = int(df.duplicated().sum())
    return {
        "duplicate_rows": duplicate_count,
        "duplicate_ratio": float(duplicate_count / len(df)) if len(df) else 0.0,
    }


def outliers_iqr(df: pd.DataFrame, k: float = 1.5) -> dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {
            "outlier_ratio_overall": 0.0,
            "per_column_outlier_ratio": {},
            "k": float(k),
        }

    ratios: dict[str, float] = {}
    total_outliers = 0
    total_non_null = 0

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            ratios[str(column)] = 0.0
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_mask = pd.Series(False, index=series.index)
        else:
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            outlier_mask = (series < lower) | (series > upper)

        outlier_count = int(outlier_mask.sum())
        ratios[str(column)] = float(outlier_count / len(series))
        total_outliers += outlier_count
        total_non_null += int(len(series))

    overall = float(total_outliers / total_non_null) if total_non_null else 0.0
    return {
        "outlier_ratio_overall": overall,
        "per_column_outlier_ratio": ratios,
        "k": float(k),
    }


def _entropy(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True, dropna=False)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _conditional_entropy(x: pd.Series, y: pd.Series) -> float:
    contingency = pd.crosstab(y, x, dropna=False)
    total = contingency.to_numpy().sum()
    if total == 0:
        return 0.0

    cond_entropy = 0.0
    for _, row in contingency.iterrows():
        row_total = row.sum()
        if row_total == 0:
            continue
        p_y = row_total / total
        p_x_given_y = row / row_total
        p_x_given_y = p_x_given_y[p_x_given_y > 0]
        cond_entropy += float(p_y * (-(p_x_given_y * np.log2(p_x_given_y)).sum()))
    return float(cond_entropy)


def _theils_u(x: pd.Series, y: pd.Series) -> float:
    x_norm = x.astype("string").fillna("<NA>")
    y_norm = y.astype("string").fillna("<NA>")
    h_x = _entropy(x_norm)
    if h_x == 0:
        return 1.0
    value = (h_x - _conditional_entropy(x_norm, y_norm)) / h_x
    return float(max(0.0, min(1.0, value)))


def correlations(df: pd.DataFrame) -> dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_corr = numeric_df.corr(method="pearson") if not numeric_df.empty else pd.DataFrame()

    categorical_columns = [c for c in df.columns if c not in numeric_df.columns]
    categorical_corr: dict[str, dict[str, float]] = {}
    for col_a in categorical_columns:
        categorical_corr[str(col_a)] = {}
        for col_b in categorical_columns:
            categorical_corr[str(col_a)][str(col_b)] = _theils_u(df[col_a], df[col_b])

    return {
        "numeric_pearson": _matrix_to_dict(numeric_corr) if not numeric_corr.empty else {},
        "categorical_theils_u": categorical_corr,
    }


def class_imbalance(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    if target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")

    target = df[target_column].astype("string").fillna("<NA>")
    counts = target.value_counts(dropna=False)
    total = int(counts.sum())
    distribution = counts / total if total else counts

    min_count = int(counts.min()) if not counts.empty else 0
    max_count = int(counts.max()) if not counts.empty else 0
    imbalance_ratio = float(max_count / min_count) if min_count > 0 else None

    probs = distribution.to_numpy(dtype=float)
    class_count = len(probs)
    balanced_prob = 1.0 / class_count if class_count else 0.0
    imbalance_degree = float(np.sqrt(np.sum((probs - balanced_prob) ** 2))) if class_count else 0.0

    return {
        "target_column": target_column,
        "class_counts": _series_to_dict(counts),
        "class_distribution": _series_to_dict(distribution),
        "imbalance_ratio": imbalance_ratio,
        "imbalance_degree": imbalance_degree,
    }


def fairness(
    df: pd.DataFrame,
    sensitive_columns: list[str],
    target_column: str | None = None,
) -> dict[str, Any]:
    missing = [c for c in sensitive_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Sensitive columns not found: {missing}")

    if target_column is not None and target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")

    result: dict[str, Any] = {}

    for sensitive in sensitive_columns:
        sensitive_series = df[sensitive].astype("string").fillna("<NA>")
        representation = sensitive_series.value_counts(normalize=True, dropna=False)
        sensitive_result: dict[str, Any] = {
            "representation_rate": _series_to_dict(representation),
        }

        if target_column is not None:
            target_series = df[target_column].astype("string").fillna("<NA>")
            crosstab = pd.crosstab(sensitive_series, target_series, normalize="index")

            statistical_rate: dict[str, dict[str, float]] = {}
            for group, row in crosstab.iterrows():
                statistical_rate[str(group)] = {str(k): float(v) for k, v in row.items()}

            tsd: dict[str, float] = {}
            for target_value in crosstab.columns:
                tsd[str(target_value)] = float(crosstab[target_value].std(ddof=0))

            sensitive_result["statistical_rate"] = statistical_rate
            sensitive_result["target_standard_deviation"] = tsd

        result[sensitive] = sensitive_result

    return result


def feature_importance(
    df: pd.DataFrame,
    target_column: str,
    top_k: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    if target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError as exc:
        raise ImportError(
            "Feature importance requires scikit-learn. Install with: pip install scikit-learn"
        ) from exc

    working_df = df.dropna(subset=[target_column]).copy()
    if working_df.empty:
        raise ValueError("Target column contains only missing values after filtering.")

    X = working_df.drop(columns=[target_column])
    if X.empty:
        raise ValueError("No features available after dropping target column.")
    X = pd.get_dummies(X, dummy_na=True).fillna(0)

    y_raw = working_df[target_column]
    if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique(dropna=True) > 10:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        y = y_raw.to_numpy()
        model_type = "regressor"
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        y = pd.factorize(y_raw.astype("string"))[0]
        model_type = "classifier"

    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top = importances.head(top_k)
    top_features = [{"feature": str(name), "importance": float(score)} for name, score in top.items()]

    return {
        "model_type": model_type,
        "top_features": top_features,
    }


def compute_metrics(
    df: pd.DataFrame,
    metrics: Iterable[str],
    target_column: str | None = None,
    sensitive_columns: list[str] | None = None,
    outlier_k: float = 1.5,
    feature_importance_top_k: int = 10,
) -> dict[str, Any]:
    selected = [m.lower() for m in metrics]
    results: dict[str, Any] = {}

    for metric in selected:
        if metric == "completeness":
            results[metric] = completeness(df)
        elif metric == "duplicates":
            results[metric] = duplicates(df)
        elif metric == "outliers":
            results[metric] = outliers_iqr(df, k=outlier_k)
        elif metric == "correlations":
            results[metric] = correlations(df)
        elif metric == "class_imbalance":
            if not target_column:
                results[metric] = {"skipped": "target_column is required"}
            else:
                results[metric] = class_imbalance(df, target_column=target_column)
        elif metric == "fairness":
            if not sensitive_columns:
                results[metric] = {"skipped": "sensitive_columns is required"}
            else:
                results[metric] = fairness(
                    df,
                    sensitive_columns=sensitive_columns,
                    target_column=target_column,
                )
        elif metric == "feature_importance":
            if not target_column:
                results[metric] = {"skipped": "target_column is required"}
            else:
                results[metric] = feature_importance(
                    df,
                    target_column=target_column,
                    top_k=feature_importance_top_k,
                )
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    return results
