"""Reusable metric formatting and serialization helpers for evaluation results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping

logger = logging.getLogger(__name__)


def format_metric(name: str, value: Any, *, precision: int = 4) -> str:
    """Format a metric value for display.

    Args:
        name: Metric name to display.
        value: Metric value to format.
        precision: Number of decimal places to use for numeric values.

    Returns:
        A human-readable metric string.
    """
    if isinstance(value, (int, float)):
        return f"{name}: {float(value):.{precision}f}"
    return f"{name}: {value}"


def print_metrics(metrics: Mapping[str, Any]) -> None:
    """Print evaluation metrics in a readable format.

    Args:
        metrics: Mapping of metric names to values.
    """
    print("Evaluation summary")
    print("=" * 40)
    for key in ("precision", "recall", "map50", "map50_95", "num_images", "evaluation_time_seconds"):
        if key in metrics:
            print(format_metric(key.replace("_", " "), metrics[key]))


def save_metrics_json(output_path: str | Path, metrics: Mapping[str, Any]) -> Path:
    """Persist evaluation metrics as JSON.

    Args:
        output_path: Destination file path.
        metrics: Mapping of metrics to store.

    Returns:
        The written JSON file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(metrics), handle, indent=2)
    logger.info("Saved metrics JSON: %s", output_path)
    return output_path


def save_metrics_txt(output_path: str | Path, metrics: Mapping[str, Any]) -> Path:
    """Persist evaluation metrics as a text report.

    Args:
        output_path: Destination file path.
        metrics: Mapping of metrics to store.

    Returns:
        The written text file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Evaluation summary", "=" * 40]
    for key in ("precision", "recall", "map50", "map50_95", "num_images", "evaluation_time_seconds"):
        if key in metrics:
            lines.append(format_metric(key.replace("_", " "), metrics[key]))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved metrics text report: %s", output_path)
    return output_path
