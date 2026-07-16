"""Transform raw source records to canonical rows based on YAML field specs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.config.schema import DatasetSpec, FieldSpec


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("/", "-").replace(".", "-")
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(normalized, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text


def _coerce_value(field: FieldSpec, value: Any) -> Any:
    if field.dtype == "float":
        return _to_float(value)
    if field.dtype == "date":
        return _to_date(value)
    if value is None:
        return None
    return str(value).strip()


def normalize_records(
    dataset_spec: DatasetSpec,
    raw_records: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_row in raw_records:
        row: dict[str, Any] = {}
        for field in dataset_spec.fields:
            source_key = field.source_keys.get(source) or field.source_keys.get("akshare")
            raw_value = raw_row.get(source_key) if source_key else None
            row[field.name] = _coerce_value(field, raw_value)
        normalized.append(row)
    return normalized



