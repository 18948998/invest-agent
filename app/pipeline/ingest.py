"""Run data-source ingestion: fetch, normalize, validate, persist."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config.loader import load_fundamental_field_specs
from app.data.normalizer import normalize_records
from app.data.validator import ValidationResult, validate_records
from app.services.fundamental_data import fetch_raw_dataset
from app.storage.sqlite_repo import save_dataset


@dataclass(frozen=True, slots=True)
class DatasetIngestSummary:
    dataset_id: str
    dataset_name_zh: str
    row_count: int
    validation: ValidationResult
    sample_row: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class IngestRunSummary:
    db_path: Path
    source: str
    dataset_summaries: list[DatasetIngestSummary]


def run_fundamental_ingest(
    symbols: list[str],
    source: str,
    field_config_dir: Path,
    db_path: Path,
) -> IngestRunSummary:
    specs = load_fundamental_field_specs(field_config_dir)
    summaries: list[DatasetIngestSummary] = []

    for spec in specs:
        raw_records = fetch_raw_dataset(
            dataset_id=spec.meta.dataset_id,
            symbols=symbols,
            source=source,
        )
        normalized = normalize_records(spec, raw_records, source=source)
        validation = validate_records(spec, normalized)
        row_count = save_dataset(db_path=db_path, dataset_spec=spec, rows=normalized)

        summaries.append(
            DatasetIngestSummary(
                dataset_id=spec.meta.dataset_id,
                dataset_name_zh=spec.meta.dataset_name_zh,
                row_count=row_count,
                validation=validation,
                sample_row=normalized[0] if normalized else None,
            )
        )

    return IngestRunSummary(db_path=db_path, source=source, dataset_summaries=summaries)

