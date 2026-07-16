"""Validate normalized records against required field constraints."""

from __future__ import annotations

from dataclasses import dataclass

from app.config.schema import DatasetSpec


@dataclass(frozen=True, slots=True)
class ValidationResult:
    dataset_id: str
    total_rows: int
    required_violations: dict[str, int]

    @property
    def is_valid(self) -> bool:
        return all(count == 0 for count in self.required_violations.values())


def validate_records(dataset_spec: DatasetSpec, rows: list[dict[str, object]]) -> ValidationResult:
    violations: dict[str, int] = {}
    required_fields = [field for field in dataset_spec.fields if field.required]

    for field in required_fields:
        missing = 0
        for row in rows:
            value = row.get(field.name)
            if value in (None, ""):
                missing += 1
        violations[field.name] = missing

    return ValidationResult(
        dataset_id=dataset_spec.meta.dataset_id,
        total_rows=len(rows),
        required_violations=violations,
    )

