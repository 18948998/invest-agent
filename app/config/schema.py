"""Typed schema models for configurable data-source field definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FieldSpec:
	name: str
	label_zh: str
	source_keys: dict[str, str]
	dtype: str
	unit: str | None
	required: bool
	description: str


@dataclass(frozen=True, slots=True)
class DatasetMeta:
	dataset_id: str
	dataset_name_zh: str
	statement_type: str
	version: int
	description: str


@dataclass(frozen=True, slots=True)
class DatasetSpec:
	meta: DatasetMeta
	fields: list[FieldSpec]

