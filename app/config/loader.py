"""Load configurable dataset/field definitions from YAML files."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from app.config.schema import DatasetMeta, DatasetSpec, FieldSpec


def _read_yaml(path: Path) -> dict[str, Any]:
	try:
		yaml = importlib.import_module("yaml")
	except ModuleNotFoundError as exc:
		raise RuntimeError("PyYAML is required to load YAML configs. Install dependencies first.") from exc
	with path.open("r", encoding="utf-8") as file:
		payload = yaml.safe_load(file) or {}
	if not isinstance(payload, dict):
		raise ValueError(f"Invalid YAML root object in {path}")
	return payload


def _to_field_spec(raw: dict[str, Any], file_path: Path) -> FieldSpec:
	source_keys = raw.get("source_keys", {})
	if not isinstance(source_keys, dict):
		raise ValueError(f"source_keys must be a map in {file_path}")
	return FieldSpec(
		name=str(raw["name"]),
		label_zh=str(raw.get("label_zh", raw["name"])),
		source_keys={str(k): str(v) for k, v in source_keys.items()},
		dtype=str(raw.get("dtype", "string")).lower(),
		unit=None if raw.get("unit") is None else str(raw.get("unit")),
		required=bool(raw.get("required", False)),
		description=str(raw.get("description", "")),
	)


def _to_dataset_spec(raw: dict[str, Any], file_path: Path) -> DatasetSpec:
	dataset_meta = raw.get("dataset", {})
	if not isinstance(dataset_meta, dict):
		raise ValueError(f"dataset block must be a map in {file_path}")

	fields_raw = raw.get("fields", [])
	if not isinstance(fields_raw, list) or not fields_raw:
		raise ValueError(f"fields must be a non-empty list in {file_path}")

	fields: list[FieldSpec] = []
	for item in fields_raw:
		if not isinstance(item, dict):
			raise ValueError(f"each field must be a map in {file_path}")
		fields.append(_to_field_spec(item, file_path))

	return DatasetSpec(
		meta=DatasetMeta(
			dataset_id=str(dataset_meta["dataset_id"]),
			dataset_name_zh=str(dataset_meta.get("dataset_name_zh", dataset_meta["dataset_id"])),
			statement_type=str(dataset_meta.get("statement_type", "unknown")),
			version=int(dataset_meta.get("version", 1)),
			description=str(dataset_meta.get("description", "")),
		),
		fields=fields,
	)


def load_fundamental_field_specs(config_dir: Path) -> list[DatasetSpec]:
	"""加载全部基本面字段定义（兼容旧接口）。"""
	return _load_specs(config_dir)


def load_specs_by_track(config_dir: Path, track: str) -> list[DatasetSpec]:
	"""按 track 筛选加载：'price' → basic_info, 'financial' → 三张财报。"""
	price_type = {"reference_and_valuation"}
	financial_type = {"financial_statement"}

	specs = _load_specs(config_dir)
	if track == "price":
		return [s for s in specs if s.meta.statement_type in price_type]
	if track == "financial":
		return [s for s in specs if s.meta.statement_type in financial_type]
	raise ValueError(f"Unsupported track: {track}. Use 'price' or 'financial'.")


def _load_specs(config_dir: Path) -> list[DatasetSpec]:
 yaml_files = sorted(config_dir.glob("*.yaml"))
 if not yaml_files:
  raise FileNotFoundError(f"No field definition YAML files under {config_dir}")

 specs: list[DatasetSpec] = []
 for yaml_file in yaml_files:
  payload = _read_yaml(yaml_file)
  specs.append(_to_dataset_spec(payload, yaml_file))
 return specs

