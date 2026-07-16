"""SQLite persistence for standardized dataset records."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.config.schema import DatasetSpec


def _sqlite_type(dtype: str) -> str:
    if dtype == "float":
        return "REAL"
    return "TEXT"


def save_dataset(
    db_path: Path,
    dataset_spec: DatasetSpec,
    rows: list[dict[str, Any]],
) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    table_name = dataset_spec.meta.dataset_id

    columns_sql = ", ".join(
        f'"{field.name}" {_sqlite_type(field.dtype)}' for field in dataset_spec.fields
    )
    column_names = [field.name for field in dataset_spec.fields]
    placeholders = ", ".join(["?"] * len(column_names))
    insert_sql = (
        f'INSERT INTO "{table_name}" ({", ".join(f"\"{name}\"" for name in column_names)}) '
        f"VALUES ({placeholders})"
    )

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        cursor.execute(f'CREATE TABLE "{table_name}" ({columns_sql})')

        if rows:
            values = [tuple(row.get(name) for name in column_names) for row in rows]
            cursor.executemany(insert_sql, values)
        conn.commit()

    return len(rows)

