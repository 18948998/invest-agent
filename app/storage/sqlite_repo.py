"""SQLite persistence for standardized dataset records."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
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
    replace: bool = True,
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
    key_field = "symbol"

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        if replace:
            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            cursor.execute(f'CREATE TABLE "{table_name}" ({columns_sql})')
        else:
            cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql})')

        if rows:
            key_values = sorted({row.get(key_field) for row in rows if row.get(key_field) is not None})
            if key_values:
                delete_placeholders = ", ".join(["?"] * len(key_values))
                cursor.execute(
                    f'DELETE FROM "{table_name}" WHERE "{key_field}" IN ({delete_placeholders})',
                    key_values,
                )
            values = [tuple(row.get(name) for name in column_names) for row in rows]
            cursor.executemany(insert_sql, values)
        conn.commit()

    return len(rows)


# ---------------------------------------------------------------------------
# update_history – track last-update timestamps per data track
# ---------------------------------------------------------------------------

UPDATE_HISTORY_DDL = """\
CREATE TABLE IF NOT EXISTS update_history (
    track       TEXT PRIMARY KEY,
    updated_at  TEXT NOT NULL,
    symbols_count INTEGER NOT NULL,
    status      TEXT NOT NULL
)
"""


def init_update_history(db_path: Path) -> None:
    """Ensure the update_history table exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(UPDATE_HISTORY_DDL)
        conn.commit()


def record_update(db_path: Path, track: str, symbols_count: int, status: str = "OK") -> None:
    """Record an update event for the given track (upsert by track)."""
    init_update_history(db_path)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """INSERT INTO update_history (track, updated_at, symbols_count, status)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(track) DO UPDATE SET
                   updated_at = excluded.updated_at,
                   symbols_count = excluded.symbols_count,
                   status = excluded.status""",
            (track, now_utc, symbols_count, status),
        )
        conn.commit()


def get_last_update(db_path: Path, track: str) -> dict[str, Any] | None:
    """Return the last update record for a track, or None if never updated."""
    init_update_history(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM update_history WHERE track = ?",
            (track,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)


def get_all_update_history(db_path: Path) -> list[dict[str, Any]]:
    """Return all update history records."""
    init_update_history(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM update_history ORDER BY track")
        return [dict(row) for row in cursor.fetchall()]


def get_latest_eps_batch(db_path: Path, symbols: list[str]) -> dict[str, dict[str, Any]]:
    """For each symbol, return {eps_basic, eps_ttm} from income_statement table, keyed by symbol.

    Note: income_statement only stores eps_basic / eps_diluted. eps_ttm falls back to eps_basic.
    """
    if not symbols:
        return {}
    placeholders = ", ".join(["?"] * len(symbols))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                f"""SELECT symbol, eps_basic, report_date
                    FROM income_statement
                    WHERE symbol IN ({placeholders})
                    ORDER BY report_date DESC""",
                symbols,
            )
        except sqlite3.OperationalError:
            return {}  # income_statement 表尚不存在，返回空
        result: dict[str, dict[str, Any]] = {}
        for row in cursor.fetchall():
            sym = row["symbol"]
            if sym not in result:
                eps = row["eps_basic"]
                result[sym] = {"eps_basic": eps, "eps_ttm": eps}
        return result



