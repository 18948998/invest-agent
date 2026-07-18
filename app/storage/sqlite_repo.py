"""SQLite persistence for standardized dataset records."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from app.config.schema import DatasetSpec

logger = logging.getLogger(__name__)


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

    # 判断是否为财报表（带 report_date 字段），决定去重键
    has_report_date = any(f.name == "report_date" for f in dataset_spec.fields)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()
        if replace:
            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            cursor.execute(f'CREATE TABLE "{table_name}" ({columns_sql})')
        else:
            cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql})')
            # --- schema 自动补齐：如果表已存在但缺某些列，动态 ALTER TABLE ADD COLUMN ---
            existing_cols = {
                row[1]
                for row in cursor.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            }
            for field in dataset_spec.fields:
                if field.name not in existing_cols:
                    col_def = f'"{field.name}" {_sqlite_type(field.dtype)}'
                    try:
                        cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN {col_def}')
                        logger.info("自动补齐列: %s.%s (%s)", table_name, field.name, _sqlite_type(field.dtype))
                    except sqlite3.OperationalError as exc:
                        logger.warning("无法添加列 %s.%s: %s", table_name, field.name, exc)

        if rows:
            if table_name == "dividend_history" and not replace:
                # 分红历史：用 (symbol, plan_notice_date) 去重
                # 同一 report_date 下可能有多次分红事件（中期+年末），预案公告日唯一
                pairs = sorted({
                    (row.get("symbol"), row.get("plan_notice_date") or row.get("report_date"))
                    for row in rows
                    if row.get("symbol") is not None
                })
                if pairs:
                    ph = ", ".join(["(?, ?)"] * len(pairs))
                    flat = [v for p in pairs for v in p]
                    cursor.execute(
                        f'DELETE FROM "{table_name}" WHERE ("symbol","plan_notice_date") IN ({ph})',
                        flat,
                    )
            elif has_report_date and not replace:
                # 财报表：用 (symbol, report_date) 复合主键去重，支持多年并存
                pairs = sorted({
                    (row.get("symbol"), row.get("report_date"))
                    for row in rows
                    if row.get("symbol") is not None and row.get("report_date") is not None
                })
                if pairs:
                    ph = ", ".join(["(?, ?)"] * len(pairs))
                    flat = [v for p in pairs for v in p]
                    cursor.execute(
                        f'DELETE FROM "{table_name}" WHERE ("symbol","report_date") IN ({ph})',
                        flat,
                    )
            else:
                # basic_info：symbol 单键去重（保持原有行为）
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


# ---------------------------------------------------------------------------
# data freshness – check if cached data is too old
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DataFreshness:
    """一次数据新鲜度检查的结果。"""
    is_price_stale: bool      # basic_info 是否过期（> 1 天）
    is_financial_stale: bool  # 三张财报是否过期（> 1 个月）
    is_dividend_stale: bool   # 分红数据是否过期（> 1 个月）
    price_last_update: str | None
    financial_last_update: str | None
    dividend_last_update: str | None
    db_path: Path

    @property
    def any_stale(self) -> bool:
        return self.is_price_stale or self.is_financial_stale or self.is_dividend_stale

    @property
    def stale_tracks(self) -> list[str]:
        tracks: list[str] = []
        if self.is_price_stale:
            tracks.append("price")
        if self.is_financial_stale:
            tracks.append("financial")
        if self.is_dividend_stale:
            tracks.append("dividend")
        return tracks


def check_data_freshness(db_path: Path) -> DataFreshness:
    """检查数据库中各类数据的最后更新时间。

    basic_info（股价/估值）超过 1 天 → price_stale
    财报三表超过 1 个月 → financial_stale
    分红数据超过 1 个月 → dividend_stale
    """
    now = datetime.now(timezone.utc)

    price_record = get_last_update(db_path, "price")
    financial_record = get_last_update(db_path, "financial")
    dividend_record = get_last_update(db_path, "dividend")

    _price_last = price_record["updated_at"] if price_record else None
    _fin_last = financial_record["updated_at"] if financial_record else None
    _div_last = dividend_record["updated_at"] if dividend_record else None

    price_stale = True
    financial_stale = True
    dividend_stale = True

    if _price_last:
        try:
            last = datetime.strptime(_price_last, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            price_stale = (now - last) > timedelta(days=1)
        except ValueError:
            pass

    if _fin_last:
        try:
            last = datetime.strptime(_fin_last, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            financial_stale = (now - last) > timedelta(days=30)
        except ValueError:
            pass

    if _div_last:
        try:
            last = datetime.strptime(_div_last, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            dividend_stale = (now - last) > timedelta(days=30)
        except ValueError:
            pass

    return DataFreshness(
        is_price_stale=price_stale,
        is_financial_stale=financial_stale,
        is_dividend_stale=dividend_stale,
        price_last_update=_price_last,
        financial_last_update=_fin_last,
        dividend_last_update=_div_last,
        db_path=db_path,
    )


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



