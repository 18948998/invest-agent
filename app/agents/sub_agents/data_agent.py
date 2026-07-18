"""Data agent —— 从 SQLite 数据库加载标准化行情与基本面数据。

职责：
  - 连接 SQLite 数据库，按表/字段/条件查询数据
  - 检查数据新鲜度（不自动刷新，将结果上报给调用方决定）
  - 返回标准化的 list[dict] 供下游 factor_agent / rule_agent 消费

契约：
  输入:  (db_path, table, fields, where_clause) 或 (db_path, symbols, fields)
  输出:  list[dict[str, Any]]  —— 每行一个 dict, key 为字段名

数据流位置：
  MainAgent → workflow.load_config → workflow.plan_filter
  → data_agent.load_*() → factor_agent → rule_agent
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.storage.tools import _TABLE_SCHEMAS, _validate_sql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  新鲜度状态
# ---------------------------------------------------------------------------

@dataclass
class FreshnessStatus:
    """数据新鲜度检查结果，不含副作用（不自动刷新）。"""
    needs_refresh: bool = False
    price_need: bool = False            # basic_info 需要刷新
    financial_need: bool = False        # 三张财报需要刷新
    price_reason: str = ""              # 空表 / 数据过期 / 数据不全（仅 N 行）
    financial_reason: str = ""
    price_count: int = 0
    financial_count: int = 0

    @property
    def tracks(self) -> list[str]:
        """哪些数据轨道需要刷新。"""
        result: list[str] = []
        if self.price_need:
            result.append("price")
        if self.financial_need:
            result.append("financial")
        return result

    @property
    def summary(self) -> str:
        """人类可读的汇总描述。"""
        parts: list[str] = []
        if self.price_need:
            parts.append(f"行情数据（{self.price_reason}）")
        if self.financial_need:
            parts.append(f"财报数据（{self.financial_reason}）")
        return "、".join(parts) if parts else "数据正常"


def check_freshness(db_path_str: str) -> FreshnessStatus:
    """检查数据新鲜度，返回状态供调用方决定是否刷新。

    检查维度（结果均基于数据库实际内容判断，不依赖内存缓存）：
      - 空表     → 表不存在或无数据
      - 过期     → basic_info > 1 天（来自 update_history 表），财报 > 30 天
      - 不全     → basic_info 表总行数 < 5000

    注意：如果 update_history 表为空但数据已存在且行数足够，会自动补写
    一条记录（当前时间），避免因缺少刷新记录而误判"过期"。

    此函数不触发任何数据刷新。调用方根据返回的 FreshnessStatus
    自行决定是否调用 refresh_basic_info / refresh_financials。
    """
    from app.storage.sqlite_repo import check_data_freshness

    MIN_ROWS = 5000
    db = Path(db_path_str)
    freshness = check_data_freshness(db)

    # --- basic_info ---
    basic_empty = not _ensure_table(db, "basic_info")
    basic_count = 0 if basic_empty else _count_table(db, "basic_info")

    # --- 财报 ---
    bs_count = 0 if not _ensure_table(db, "balance_sheet") else _count_table(db, "balance_sheet")
    is_count = 0 if not _ensure_table(db, "income_statement") else _count_table(db, "income_statement")
    fin_empty = bs_count == 0 or is_count == 0
    fin_count = min(bs_count, is_count)

    # --- 分红 ---
    div_empty = not _ensure_table(db, "dividend_history")

    # --- 汇总判断 ---
    price_need = basic_empty or freshness.is_price_stale or basic_count < MIN_ROWS
    price_reason = (
        "空表" if basic_empty
        else "数据过期" if freshness.is_price_stale
        else f"数据不全（仅 {basic_count} 行）" if basic_count < MIN_ROWS
        else ""
    )

    financial_need = fin_empty or freshness.is_financial_stale or fin_count < MIN_ROWS
    financial_reason = (
        "空表" if fin_empty
        else "数据过期" if freshness.is_financial_stale
        else f"数据不全（仅 {fin_count} 行）" if fin_count < MIN_ROWS
        else ""
    )

    # 分红表为空时也标记需要刷新（财报刷新会同时拉分红）
    if div_empty and not financial_need:
        financial_need = True
        financial_reason = "分红表为空"

    status = FreshnessStatus(
        needs_refresh=price_need or financial_need,
        price_need=price_need,
        financial_need=financial_need,
        price_reason=price_reason,
        financial_reason=financial_reason,
        price_count=basic_count,
        financial_count=fin_count,
    )

    if status.needs_refresh:
        logger.info("数据新鲜度检查: %s", status.summary)

    return status


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_BASIC_FIELDS: tuple[str, ...] = (
    "symbol", "name", "market", "industry", "listing_date",
    "current_price", "pe_ttm", "pb", "ps_ttm", "market_cap",
    "eps_basic", "eps_ttm", "bvps",
    "dividend_per_share", "dividend_payout_ratio", "dividend_yield", "roe",
)

# 三张财报表共有的标识字段
_FINANCIAL_KEY_FIELDS: tuple[str, ...] = ("symbol", "name", "report_date", "announce_date")


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _ensure_table(db_path: Path, table: str) -> bool:
    """确认表在数据库中存在。"""
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            return cursor.fetchone() is not None
    except Exception:
        return False


def _count_table(db_path: Path, table: str) -> int:
    """返回表的行数，表不存在返回 0。"""
    if not _ensure_table(db_path, table):
        return 0
    try:
        with sqlite3.connect(str(db_path)) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
    except Exception:
        return 0


def _quote(identifier: str) -> str:
    """双引号安全包裹标识符。"""
    if not identifier.replace("_", "").replace(".", "").isalnum():
        raise ValueError(f"Illegal identifier: {identifier!r}")
    return f'"{identifier}"'


def _table_columns(db_path: Path, table: str) -> list[str]:
    """返回表中实际存在的列名列表（防御旧 schema 缺列问题）。"""
    try:
        with sqlite3.connect(str(db_path)) as conn:
            return [row[1] for row in conn.execute(f"PRAGMA table_info({_quote(table)})").fetchall()]
    except Exception:
        return []


def _rows_to_dicts(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    """把 Row 游标的结果转成 list[dict]。"""
    return [dict(row) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def load_basic_info(
    db_path: str | Path,
    symbols: list[str] | None = None,
    fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """从 basic_info 表加载行情与估值数据。

    Args:
        db_path: SQLite 数据库文件路径。
        symbols: 要查询的股票代码列表，为 None 则查全部。
        fields:  要返回的字段列表，为 None 则返回所有默认字段。

    Returns:
        list[dict] —— 每行一个 dict。表不存在或没数据时返回空列表。

    Example:
        >>> rows = load_basic_info("data/invest.db", symbols=["000001", "000002"])
        >>> rows[0]["symbol"], rows[0]["pe_ttm"]
        ('000001', 4.5)
    """
    db = Path(db_path)
    table = "basic_info"

    # 字段白名单校验 & 默认值填充
    allowed = frozenset(_TABLE_SCHEMAS.get(table, {}))
    if fields is None:
        columns = [f for f in DEFAULT_BASIC_FIELDS if f in allowed]
    else:
        columns = [f for f in fields if f in allowed]
        unknown = set(fields) - allowed
        if unknown:
            logger.warning("basic_info 查询含未知字段，已忽略: %s", unknown)

    if not columns:
        logger.warning("basic_info 无有效字段可查询")
        return []

    if not _ensure_table(db, table):
        logger.warning("basic_info 表不存在: %s", db)
        return []

    col_str = ", ".join(_quote(c) for c in columns)
    query = f"SELECT {col_str} FROM {_quote(table)}"

    params: list[str] = []
    if symbols:
        placeholders = ", ".join(["?"] * len(symbols))
        query += f" WHERE symbol IN ({placeholders})"
        params.extend(symbols)

    try:
        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return _rows_to_dicts(cursor)
    except sqlite3.OperationalError as exc:
        logger.warning("basic_info 查询失败: %s", exc)
        return []


def load_latest_financial(
    db_path: str | Path,
    table: str,
    symbols: list[str] | None = None,
    fields: list[str] | None = None,
    report_type: str = "年报",
) -> list[dict[str, Any]]:
    """加载财报表（income_statement / balance_sheet / cash_flow_statement）最新一期数据。

    每只股票取 report_date 最大的记录，默认只取年报（report_type='年报'），
    与 basic_info 通过 symbol 对齐。

    Args:
        db_path:  SQLite 数据库文件路径。
        table:   表名（income_statement / balance_sheet / cash_flow_statement）。
        symbols: 要查询的股票代码列表，为 None 则查全部。
        fields:  要返回的字段列表，为 None 则返回所有字段（排除 key 字段重复）。
        report_type: 报告类型过滤，"年度报告"（默认）或 "半年度报告"。

    Returns:
        list[dict] —— 每只股票最新的财报记录。

    Example:
        >>> rows = load_latest_financial("data/invest.db", "balance_sheet",
        ...                               symbols=["000001"])
        >>> rows[0]["total_assets"]
        5134567891234.0
    """
    db = Path(db_path)
    allowed = frozenset(_TABLE_SCHEMAS.get(table, {}))
    if not allowed:
        logger.warning("未知表: %s", table)
        return []

    if fields is None:
        columns = [c for c in allowed if c not in _FINANCIAL_KEY_FIELDS]
    else:
        columns = [f for f in fields if f in allowed]
        unknown = set(fields) - allowed
        if unknown:
            logger.warning("%s 查询含未知字段，已忽略: %s", table, unknown)

    # 始终加上 symbol 用于对齐
    if "symbol" not in columns:
        columns.insert(0, "symbol")

    if not _ensure_table(db, table):
        logger.warning("表 %s 不存在: %s", table, db)
        return []

    col_str = ", ".join(_quote(c) for c in columns)

    # 防御性检查：report_type 列是否存在且有实际数据
    # 旧 schema 无此列，或列存在但全是 NULL（数据源未填充）→ 跳过过滤
    actual_cols = _table_columns(db, table)
    has_report_type = "report_type" in actual_cols
    if has_report_type:
        try:
            with sqlite3.connect(str(db)) as conn:
                cnt = conn.execute(
                    f"SELECT COUNT(*) FROM {_quote(table)} WHERE report_type IS NOT NULL"
                ).fetchone()[0]
            if cnt == 0:
                has_report_type = False
                logger.info(
                    "%s 表 report_type 列全为 NULL（数据源未填充），跳过报告类型过滤", table
                )
        except sqlite3.OperationalError:
            has_report_type = False

    if has_report_type:
        # 子查询：每只股票最新的 report_date（按 report_type 过滤，默认年报）
        inner_where = "WHERE report_type = ?"
        inner_params: list[str] = [report_type]
        outer_params: list[str] = []
        if symbols:
            placeholders = ", ".join(["?"] * len(symbols))
            inner_where += f" AND symbol IN ({placeholders})"
            inner_params.extend(symbols)
            outer_params = list(symbols)

        query = f"""
            SELECT {col_str}
            FROM {_quote(table)}
            WHERE (symbol, report_date) IN (
                SELECT symbol, MAX(report_date)
                FROM {_quote(table)}
                {inner_where}
                GROUP BY symbol
            )
            AND report_type = ?
        """
        if outer_params:
            query += f" AND symbol IN ({', '.join(['?'] * len(outer_params))})"

        params = inner_params + outer_params + [report_type]
    else:
        # 旧 schema 没有 report_type 列，跳过过滤直接取最新 report_date
        logger.info("%s 表缺少 report_type 列（旧 schema），跳过报告类型过滤", table)
        inner_params: list[str] = []
        symbol_filter = ""
        if symbols:
            placeholders = ", ".join(["?"] * len(symbols))
            symbol_filter = f" WHERE symbol IN ({placeholders})"
            inner_params.extend(symbols)

        query = f"""
            SELECT {col_str}
            FROM {_quote(table)}
            WHERE (symbol, report_date) IN (
                SELECT symbol, MAX(report_date)
                FROM {_quote(table)}
                {symbol_filter}
                GROUP BY symbol
            )
        """
        if symbols:
            query += f" AND symbol IN ({', '.join(['?'] * len(symbols))})"
        params = inner_params + (list(symbols) if symbols else [])

    try:
        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return _rows_to_dicts(cursor)
    except sqlite3.OperationalError as exc:
        logger.warning("%s 查询失败: %s", table, exc)
        return []


DEFAULT_DIVIDEND_FIELDS: tuple[str, ...] = (
    "symbol", "name",
    "plan_profile", "pretax_bonus_per_share", "bonus_share_ratio",
    "assign_progress", "plan_notice_date", "equity_record_date",
    "ex_dividend_date", "implement_notice_date", "report_date",
)


def load_income_statement_history(
    db_path: str | Path,
    symbols: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """加载利润表的全部历史记录，按 symbol 分组供历史趋势因子计算。

    用于计算 defance 策略中的"过去10年 EPS 增长"和"连续盈利年数"等条件。

    Args:
        db_path: SQLite 数据库文件路径。
        symbols: 要查询的股票代码列表，为 None 则查全部。

    Returns:
        dict[str, list[dict]]: {symbol: [{report_date, eps_basic, net_profit, ...}, ...]}
        每个 symbol 的记录按 report_date 降序排列（最新在前）。
    """
    db = Path(db_path)
    table = "income_statement"

    if not _ensure_table(db, table):
        logger.warning("income_statement 表不存在，无法加载历史数据")
        return {}

    actual_cols = _table_columns(db, table)
    # 至少需要 symbol 和 eps_basic
    needed = {"symbol", "report_date", "eps_basic", "net_profit", "net_profit_attributable_to_parent"}
    available = [c for c in needed if c in actual_cols]
    if "symbol" not in available:
        return {}

    col_str = ", ".join(_quote(c) for c in available)

    query = f"SELECT {col_str} FROM {_quote(table)}"
    params: list[str] = []
    conditions: list[str] = [
        "report_date LIKE '%-12-31'"   # 只取年报（report_date = YYYY-12-31）
    ]

    if symbols:
        placeholders = ", ".join(["?"] * len(symbols))
        conditions.append(f"symbol IN ({placeholders})")
        params.extend(symbols)

    query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY report_date DESC"

    try:
        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = _rows_to_dicts(cursor)
    except sqlite3.OperationalError as exc:
        logger.warning("income_statement 历史查询失败: %s", exc)
        return {}

    # 按 symbol 分组
    from collections import defaultdict
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sym = str(row.get("symbol", ""))
        if sym:
            grouped[sym].append(row)

    return dict(grouped)


def load_dividend_history(
    db_path: str | Path,
    symbols: list[str] | None = None,
    fields: list[str] | None = None,
    order_by: str = "plan_notice_date DESC",
) -> list[dict[str, Any]]:
    """从 dividend_history 表加载分红送转历史数据。

    Args:
        db_path: SQLite 数据库文件路径。
        symbols: 要查询的股票代码列表，为 None 则查全部。
        fields:  要返回的字段列表，为 None 则返回所有默认字段。
        order_by: 排序方式，默认按预案公告日降序（最新的在前）。

    Returns:
        list[dict] —— 每次分红事件一行。表不存在或没数据时返回空列表。

    Example:
        >>> rows = load_dividend_history("data/invest.db", symbols=["600519"])
        >>> rows[0]["plan_profile"], rows[0]["pretax_bonus_per_share"]
        ('10派280.2423元(含税)', 28.02423)
    """
    db = Path(db_path)
    table = "dividend_history"

    allowed = frozenset(_TABLE_SCHEMAS.get(table, {}))
    if fields is None:
        columns = [f for f in DEFAULT_DIVIDEND_FIELDS if f in allowed]
    else:
        columns = [f for f in fields if f in allowed]
        unknown = set(fields) - allowed
        if unknown:
            logger.warning("dividend_history 查询含未知字段，已忽略: %s", unknown)

    if not columns:
        logger.warning("dividend_history 无有效字段可查询")
        return []

    if not _ensure_table(db, table):
        logger.warning("dividend_history 表不存在: %s", db)
        return []

    col_str = ", ".join(_quote(c) for c in columns)
    query = f"SELECT {col_str} FROM {_quote(table)}"

    params: list[str] = []
    if symbols:
        placeholders = ", ".join(["?"] * len(symbols))
        query += f" WHERE symbol IN ({placeholders})"
        params.extend(symbols)

    if order_by:
        query += f" ORDER BY {order_by}"

    try:
        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return _rows_to_dicts(cursor)
    except sqlite3.OperationalError as exc:
        logger.warning("dividend_history 查询失败: %s", exc)
        return []


def load_by_query(
    db_path: str | Path,
    sql: str,
) -> tuple[list[dict[str, Any]], str]:
    """执行经过安全校验的通用查询。

    Args:
        db_path: SQLite 数据库文件路径。
        sql:     完整的 SELECT 语句（会经过安全校验）。

    Returns:
        (rows, error) —— rows 为 list[dict]，error 为空字符串表示成功。

    Example:
        >>> rows, err = load_by_query(
        ...     "data/invest.db",
        ...     "SELECT symbol, name, pe_ttm FROM basic_info WHERE pe_ttm < 15"
        ... )
    """
    db = Path(db_path)

    ok, err = _validate_sql(sql)
    if not ok:
        logger.warning("SQL 校验拒绝: %s", err)
        return [], f"SQL 校验拒绝: {err}"

    if not db.exists():
        return [], f"数据库文件不存在: {db_path}"

    try:
        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            return _rows_to_dicts(cursor), ""
    except sqlite3.OperationalError as exc:
        logger.warning("查询执行失败: %s", exc)
        return [], f"查询执行失败: {exc}"
    except Exception as exc:
        logger.exception("查询异常")
        return [], f"查询异常: {exc}"


def get_all_table_names(db_path: str | Path) -> list[str]:
    """返回数据库中所有实际存在的表名。"""
    db = Path(db_path)
    if not db.exists():
        return []
    try:
        with sqlite3.connect(str(db)) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            return [row[0] for row in cursor.fetchall()]
    except Exception:
        return []
