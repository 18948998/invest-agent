"""LangChain tools for agent-driven database queries.

提供两个核心工具让 agent 自由探索投资数据库：
- query_database: 执行任意 SELECT（支持 JOIN、聚合、分组、子查询等）
- list_tables: 查看可用表和字段结构

安全机制：
- 仅允许 SELECT 语句
- 字段名白名单校验（阻止 SQL 注入）
- 结果行数上限防止上下文溢出
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ==============================================================================
#  Schema snapshot —— 与 configs/fundamental_fields/*.yaml 保持一致
# ==============================================================================

_TABLE_SCHEMAS: dict[str, dict[str, str]] = {
    "basic_info": {
        "symbol": "TEXT", "name": "TEXT", "market": "TEXT",
        "industry": "TEXT", "listing_date": "TEXT",
        "current_price": "REAL", "pe_ttm": "REAL", "pb": "REAL",
        "ps_ttm": "REAL", "market_cap": "REAL",
        "eps_basic": "REAL", "eps_ttm": "REAL", "bvps": "REAL",
        "dividend_per_share": "REAL", "dividend_payout_ratio": "REAL",
        "dividend_yield": "REAL", "roe": "REAL",
    },
    "income_statement": {
        "symbol": "TEXT", "name": "TEXT",
        "report_date": "TEXT", "announce_date": "TEXT",
        "total_revenue": "REAL", "revenue": "REAL",
        "operating_cost": "REAL", "selling_expense": "REAL",
        "administrative_expense": "REAL",
        "research_and_development_expense": "REAL",
        "financial_expense": "REAL", "operating_profit": "REAL",
        "total_profit": "REAL", "income_tax_expense": "REAL",
        "net_profit": "REAL",
        "net_profit_attributable_to_parent": "REAL",
        "net_profit_excluding_non_recurring": "REAL",
        "eps_basic": "REAL", "eps_diluted": "REAL",
    },
    "balance_sheet": {
        "symbol": "TEXT", "name": "TEXT",
        "report_date": "TEXT", "announce_date": "TEXT",
        "cash_and_equivalents": "REAL", "accounts_receivable": "REAL",
        "inventory": "REAL", "total_current_assets": "REAL",
        "fixed_assets": "REAL", "intangible_assets": "REAL",
        "goodwill": "REAL", "total_assets": "REAL",
        "short_term_borrowings": "REAL",
        "total_current_liabilities": "REAL",
        "long_term_borrowings": "REAL", "total_liabilities": "REAL",
        "share_capital": "REAL", "retained_earnings": "REAL",
        "equity_attributable_to_parent": "REAL",
    },
    "cash_flow_statement": {
        "symbol": "TEXT", "name": "TEXT",
        "report_date": "TEXT", "announce_date": "TEXT",
        "cash_inflow_from_operating_activities": "REAL",
        "cash_outflow_from_operating_activities": "REAL",
        "net_cash_flow_from_operating_activities": "REAL",
        "cash_paid_for_long_term_assets": "REAL",
        "net_cash_flow_from_investing_activities": "REAL",
        "net_cash_flow_from_financing_activities": "REAL",
        "net_increase_in_cash_and_equivalents": "REAL",
        "ending_cash_and_equivalents_balance": "REAL",
    },
}

_ALLOWED_TABLES = frozenset(_TABLE_SCHEMAS.keys())
_ALLOWED_COLUMNS: dict[str, frozenset[str]] = {
    t: frozenset(cols.keys()) for t, cols in _TABLE_SCHEMAS.items()
}
_ALL_KNOWN_COLUMNS: frozenset[str] = frozenset(
    col for cols in _ALLOWED_COLUMNS.values() for col in cols
)

# ==============================================================================
#  SQL 安全校验
# ==============================================================================

_FORBIDDEN_TOKENS: tuple[str, ...] = (
    ";", "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE",
    "UNION", "--", "/*", "*/", "EXEC", "EXECUTE", "ATTACH", "DETACH",
    "PRAGMA", "VACUUM", "REINDEX", "SAVEPOINT", "RELEASE",
)

_SQL_KEYWORDS: frozenset[str] = frozenset({
    "AND", "OR", "NOT", "IN", "LIKE", "BETWEEN", "IS", "NULL",
    "TRUE", "FALSE", "SELECT", "FROM", "WHERE", "ORDER", "BY",
    "GROUP", "HAVING", "LIMIT", "OFFSET", "AS", "ON", "JOIN",
    "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "FULL",
    "CASE", "WHEN", "THEN", "ELSE", "END", "ASC", "DESC",
    "DISTINCT", "ALL", "COUNT", "SUM", "AVG", "MAX", "MIN",
    "EXISTS", "CAST", "COALESCE", "NULLIF", "IFNULL",
    "ROUND", "ABS", "UPPER", "LOWER", "LENGTH", "SUBSTR",
    "REPLACE", "TRIM", "TYPEOF", "TOTAL", "GROUP_CONCAT",
    "INTEGER", "REAL", "TEXT", "BLOB", "NUMERIC",
    "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "INDEX",
    "UNIQUE", "CHECK", "DEFAULT", "CONSTRAINT", "TABLE",
    "USING", "NATURAL", "UNION", "INTERSECT", "EXCEPT",
    "OVER", "PARTITION", "ROWS", "RANGE", "PRECEDING",
    "FOLLOWING", "UNBOUNDED", "CURRENT", "ROW", "FIRST",
    "LAST", "VALUES", "FILTER", "RECURSIVE", "MATERIALIZED",
    "IIF", "LAG", "LEAD", "ROW_NUMBER", "RANK", "DENSE_RANK",
    "NTILE", "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE",
    "STRFTIME", "DATE", "TIME", "DATETIME", "JULIANDAY",
    "hex", "like", "glob",
})


def _extract_identifiers(sql: str) -> set[str]:
    """从 SQL 中提取所有可能是列名/表名的标识符。"""
    # 先去掉字符串字面量，避免匹配到字符串内容
    cleaned = re.sub(r"'[^']*'", "", sql)
    cleaned = re.sub(r'"[^"]*"', "", cleaned)
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", cleaned)
    return {t for t in tokens if t.upper() not in _SQL_KEYWORDS}


def _validate_sql(sql: str) -> tuple[bool, str]:
    """校验 SQL 安全性。

    Returns:
        (ok, error_msg) —— ok=True 表示通过校验。
    """
    upper = sql.upper().strip()

    # 1. 必须是 SELECT
    if not upper.startswith("SELECT"):
        return False, "仅允许 SELECT 语句"

    # 2. 检查危险 token
    for tok in _FORBIDDEN_TOKENS:
        if tok in upper:
            return False, f"SQL 含危险 token: '{tok}'"

    # 3. 标识符白名单校验
    identifiers = _extract_identifiers(sql)
    for ident in identifiers:
        if ident not in _ALL_KNOWN_COLUMNS and ident not in _ALLOWED_TABLES:
            return False, (
                f"SQL 含未知标识符: '{ident}'（不在字段白名单内）。"
                f"请用 list_tables 工具查看可用表和字段。"
            )

    return True, ""


# ==============================================================================
#  结果格式化
# ==============================================================================

MAX_RESULT_ROWS: int = 200


def _format_rows(rows: list[dict[str, Any]], max_rows: int = MAX_RESULT_ROWS) -> str:
    """将查询结果格式化为可读文本表格。"""
    if not rows:
        return "(查询结果为空)"

    truncated = rows[:max_rows]
    headers = list(truncated[0].keys())

    # 计算每列宽度
    col_widths: dict[str, int] = {}
    for h in headers:
        col_widths[h] = len(h)
        for row in truncated:
            v = row.get(h)
            s = _format_value(v)
            col_widths[h] = max(col_widths[h], len(s))

    lines: list[str] = []

    # 表头
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    lines.append(header_line)
    lines.append("-+-".join("-" * col_widths[h] for h in headers))

    # 数据行
    for row in truncated:
        values = [_format_value(row.get(h)).ljust(col_widths[h]) for h in headers]
        lines.append(" | ".join(values))

    result = "\n".join(lines)

    if len(rows) > max_rows:
        result += f"\n\n... (共 {len(rows)} 行，仅显示前 {max_rows} 行)"
    else:
        result += f"\n\n(共 {len(rows)} 行)"

    return result


def _format_value(value: Any) -> str:
    """智能格式化单个值。"""
    if value is None:
        return "NULL"
    if isinstance(value, float):
        if abs(value) >= 1e8:
            return f"{value:.4e}"
        elif abs(value) < 0.01 and value != 0:
            return f"{value:.6f}"
        else:
            return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


# ==============================================================================
#  LangChain Tools
# ==============================================================================

@tool
def query_database(sql: str, db_path: str) -> str:
    """在投资数据库中执行 SELECT 查询。支持多表 JOIN、聚合函数、GROUP BY、ORDER BY、子查询等。

可用表及字段：

basic_info（基础信息 + 估值）—— symbol, name, market, industry, listing_date,
  current_price, pe_ttm, pb, ps_ttm, market_cap, eps_basic, eps_ttm,
  bvps, dividend_per_share, dividend_payout_ratio, dividend_yield, roe

income_statement（利润表）—— symbol, name, report_date, announce_date, total_revenue,
  revenue, operating_cost, selling_expense, administrative_expense,
  research_and_development_expense, financial_expense, operating_profit,
  total_profit, income_tax_expense, net_profit,
  net_profit_attributable_to_parent, net_profit_excluding_non_recurring,
  eps_basic, eps_diluted

balance_sheet（资产负债表）—— symbol, name, report_date, announce_date,
  cash_and_equivalents, accounts_receivable, inventory, total_current_assets,
  fixed_assets, intangible_assets, goodwill, total_assets,
  short_term_borrowings, total_current_liabilities, long_term_borrowings,
  total_liabilities, share_capital, retained_earnings, equity_attributable_to_parent

cash_flow_statement（现金流量表）—— symbol, name, report_date, announce_date,
  cash_inflow_from_operating_activities, cash_outflow_from_operating_activities,
  net_cash_flow_from_operating_activities, cash_paid_for_long_term_assets,
  net_cash_flow_from_investing_activities, net_cash_flow_from_financing_activities,
  net_increase_in_cash_and_equivalents, ending_cash_and_equivalents_balance

多表 JOIN 示例：
  LEFT JOIN income_statement   ON basic_info.symbol = income_statement.symbol
  LEFT JOIN balance_sheet      ON basic_info.symbol = balance_sheet.symbol
  LEFT JOIN cash_flow_statement ON basic_info.symbol = cash_flow_statement.symbol

Tips：
  - 财报表（income_statement/balance_sheet/cash_flow_statement）有多期数据，
    查最新一期请用 ORDER BY report_date DESC LIMIT 1 或子查询。
  - 用 GROUP BY symbol 配合 MAX(report_date) 可获取每只股票的最新报告期。
  - basic_info 每只股票只有一行，可直接 JOIN。

Args:
  sql: 完整 SQL SELECT 语句。
  db_path: SQLite 数据库文件路径。
"""
    ok, err = _validate_sql(sql)
    if not ok:
        logger.warning("SQL 校验拒绝: %s | SQL: %.200s", err, sql)
        return f"查询被拒绝: {err}"

    db = Path(db_path)
    if not db.exists():
        return f"数据库文件不存在: {db_path}"

    try:
        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            rows = [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError as exc:
        logger.warning("SQL 执行失败: %s", exc)
        return f"查询执行失败: {exc}\n请检查 SQL 语法、表名和字段名。"
    except Exception as exc:
        logger.exception("SQL 执行异常")
        return f"查询异常: {exc}"

    return _format_rows(rows)


@tool
def list_tables(db_path: str) -> str:
    """列出投资数据库中所有可用表及其字段结构。

用于了解有哪些数据可以查询，以及每个字段的含义。

Args:
  db_path: SQLite 数据库文件路径。
"""
    table_labels = {
        "basic_info": "基础信息与估值指标（每只股票一行）",
        "income_statement": "利润表（多期，按 report_date 区分）",
        "balance_sheet": "资产负债表（多期，按 report_date 区分）",
        "cash_flow_statement": "现金流量表（多期，按 report_date 区分）",
    }

    join_guide = """
---

## 表关联关系
所有表通过 `symbol` 字段关联。财报三表额外需要用 `report_date` 对齐报告期。

### 获取每只股票最新财报的典型写法：
```sql
SELECT b.symbol, b.name, b.pe_ttm, i.net_profit, bs.total_assets
FROM basic_info b
LEFT JOIN (
    SELECT symbol, MAX(report_date) AS latest_date
    FROM income_statement GROUP BY symbol
) latest ON b.symbol = latest.symbol
LEFT JOIN income_statement i
    ON b.symbol = i.symbol AND i.report_date = latest.latest_date
LEFT JOIN balance_sheet bs
    ON b.symbol = bs.symbol AND bs.report_date = latest.latest_date
WHERE b.pe_ttm > 0 AND b.pe_ttm < 30
ORDER BY b.pe_ttm ASC
LIMIT 20
```
"""

    lines = [""]
    for table_name, columns in _TABLE_SCHEMAS.items():
        label = table_labels.get(table_name, table_name)
        lines.append(f"### {table_name}  ({label})")
        for col, dtype in columns.items():
            lines.append(f"  {col}  ({dtype})")

    lines.append(join_guide)

    # 检查数据库中实际存在的表
    db = Path(db_path)
    if db.exists():
        try:
            with sqlite3.connect(str(db)) as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                actual = [row[0] for row in cursor.fetchall()]
            lines.append(f"\n> 数据库中实际存在的表: {', '.join(actual)}")
        except Exception:
            pass

    return "\n".join(lines)
