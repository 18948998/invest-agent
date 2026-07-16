"""Table column definitions for Chinese output display."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TableField:
    key: str
    label: str
    format_type: str = "text"


DEFAULT_TABLE_FIELDS: list[TableField] = [
    TableField(key="symbol", label="股票代码"),
    TableField(key="name", label="股票名称"),
    TableField(key="pe_ttm", label="市盈率(PE-TTM)", format_type="number"),
    TableField(key="current_price", label="当前价格", format_type="price"),
    TableField(key="safe_buy_price", label="安全买入价格", format_type="price"),
    TableField(
        key="intrinsic_value_price", label="价值价格", format_type="price"
    ),
]

