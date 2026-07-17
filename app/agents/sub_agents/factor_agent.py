"""Factor agent —— 从标准化数据中计算量化因子值。

职责：
  - 接收 data_agent 产出的标准化数据（list[dict]）
  - 计算基本面 / 估值 / 财务健康因子
  - 输出带因子值的标准化记录供 rule_agent 消费

设计原则：
  - 纯计算，无副作用 —— 不写文件、不调 API
  - 输出完全由输入决定 —— 幂等
  - 缺失值统一用 None 处理

契约：
  输入:  data: list[dict]         —— data_agent 产出的行数据
         strategy_config: dict     —— 策略 YAML 配置（可选，用于指定需要哪些因子）
  输出:  FactorResult
         - enriched_rows: list[dict]   —— 原始行 + 计算后的因子键值
         - factor_names: list[str]     —— 实际计算了哪些因子
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  产出模型
# ---------------------------------------------------------------------------

class FactorSet(BaseModel):
    """单只股票的一组因子值。"""

    symbol: str = ""
    name: str = ""

    # ---- 估值因子（直接从 basic_info 映射） ----
    pe_ttm: float | None = Field(default=None, description="滚动市盈率")
    pb: float | None = Field(default=None, description="市净率")
    ps_ttm: float | None = Field(default=None, description="滚动市销率")
    market_cap: float | None = Field(default=None, description="总市值（元）")

    # ---- 盈利因子 ----
    roe: float | None = Field(default=None, description="净资产收益率")
    earnings_yield: float | None = Field(default=None, description="盈利收益率 = 1/PE_TTM")
    eps_basic: float | None = Field(default=None, description="基本每股收益")
    bvps: float | None = Field(default=None, description="每股净资产")

    # ---- 分红因子 ----
    dividend_yield: float | None = Field(default=None, description="股息率")
    dividend_per_share: float | None = Field(default=None, description="每股股利")
    dividend_payout_ratio: float | None = Field(default=None, description="分红率")

    # ---- 财务健康因子（需 join 财报表） ----
    debt_to_equity: float | None = Field(default=None, description="负债权益比 = total_liabilities / equity")
    current_ratio: float | None = Field(default=None, description="流动比率 = current_assets / current_liabilities")
    total_assets: float | None = Field(default=None, description="总资产")
    net_profit: float | None = Field(default=None, description="净利润")


class FactorResult(BaseModel):
    """因子计算的完整产出。"""

    enriched_rows: list[dict[str, Any]] = Field(default_factory=list)
    factor_sets: list[FactorSet] = Field(default_factory=list)
    factor_names: list[str] = Field(default_factory=list)
    input_count: int = 0
    output_count: int = 0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
#  因子计算
# ---------------------------------------------------------------------------

# 直接映射：basic_info 字段 → 因子值，无需额外计算
_DIRECT_MAP: dict[str, str] = {
    "pe_ttm": "pe_ttm",
    "pb": "pb",
    "ps_ttm": "ps_ttm",
    "market_cap": "market_cap",
    "roe": "roe",
    "eps_basic": "eps_basic",
    "bvps": "bvps",
    "dividend_yield": "dividend_yield",
    "dividend_per_share": "dividend_per_share",
    "dividend_payout_ratio": "dividend_payout_ratio",
    "current_price": "current_price",
}

# 所有默认因子列表
DEFAULT_FACTOR_NAMES: list[str] = [
    "pe_ttm", "pb", "ps_ttm", "market_cap",
    "roe", "earnings_yield", "eps_basic", "bvps",
    "dividend_yield", "dividend_per_share", "dividend_payout_ratio",
    "debt_to_equity", "current_ratio",
]


def _safe_float(value: Any) -> float | None:
    """安全转换为 float，无法转换返回 None。"""
    if value is None:
        return None
    try:
        f = float(value)
        if f != f:  # NaN check
            return None
        return f
    except (ValueError, TypeError):
        return None


def _compute_earnings_yield(pe_ttm: float | None) -> float | None:
    """盈利收益率 = 1 / PE_TTM。"""
    if pe_ttm is not None and pe_ttm > 0:
        return 1.0 / pe_ttm
    return None


def _compute_debt_to_equity(
    total_liabilities: float | None,
    equity: float | None,
) -> float | None:
    """负债权益比 = total_liabilities / equity_attributable_to_parent。"""
    if equity is not None and equity > 0 and total_liabilities is not None:
        return total_liabilities / equity
    return None


def _compute_current_ratio(
    current_assets: float | None,
    current_liabilities: float | None,
) -> float | None:
    """流动比率 = total_current_assets / total_current_liabilities。"""
    if (
        current_assets is not None
        and current_liabilities is not None
        and current_liabilities > 0
    ):
        return current_assets / current_liabilities
    return None


# ---------------------------------------------------------------------------
#  公开 API
# ---------------------------------------------------------------------------

def compute_factors(
    data: list[dict[str, Any]],
    strategy_config: dict | None = None,
) -> FactorResult:
    """对 data_agent 产出的标准化数据计算量化因子。

    Args:
        data:            data_agent 产出的 list[dict]，每行至少含 symbol。
        strategy_config: 策略 YAML 配置（当前保留，后续可配置化选择因子集）。

    Returns:
        FactorResult —— 含 enriched_rows（原始数据 + 因子键值）、factor_sets、告警信息。

    Example:
        >>> rows = load_basic_info("data/invest.db")
        >>> result = compute_factors(rows)
        >>> result.factor_sets[0].pe_ttm
        4.5
        >>> result.factor_sets[0].earnings_yield
        0.2222
    """
    warnings: list[str] = []
    factor_sets: list[FactorSet] = []

    for row in data:
        symbol = str(row.get("symbol", ""))
        name = str(row.get("name", ""))

        if not symbol:
            warnings.append("跳过无 symbol 的行")
            continue

        # ---- 直接映射因子 ----
        pe_ttm = _safe_float(row.get("pe_ttm"))
        pb = _safe_float(row.get("pb"))
        ps_ttm = _safe_float(row.get("ps_ttm"))
        market_cap = _safe_float(row.get("market_cap"))
        roe = _safe_float(row.get("roe"))
        eps_basic = _safe_float(row.get("eps_basic"))
        bvps = _safe_float(row.get("bvps"))
        dividend_yield = _safe_float(row.get("dividend_yield"))
        dividend_per_share = _safe_float(row.get("dividend_per_share"))
        dividend_payout_ratio = _safe_float(row.get("dividend_payout_ratio"))

        # ---- 计算因子 ----
        earnings_yield = _compute_earnings_yield(pe_ttm)
        debt_to_equity = _compute_debt_to_equity(
            _safe_float(row.get("total_liabilities")),
            _safe_float(row.get("equity_attributable_to_parent")),
        )
        current_ratio = _compute_current_ratio(
            _safe_float(row.get("total_current_assets")),
            _safe_float(row.get("total_current_liabilities")),
        )

        fs = FactorSet(
            symbol=symbol,
            name=name,
            pe_ttm=pe_ttm,
            pb=pb,
            ps_ttm=ps_ttm,
            market_cap=market_cap,
            roe=roe,
            earnings_yield=earnings_yield,
            eps_basic=eps_basic,
            bvps=bvps,
            dividend_yield=dividend_yield,
            dividend_per_share=dividend_per_share,
            dividend_payout_ratio=dividend_payout_ratio,
            debt_to_equity=debt_to_equity,
            current_ratio=current_ratio,
            total_assets=_safe_float(row.get("total_assets")),
            net_profit=_safe_float(row.get("net_profit")),
        )
        factor_sets.append(fs)

    # 为每行原始数据附加因子键值（向后兼容）
    enriched_rows: list[dict[str, Any]] = []
    for fs in factor_sets:
        enriched = {
            "symbol": fs.symbol,
            "name": fs.name,
            "pe_ttm": fs.pe_ttm,
            "pb": fs.pb,
            "ps_ttm": fs.ps_ttm,
            "market_cap": fs.market_cap,
            "roe": fs.roe,
            "earnings_yield": fs.earnings_yield,
            "eps_basic": fs.eps_basic,
            "bvps": fs.bvps,
            "dividend_yield": fs.dividend_yield,
            "dividend_per_share": fs.dividend_per_share,
            "dividend_payout_ratio": fs.dividend_payout_ratio,
            "debt_to_equity": fs.debt_to_equity,
            "current_ratio": fs.current_ratio,
            "total_assets": fs.total_assets,
            "net_profit": fs.net_profit,
        }
        # 合并原始数据
        original = next((r for r in data if r.get("symbol") == fs.symbol), {})
        for k, v in original.items():
            if k not in enriched:
                enriched[k] = v
        enriched_rows.append(enriched)

    return FactorResult(
        enriched_rows=enriched_rows,
        factor_sets=factor_sets,
        factor_names=DEFAULT_FACTOR_NAMES,
        input_count=len(data),
        output_count=len(factor_sets),
        warnings=warnings,
    )


def compute_factors_with_financials(
    basic_rows: list[dict[str, Any]],
    balance_rows: list[dict[str, Any]] | None = None,
    income_rows: list[dict[str, Any]] | None = None,
) -> FactorResult:
    """在 basic_info 基础上合并财务报表数据，计算更全面的因子。

    比 compute_factors() 多了债务和流动性因子（需财报表数据）。

    Args:
        basic_rows:   basic_info 表数据。
        balance_rows: balance_sheet 表最新一期数据（按 symbol 对齐）。
        income_rows:  income_statement 表最新一期数据（按 symbol 对齐）。

    Returns:
        FactorResult
    """
    # 构建查找表
    balance_map: dict[str, dict[str, Any]] = {}
    if balance_rows:
        balance_map = {r["symbol"]: r for r in balance_rows if r.get("symbol")}

    income_map: dict[str, dict[str, Any]] = {}
    if income_rows:
        income_map = {r["symbol"]: r for r in income_rows if r.get("symbol")}

    merged: list[dict[str, Any]] = []
    for row in basic_rows:
        sym = row.get("symbol", "")
        enriched = dict(row)
        if sym in balance_map:
            enriched.update(balance_map[sym])
        if sym in income_map:
            enriched.update(income_map[sym])
        merged.append(enriched)

    return compute_factors(merged)
