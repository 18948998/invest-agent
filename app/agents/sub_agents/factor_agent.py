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

import datetime as dt
import logging
from collections import defaultdict
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
    current_price: float | None = Field(default=None, description="当前股价（元）")
    pe_3yr_avg: float | None = Field(default=None, description="PE(3年均) = 当前股价 / 最近3年EPS均值")

    # ---- 盈利因子 ----
    roe: float | None = Field(default=None, description="净资产收益率")
    earnings_yield: float | None = Field(default=None, description="盈利收益率 = 1/PE_TTM")
    eps_basic: float | None = Field(default=None, description="基本每股收益")
    bvps: float | None = Field(default=None, description="每股净资产")

    # ---- 分红因子 ----
    pretax_bonus_per_share: float | None = Field(default=None, description="税前每股股利（元/股，来自分红送转历史表最新记录）")
    dividend_years_count: int | None = Field(default=None, description="连续分红年数（从最新年份往前回溯，无间断的年份数。如最近5年每年都有分红则=5）")
    dividend_yield: float | None = Field(default=None, description="股息率")
    dividend_per_share: float | None = Field(default=None, description="每股股利")
    dividend_payout_ratio: float | None = Field(default=None, description="分红率")

    # ---- 财务健康因子（需 join 财报表） ----
    debt_to_equity: float | None = Field(default=None, description="负债权益比 = total_liabilities / equity")
    current_ratio: float | None = Field(default=None, description="流动比率 = current_assets / current_liabilities")
    total_assets: float | None = Field(default=None, description="总资产")
    net_profit: float | None = Field(default=None, description="净利润")
    long_term_borrowings: float | None = Field(default=None, description="长期借款/长期债务（来自资产负债表）")
    net_current_assets: float | None = Field(default=None, description="流动资产净额 = total_current_assets - total_current_liabilities")
    long_term_debt_to_net_ca_ratio: float | None = Field(default=None, description="长期债务/流动资产净额 = long_term_borrowings / net_current_assets")

    # ---- 历史趋势因子（需历史利润表数据） ----
    eps_growth_10yr_3yr_avg: float | None = Field(default=None, description="过去10年EPS增长率（期初3年均值vs期末3年均值），如0.5表示增长50%")
    consecutive_profitable_years: int | None = Field(default=None, description="连续盈利年数（从最新年份回溯，每年净利润>0）")


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
    "pretax_bonus_per_share": "pretax_bonus_per_share",
    "dividend_yield": "dividend_yield",
    "dividend_per_share": "dividend_per_share",
    "dividend_payout_ratio": "dividend_payout_ratio",
    "current_price": "current_price",
}

# 所有默认因子列表
DEFAULT_FACTOR_NAMES: list[str] = [
    "pe_ttm", "pb", "ps_ttm", "market_cap",
    "roe", "earnings_yield", "eps_basic", "bvps",
    "pretax_bonus_per_share", "dividend_years_count",
    "dividend_yield", "dividend_per_share", "dividend_payout_ratio",
    "debt_to_equity", "current_ratio",
    "long_term_borrowings", "net_current_assets", "long_term_debt_to_net_ca_ratio",
    "eps_growth_10yr_3yr_avg", "consecutive_profitable_years",
    "pe_3yr_avg",
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
#  历史趋势因子计算
# ---------------------------------------------------------------------------

def _compute_eps_growth_10yr_3yr_avg(
    eps_by_year: list[tuple[int, float]],
    current_year: int | None = None,
) -> float | None:
    """计算过去10年 EPS 增长率（期初/期末各取3年均值对比）。

    Args:
        eps_by_year: [(year, eps_basic), ...] 按年份降序排列（最新在前）。
        current_year: 当前年份，用于确定10年窗口。

    Returns:
        float | None: 增长率（如 0.5 表示增长 50%），数据不足则返回 None。
    """
    if not eps_by_year or len(eps_by_year) < 6:
        return None

    if current_year is None:
        current_year = dt.datetime.now().year

    # 取最近10个财年的数据
    start_year = current_year - 11  # 如2026年取2016-2025年
    recent = [(y, eps) for y, eps in eps_by_year if start_year < y < current_year]
    recent.sort(key=lambda x: x[0])  # 按年份升序

    if len(recent) < 6:
        return None

    # 期初3年均值（最早3年）
    first_3 = [eps for _, eps in recent[:3] if eps is not None]
    # 期末3年均值（最新3年）
    last_3 = [eps for _, eps in recent[-3:] if eps is not None]

    if len(first_3) < 3 or len(last_3) < 3:
        return None

    first_avg = sum(first_3) / len(first_3)
    last_avg = sum(last_3) / len(last_3)

    if first_avg == 0:
        return None

    return (last_avg - first_avg) / abs(first_avg)


def _compute_consecutive_profitable_years(
    profit_by_year: list[tuple[int, float]],
) -> int:
    """计算连续盈利年数（从最新年份回溯，每年净利润 > 0）。

    Args:
        profit_by_year: [(year, net_profit), ...] 按年份降序排列。

    Returns:
        int: 连续盈利年数。
    """
    if not profit_by_year:
        return 0

    sorted_years = sorted(profit_by_year, key=lambda x: x[0], reverse=True)
    consecutive = 0
    for y, profit in sorted_years:
        if profit is not None and profit > 0:
            consecutive += 1
        else:
            break
    return consecutive


def _compute_pe_3yr_avg(
    eps_by_year: list[tuple[int, float]],
    current_price: float | None,
) -> float | None:
    """PE(3年均) = 当前股价 / 最近3年EPS均值。

    取最近3个财年的EPS平均值（如数据不足3年返回None）。

    Args:
        eps_by_year: [(year, eps_basic), ...] 按年份降序排列（最新在前）。
        current_price: 当前股价。

    Returns:
        float | None: PE(3年均)，数据不足或EPS均值≤0时返回None。
    """
    if current_price is None or current_price <= 0:
        return None
    if not eps_by_year or len(eps_by_year) < 1:
        return None

    # 按年份降序，取最近3年
    sorted_eps = sorted(eps_by_year, key=lambda x: x[0], reverse=True)
    recent_eps = [eps for _, eps in sorted_eps[:3] if eps is not None and eps > 0]

    if len(recent_eps) < 3:
        return None

    avg_eps = sum(recent_eps) / len(recent_eps)
    if avg_eps <= 0:
        return None

    return current_price / avg_eps


def enrich_with_historical_factors(
    factor_result: FactorResult,
    income_history: dict[str, list[dict[str, Any]]],
) -> FactorResult:
    """用历史利润表数据计算并附加历史趋势因子到 FactorResult。

    Args:
        factor_result: 已有的因子计算结果。
        income_history: 按 symbol 分组的历史利润表数据。

    Returns:
        FactorResult —— 含附加历史因子的结果。
    """
    current_year = dt.datetime.now().year

    for fs in factor_result.factor_sets:
        sym = fs.symbol
        records = income_history.get(sym, [])

        # 提取 (year, eps_basic) 和 (year, net_profit) 和 (year, np_attributable)
        eps_by_year: list[tuple[int, float]] = []
        profit_by_year: list[tuple[int, float]] = []
        np_attr_by_year: list[tuple[int, float]] = []
        seen_years: set[int] = set()

        for r in records:
            rd = r.get("report_date", "")
            if not rd:
                continue
            try:
                if isinstance(rd, str):
                    y = dt.datetime.strptime(rd[:10], "%Y-%m-%d").year
                else:
                    y = rd.year if hasattr(rd, "year") else int(str(rd)[:4])
            except (ValueError, TypeError):
                continue

            # 每年只取一条（取第一条即最新 report_date 的，因为已按 DESC 排序）
            if y in seen_years:
                continue
            seen_years.add(y)

            eps = _safe_float(r.get("eps_basic"))
            profit = _safe_float(r.get("net_profit"))
            np_attr = _safe_float(r.get("net_profit_attributable_to_parent"))

            if eps is not None:
                eps_by_year.append((y, eps))
            if profit is not None:
                profit_by_year.append((y, profit))
            if np_attr is not None:
                np_attr_by_year.append((y, np_attr))

        # 计算历史因子：eps_growth 优先用 eps_basic，无数据时用归母净利润兜底
        if eps_by_year:
            fs.eps_growth_10yr_3yr_avg = _compute_eps_growth_10yr_3yr_avg(eps_by_year, current_year)
        elif np_attr_by_year:
            fs.eps_growth_10yr_3yr_avg = _compute_eps_growth_10yr_3yr_avg(np_attr_by_year, current_year)

        fs.consecutive_profitable_years = _compute_consecutive_profitable_years(profit_by_year)

        # 计算 PE(3年均)
        fs.pe_3yr_avg = _compute_pe_3yr_avg(eps_by_year, fs.current_price)

    # 更新 enriched_rows
    for enriched in factor_result.enriched_rows:
        sym = enriched.get("symbol", "")
        for fs in factor_result.factor_sets:
            if fs.symbol == sym:
                enriched["eps_growth_10yr_3yr_avg"] = fs.eps_growth_10yr_3yr_avg
                enriched["consecutive_profitable_years"] = fs.consecutive_profitable_years
                enriched["pe_3yr_avg"] = fs.pe_3yr_avg
                enriched["long_term_borrowings"] = fs.long_term_borrowings
                enriched["net_current_assets"] = fs.net_current_assets
                enriched["long_term_debt_to_net_ca_ratio"] = fs.long_term_debt_to_net_ca_ratio
                break

    return factor_result


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
        current_price = _safe_float(row.get("current_price"))
        roe = _safe_float(row.get("roe"))
        eps_basic = _safe_float(row.get("eps_basic"))
        bvps = _safe_float(row.get("bvps"))
        dividend_yield = _safe_float(row.get("dividend_yield"))
        dividend_per_share = _safe_float(row.get("dividend_per_share"))
        dividend_payout_ratio = _safe_float(row.get("dividend_payout_ratio"))
        pretax_bonus_per_share = _safe_float(row.get("pretax_bonus_per_share"))
        dividend_years_count = row.get("dividend_years_count")
        if dividend_years_count is not None:
            dividend_years_count = int(dividend_years_count) if isinstance(dividend_years_count, (int, float, str)) else None

        # ---- 兜底计算：basic_info 缺失 ROE 时从 net_profit / equity 计算 ----
        if roe is None:
            net_profit = _safe_float(row.get("net_profit"))
            equity = _safe_float(row.get("equity_attributable_to_parent"))
            if net_profit is not None and equity is not None and equity > 0:
                roe = (net_profit / equity) * 100

        # ---- 兜底计算：basic_info 缺失股息率时从 pretax_bonus / current_price 估算 ----
        if dividend_yield is None:
            pbps = _safe_float(row.get("pretax_bonus_per_share"))
            price = _safe_float(row.get("current_price"))
            if pbps is not None and price is not None and price > 0:
                dividend_yield = (pbps / price) * 100

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

        # ---- 新增计算字段 ----
        long_term_borrowings = _safe_float(row.get("long_term_borrowings"))

        total_current_assets = _safe_float(row.get("total_current_assets"))
        total_current_liabilities = _safe_float(row.get("total_current_liabilities"))
        net_current_assets: float | None = None
        if total_current_assets is not None and total_current_liabilities is not None:
            net_current_assets = total_current_assets - total_current_liabilities

        long_term_debt_to_net_ca_ratio: float | None = None
        if long_term_borrowings is not None and net_current_assets is not None and net_current_assets > 0:
            long_term_debt_to_net_ca_ratio = long_term_borrowings / net_current_assets

        fs = FactorSet(
            symbol=symbol,
            name=name,
            pe_ttm=pe_ttm,
            pb=pb,
            ps_ttm=ps_ttm,
            market_cap=market_cap,
            current_price=current_price,
            roe=roe,
            earnings_yield=earnings_yield,
            eps_basic=eps_basic,
            bvps=bvps,
            pretax_bonus_per_share=pretax_bonus_per_share,
            dividend_years_count=dividend_years_count,
            dividend_yield=dividend_yield,
            dividend_per_share=dividend_per_share,
            dividend_payout_ratio=dividend_payout_ratio,
            debt_to_equity=debt_to_equity,
            current_ratio=current_ratio,
            total_assets=_safe_float(row.get("total_assets")),
            net_profit=_safe_float(row.get("net_profit")),
            long_term_borrowings=long_term_borrowings,
            net_current_assets=net_current_assets,
            long_term_debt_to_net_ca_ratio=long_term_debt_to_net_ca_ratio,
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
            "pretax_bonus_per_share": fs.pretax_bonus_per_share,
            "dividend_years_count": fs.dividend_years_count,
            "dividend_yield": fs.dividend_yield,
            "dividend_per_share": fs.dividend_per_share,
            "dividend_payout_ratio": fs.dividend_payout_ratio,
            "debt_to_equity": fs.debt_to_equity,
            "current_ratio": fs.current_ratio,
            "total_assets": fs.total_assets,
            "net_profit": fs.net_profit,
            "long_term_borrowings": fs.long_term_borrowings,
            "net_current_assets": fs.net_current_assets,
            "long_term_debt_to_net_ca_ratio": fs.long_term_debt_to_net_ca_ratio,
            "eps_growth_10yr_3yr_avg": fs.eps_growth_10yr_3yr_avg,
            "consecutive_profitable_years": fs.consecutive_profitable_years,
            "pe_3yr_avg": fs.pe_3yr_avg,
            "current_price": fs.current_price,
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
    dividend_rows: list[dict[str, Any]] | None = None,
) -> FactorResult:
    """在 basic_info 基础上合并财务报表数据，计算更全面的因子。

    比 compute_factors() 多了债务和流动性因子（需财报表数据），
    以及税前每股股利（需分红送转历史表数据）。

    Args:
        basic_rows:    basic_info 表数据。
        balance_rows:  balance_sheet 表最新一期数据（按 symbol 对齐）。
        income_rows:   income_statement 表最新一期数据（按 symbol 对齐）。
        dividend_rows: dividend_history 表数据（按 symbol 对齐，取每股最新分红）。

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

    # 分红：按 symbol 聚合，取最新 pretax_bonus_per_share + 计算连续分红年数
    dividend_map: dict[str, float] = {}
    dividend_years_map: dict[str, int] = {}
    if dividend_rows:
        per_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in dividend_rows:
            sym = str(r.get("symbol", ""))
            if sym:
                per_symbol[sym].append(r)

        for sym, records in per_symbol.items():
            # --- 最新 pretax_bonus_per_share（记录已按 plan_notice_date DESC 排序）---
            for r in records:
                val = _safe_float(r.get("pretax_bonus_per_share"))
                if val is not None:
                    dividend_map[sym] = val
                    break

            # --- 连续分红年数：从最新年份往前回溯 ---
            years: set[int] = set()
            for r in records:
                date_str = r.get("plan_notice_date", "")
                if not date_str:
                    continue
                try:
                    y = dt.datetime.strptime(date_str[:10], "%Y-%m-%d").year
                    years.add(y)
                except (ValueError, AttributeError):
                    continue

            if years:
                sorted_y = sorted(years, reverse=True)
                consecutive = 1
                for i in range(1, len(sorted_y)):
                    if sorted_y[i - 1] - sorted_y[i] == 1:
                        consecutive += 1
                    else:
                        break
                dividend_years_map[sym] = consecutive

    merged: list[dict[str, Any]] = []
    for row in basic_rows:
        sym = row.get("symbol", "")
        enriched = dict(row)
        if sym in balance_map:
            enriched.update(balance_map[sym])
        if sym in income_map:
            enriched.update(income_map[sym])
        if sym in dividend_map:
            enriched["pretax_bonus_per_share"] = dividend_map[sym]
        if sym in dividend_years_map:
            enriched["dividend_years_count"] = dividend_years_map[sym]
        merged.append(enriched)

    return compute_factors(merged)
