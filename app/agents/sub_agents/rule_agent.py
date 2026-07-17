"""Rule agent —— 按策略配置中定义的规则对候选标的打分排序。

职责：
  - 接收 factor_agent 产出的因子值 + 策略 YAML 配置
  - 按 hard_filters 逐条检查，不合格的打低分或排除
  - 按 universe 限制选股范围
  - 输出按 score 降序排列的候选列表 + 每只的 rule_details

设计原则：
  - 纯规则引擎，不调 LLM —— 打分逻辑完全由配置驱动
  - 规则失败不等于零分 —— 用扣分机制，让"接近"的候选仍有机会
  - 每个规则的权重可通过配置 tune

契约：
  输入:  candidates: list[dict]    —— factor_agent 产出的富化数据
         strategy_config: dict     —— 策略 YAML 配置（含 hard_filters）
  输出:  RuleResult
         - scored:   list[RuleVerdict]   —— 通过或有分的候选
         - excluded: list[str]           —— 被排除的 symbol 及原因
         - metadata: 统计信息
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
#  中文字段别名表 —— 供 translate_description 的 LLM prompt 使用
#  中文关键词 → 标准字段名，解决 LLM "账面值" 不知道映射到 pb 的问题
# =============================================================================

FIELD_ALIASES_ZH: dict[str, str] = {
    # 估值
    "市盈率": "pe_ttm",       "pe": "pe_ttm",         "ttm市盈率": "pe_ttm",
    "市净率": "pb",           "pb": "pb",             "账面值": "pb",
    "价格账面值": "pb",       "价格账面值比": "pb",      "市账率": "pb",
    # 盈利
    "净资产收益率": "roe",     "roe": "roe",           "每股收益": "eps_basic",
    # 分红
    "股息率": "dividend_yield", "分红率": "dividend_yield",
    # 规模
    "市值": "market_cap",     "总市值": "market_cap",
    # 营收（利润表）
    "营业收入": "revenue",    "销售额": "revenue",     "年销售额": "revenue",
    "营收": "revenue",
    # 资产（资产负债表）
    "总资产": "total_assets", "资产总额": "total_assets",
    # 财务健康
    "负债权益比": "debt_to_equity",   "资产负债率": "debt_to_equity",
    "负债": "debt_to_equity",
    "流动比率": "current_ratio",       "流动比": "current_ratio",
    "流动资产": "current_ratio",
    "长期债务": "debt_to_equity",
    # 复合指标
    "市盈率×市净率": "pe_pb",  "市盈率乘市净率": "pe_pb",
    "pe乘pb": "pe_pb",        "市盈率与价格账面值之比的乘积": "pe_pb",
    "市盈率与市净率乘积": "pe_pb",  "市盈率.*乘积": "pe_pb",
    # 股市术语（映射到 pe_ttm）
    "利润的": "pe_ttm",       "股价": "pe_ttm",
    # 每股
    "每股基本收益": "eps_basic", "eps": "eps_basic",
}

# =============================================================================
#  计算字段 —— 跨多字段复合计算，如 pe_pb = pe_ttm × pb
#  由 apply_rules 执行时动态计算
# =============================================================================

COMPUTED_FIELDS: dict[str, dict[str, Any]] = {
    "pe_pb": {
        "fields": ["pe_ttm", "pb"],
        "display": "PE × PB",
        "description": "市盈率与市净率的乘积（格雷厄姆深度价值条件）",
    },
}

# =============================================================================
#  展示标签映射
# =============================================================================

_FIELD_LABEL: dict[str, str] = {
    "pe_ttm": "PE(ttm)",        "pb": "PB",             "roe": "ROE(%)",
    "dividend_yield": "股息率(%)", "market_cap": "市值(元)",
    "eps_basic": "EPS",         "debt_to_equity": "负债权益比",
    "current_ratio": "流动比率",  "pe_pb": "PE×PB",
    "revenue": "营收(元)",       "total_assets": "总资产(元)",
}


# ---------------------------------------------------------------------------
#  数据模型
# ---------------------------------------------------------------------------

class RuleDetail(BaseModel):
    """单条规则的匹配结果。"""

    rule_name: str = ""
    passed: bool = True
    skipped: bool = False      # True = 该字段数据缺失，跳过而非判失败
    actual_value: float | None = None
    expected: str = ""         # 规则的文字描述，如 "0 < pe_ttm <= 15"
    score_contribution: float = 0.0  # 这条规则贡献的分数


class RuleVerdict(BaseModel):
    """一只候选标的的完整评分结果。"""

    symbol: str = ""
    name: str = ""
    total_score: float = 0.0
    rules: list[RuleDetail] = Field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0     # 因数据缺失而跳过的规则数
    factors: dict[str, Any] = Field(default_factory=dict)


class RuleResult(BaseModel):
    """规则评分的完整产出。"""

    scored: list[RuleVerdict] = Field(default_factory=list)
    excluded: list[dict[str, str]] = Field(default_factory=list)  # [{symbol, reason}]
    total_candidates: int = 0
    passed_candidates: int = 0
    excluded_count: int = 0


# ---------------------------------------------------------------------------
#  规则执行
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float | None = None) -> float | None:
    """安全转换为 float。"""
    if value is None:
        return default
    try:
        f = float(value)
        if f != f:
            return default
        return f
    except (ValueError, TypeError):
        return default


def _score_factor(
    value: float | None,
    rule: dict[str, Any],
    weight: float = 1.0,
) -> tuple[float, float, bool, bool]:
    """对单个因子打分，返回 (raw_score, weighted_score, passed, skipped)。

    打分策略（格雷厄姆风格）：
      - 完全通过规则 → 满分 1.0 * weight
      - 数据缺失 → 跳过，给满分，不扣分（标记为 skipped）
      - 在硬约束容差内 → 根据偏离程度线性扣分
      - 严重偏离 → 0 分

    Args:
        value:  因子实际值。
        rule:   {min, max} 规则容差。
        weight: 该规则的权重。

    Returns:
        (raw, weighted, passed, skipped)
    """
    rule_min = None if rule.get("min") is None else float(rule["min"])
    rule_max = None if rule.get("max") is None else float(rule["max"])

    if value is None:
        return 1.0, weight, True, True   # 数据缺失 → 跳过，给满分

    passed = True
    raw = 1.0

    if rule_min is not None and value < rule_min:
        if rule_min > 0 and value > 0:
            ratio = value / rule_min
            raw = max(0.0, ratio)  # 线性扣：value=min*0.7 得 0.7 分
        else:
            raw = 0.0
        passed = False

    if rule_max is not None and value > rule_max:
        if rule_max > 0:
            ratio = rule_max / max(value, 1e-9)
            raw = min(raw, max(0.0, ratio))
        else:
            raw = min(raw, 0.0)
        passed = False

    return raw, raw * weight, passed, False


def _apply_universe(
    candidates: list[dict[str, Any]],
    universe: dict[str, Any],
) -> list[dict[str, Any]]:
    """按 universe 配置预过滤候选池。

    支持：
      - exclude_st: True → 排除名称含 "ST"/"*ST"/"PT"/"退" 的股票
      - min_listing_days: >= N → 排除上市不足 N 天的股票
      - 隐式：排除 price=0/None 的股票（停牌/退市/无行情数据）
    """
    if not universe:
        return candidates

    result = list(candidates)
    excluded_count = 0

    # 排除 price=0 或 None 的股票（停牌/退市/无交易）
    before = len(result)
    result = [
        r for r in result
        if _safe_float(r.get("current_price")) and _safe_float(r.get("current_price")) > 0
    ]
    diff = before - len(result)
    if diff:
        logger.info("universe: 排除 %d 只（price=0 或无行情）", diff)

    # exclude_st：排除 S T/*ST/PT/退市 股
    if universe.get("exclude_st"):
        _ST_PATTERNS = ("ST", "PT", "退")
        before = len(result)
        result = [
            r for r in result
            if not any(p in str(r.get("name", "")) for p in _ST_PATTERNS)
        ]
        excluded_count = before - len(result)
        if excluded_count:
            logger.info("universe.exclude_st: 排除 %d 只", excluded_count)

    # min_listing_days：排除上市不足 N 天的次新股
    import datetime as _dt
    min_days = universe.get("min_listing_days")
    if isinstance(min_days, (int, float)) and min_days > 0:
        cutoff = (_dt.datetime.now().date()
                  - _dt.timedelta(days=int(min_days)))
        before = len(result)
        filtered: list[dict[str, Any]] = []
        for r in result:
            ld = r.get("listing_date")
            if ld is None:
                filtered.append(r)  # 无上市日期的不排除
                continue
            try:
                if isinstance(ld, str):
                    ld_date = _dt.datetime.strptime(ld[:10], "%Y-%m-%d").date()
                else:
                    ld_date = ld
                if ld_date <= cutoff:
                    filtered.append(r)
            except (ValueError, TypeError):
                filtered.append(r)
        result = filtered
        diff = before - len(result)
        if diff:
            logger.info("universe.min_listing_days(%d): 排除 %d 只", int(min_days), diff)

    return result


def apply_rules(
    candidates: list[dict[str, Any]],
    strategy_config: dict[str, Any] | None = None,
    *,
    filter_only: bool = False,
) -> RuleResult:
    """按策略配置的硬约束规则对候选标的打分或简单筛选。

    Args:
        candidates:       factor_agent 产出的富化数据（list[dict]）。
        strategy_config:  策略 YAML 配置字典。未提供时使用内置默认规则。
        filter_only:      True 时只做条件筛选（全部通过才保留），不评分不排序。

    Returns:
        RuleResult —— 含 scored 和 excluded 列表。
    """
    if strategy_config is None:
        strategy_config = _default_graham_config()

    # ---- universe 预过滤 ----
    universe = strategy_config.get("universe", {})
    candidates = _apply_universe(candidates, universe)

    screen_cfg = strategy_config.get("screen", {})
    hard_filters: dict[str, dict[str, Any]] = screen_cfg.get("hard_filters", {})

    if not hard_filters:
        logger.warning("策略配置中未定义 hard_filters，所有候选直接通过")
        verdicts = [
            RuleVerdict(
                symbol=str(c.get("symbol", "")),
                name=str(c.get("name", "")),
                total_score=1.0,
                passed_count=0,
                failed_count=0,
            )
            for c in candidates
        ]
        return RuleResult(
            scored=verdicts,
            total_candidates=len(candidates),
            passed_candidates=len(verdicts),
        )

    # 规则名称到数据字段的映射（基础字段）
    _field_map = {
        "pe_ttm": "pe_ttm",
        "pb": "pb",
        "roe": "roe",
        "dividend_yield": "dividend_yield",
        "market_cap": "market_cap",
        "eps_basic": "eps_basic",
        "debt_to_equity": "debt_to_equity",
        "current_ratio": "current_ratio",
        "revenue": "revenue",            # 利润表：营业收入
        "total_assets": "total_assets",  # 资产负债表：总资产
    }

    # 辅助：从候选数据中提取规则值（支持计算字段）
    def _resolve_value(candidate: dict[str, Any], rule_name: str) -> float | None:
        if rule_name in COMPUTED_FIELDS:
            pe = _safe_float(candidate.get("pe_ttm"))
            pb = _safe_float(candidate.get("pb"))
            if pe is not None and pb is not None:
                return pe * pb
            return None
        field = _field_map.get(rule_name, rule_name)
        return _safe_float(candidate.get(field))

    # 辅助：构建计算字段的因子值
    def _build_computed_factors(candidate: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, meta in COMPUTED_FIELDS.items():
            result[name] = _resolve_value(candidate, name)
        return result

    verdicts: list[RuleVerdict] = []
    excluded: list[dict[str, str]] = []

    for candidate in candidates:
        symbol = str(candidate.get("symbol", ""))
        name = str(candidate.get("name", ""))

        if not symbol:
            excluded.append({"symbol": "", "reason": "缺 symbol"})
            continue

        details: list[RuleDetail] = []
        passed_count = 0
        failed_count = 0
        skipped_count = 0
        total_score = 0.0

        for rule_name, rule_def in hard_filters.items():
            actual_value = _resolve_value(candidate, rule_name)

            raw_score, weighted_score, passed, skipped = _score_factor(actual_value, rule_def)

            total_score += weighted_score
            if passed:
                passed_count += 1
            else:
                failed_count += 1
            if skipped:
                skipped_count += 1

            # 生成规则描述
            rule_min = rule_def.get("min")
            rule_max = rule_def.get("max")
            desc_parts: list[str] = []
            if rule_min is not None:
                desc_parts.append(f">= {rule_min}")
            if rule_max is not None:
                desc_parts.append(f"<= {rule_max}")
            expected = " AND ".join(desc_parts) if desc_parts else "(无约束)"

            details.append(RuleDetail(
                rule_name=rule_name,
                passed=passed,
                skipped=skipped,
                actual_value=actual_value,
                expected=expected,
                score_contribution=weighted_score,
            ))

        verdicts.append(RuleVerdict(
            symbol=symbol,
            name=name,
            total_score=round(total_score, 4),
            rules=details,
            passed_count=passed_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            factors={
                **{k: candidate.get(k) for k in _field_map.values() if k in candidate},
                **_build_computed_factors(candidate),
            },
        ))

    # --- filter_only 模式：只保留全部通过的 ---
    if filter_only:
        verdicts = [v for v in verdicts if v.failed_count == 0]
        # 不排序，保持原始顺序（或按 symbol 排序）
        verdicts.sort(key=lambda v: v.symbol)

    # 按总分降序排列（非 filter_only 模式）
    if not filter_only:
        verdicts.sort(key=lambda v: v.total_score, reverse=True)

    return RuleResult(
        scored=verdicts,
        excluded=excluded,
        total_candidates=len(candidates),
        passed_candidates=len([v for v in verdicts if v.failed_count == 0]),
        excluded_count=len(excluded),
    )


# ---------------------------------------------------------------------------
#  内置默认规则（格雷厄姆深度价值策略）
# ---------------------------------------------------------------------------

def _default_graham_config() -> dict[str, Any]:
    """当未提供策略配置文件时使用的内置默认规则。"""
    return {
        "screen": {
            "hard_filters": {
                "pe_ttm":          {"min": 0,    "max": 15},
                "pb":              {"min": 0,    "max": 1.5},
                "roe":             {"min": 10},
                "dividend_yield":  {"min": 0},
                "market_cap":      {"min": 50e8},
            }
        }
    }
