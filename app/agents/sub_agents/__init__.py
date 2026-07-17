"""Sub-agent package —— 数据 → 因子 → 规则 三阶段流水线。

各 agent 职责明确，无状态，完全通过输入/输出契约协作：

    data_agent   —— 从 SQLite 加载标准化行情与财报数据
    factor_agent —— 计算估值、盈利、财务健康等量化因子
    rule_agent   —— 按策略配置的硬约束规则打分排序
"""
from app.agents.sub_agents.data_agent import (
    load_basic_info,
    load_by_query,
    load_latest_financial,
    get_all_table_names,
)
from app.agents.sub_agents.factor_agent import (
    FactorSet,
    FactorResult,
    compute_factors,
    compute_factors_with_financials,
    DEFAULT_FACTOR_NAMES,
)
from app.agents.sub_agents.rule_agent import (
    RuleDetail,
    RuleVerdict,
    RuleResult,
    apply_rules,
)

__all__ = [
    # data_agent
    "load_basic_info",
    "load_by_query",
    "load_latest_financial",
    "get_all_table_names",
    # factor_agent
    "FactorSet",
    "FactorResult",
    "compute_factors",
    "compute_factors_with_financials",
    "DEFAULT_FACTOR_NAMES",
    # rule_agent
    "RuleDetail",
    "RuleVerdict",
    "RuleResult",
    "apply_rules",
]
