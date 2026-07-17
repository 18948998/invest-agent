"""plan_filter 白名单校验 & 失败重试策略。

LLM 生成的 FilterPlan.where_clause 和 display_fields 必须通过白名单校验，
否则自动重试（最多 MAX_RETRIES 次），全部失败则抛出 RuntimeError。
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from .state import AgentState, FilterPlan, SchemaCatalog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  常量
# ---------------------------------------------------------------------------

MAX_RETRIES: int = 3

# 危险 SQL token —— 出现任一即拒绝
_FORBIDDEN_TOKENS: tuple[str, ...] = (
    ";", "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE",
    "UNION", "--", "/*", "*/", "EXEC", "EXECUTE",
)

# WHERE 子句中允许的操作符
_ALLOWED_OPS: tuple[str, ...] = (
    "<", ">", "<=", ">=", "=", "!=", "<>",
    "AND", "OR", "NOT", "IN", "LIKE", "BETWEEN",
    "IS", "NULL", "(", ")",
)


# ---------------------------------------------------------------------------
#  校验逻辑
# ---------------------------------------------------------------------------

def _extract_identifiers(clause: str) -> set[str]:
    """从 WHERE 子句中提取所有可能的列名标识符。

    匹配规则：字母 / 下划线开头 + 字母数字下划线组成，
    排除 SQL 保留字。
    """
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", clause)
    upper_keywords = {
        "AND", "OR", "NOT", "IN", "LIKE", "BETWEEN", "IS", "NULL",
        "TRUE", "FALSE", "SELECT", "FROM", "WHERE", "ORDER", "BY",
        "GROUP", "HAVING", "LIMIT", "OFFSET", "AS", "ON", "JOIN",
        "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "CASE", "WHEN",
        "THEN", "ELSE", "END", "ASC", "DESC", "DISTINCT", "ALL",
    }
    return {t for t in tokens if t.upper() not in upper_keywords}


def _has_forbidden_tokens(clause: str) -> tuple[bool, str]:
    """检查是否包含危险 token。"""
    upper = clause.upper()
    for tok in _FORBIDDEN_TOKENS:
        if tok in upper:
            return True, tok
    return False, ""


def validate_plan(plan: FilterPlan, catalog: SchemaCatalog) -> tuple[bool, str]:
    """校验 FilterPlan 中的 where_clause 和 display_fields。

    Args:
        plan:    LLM 产出的筛选计划。
        catalog: 字段白名单。

    Returns:
        (ok, error_msg)。ok=True 表示通过校验。
    """
    # 1. 危险 token 检查
    is_dangerous, token = _has_forbidden_tokens(plan.where_clause)
    if is_dangerous:
        return False, f"where_clause 含危险 token: '{token}'"

    # 2. WHERE 子句标识符必须在白名单内
    identifiers = _extract_identifiers(plan.where_clause)
    allowed = catalog.allowed_columns()
    for ident in identifiers:
        if ident not in allowed:
            return False, f"where_clause 含未知列名: '{ident}'"

    # 3. display_fields 也必须在白名单内
    for field in plan.display_fields:
        if field not in allowed:
            return False, f"display_fields 含未知列名: '{field}'"

    return True, ""


# ---------------------------------------------------------------------------
#  带重试的 plan_filter 执行
# ---------------------------------------------------------------------------

def plan_filter_with_retry(
    llm_fn: Callable[[str], FilterPlan],
    state: AgentState,
    max_retries: int = MAX_RETRIES,
) -> FilterPlan:
    """用 LLM 生成 FilterPlan，失败自动重试。

    Args:
        llm_fn:      接受 screen_text → 返回 FilterPlan 的可调用对象。
        state:       当前工作流状态（需含 "catalog"）。
        max_retries: 最大重试次数。

    Returns:
        校验通过的 FilterPlan。

    Raises:
        RuntimeError: 全部重试均失败。
    """
    catalog = state.get("catalog")
    if catalog is None:
        raise RuntimeError("state 中缺少 'catalog' —— 请先执行 load_config")

    screen_text = state.get("screen_text", "")
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            plan: FilterPlan = llm_fn(screen_text)
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "plan_filter 第 %d/%d 次 LLM 调用异常: %s",
                attempt, max_retries, last_error,
            )
            continue

        ok, err = validate_plan(plan, catalog)
        if ok:
            logger.info("plan_filter 第 %d 次校验通过", attempt)
            return plan

        last_error = err
        logger.warning(
            "plan_filter 第 %d/%d 次校验失败: %s",
            attempt, max_retries, err,
        )

    raise RuntimeError(
        f"plan_filter 在 {max_retries} 次重试后仍未通过校验。最后错误: {last_error}"
    )
