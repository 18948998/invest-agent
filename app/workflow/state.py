"""Workflow state definitions —— AgentState + FilterPlan + AnalysisReport + SchemaCatalog."""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
#  LLM 产出
# ---------------------------------------------------------------------------

class FilterPlan(BaseModel):
    """LLM 从自然语言筛选描述解析出的可执行筛选计划。

    where_clause   —— 安全校验后用于数据库 SELECT
    display_fields —— 用户要看到的表格列名（必须在字段白名单内）
    """

    where_clause: str = Field(
        default="",
        description="SQL WHERE clause, e.g. 'pe_ttm < 20 AND pb < 2 AND market_cap > 100e8'",
    )
    display_fields: list[str] = Field(
        default_factory=list,
        description="Column names to display in the per‑stock table",
    )
    reasoning: str = Field(
        default="",
        description="How the natural‑language description was translated",
    )


class AnalysisReport(BaseModel):
    """基本面分析报告 —— 每只候选股票一份，由 analyst_agent 产出。"""

    symbol: str = ""
    name: str = ""
    summary: str = ""
    strengths: list[str] = Field(default_factory=list, description="优势亮点")
    risks: list[str] = Field(default_factory=list, description="风险点")
    score: float = Field(ge=0, le=100, default=0, description="综合评分 0‑100")
    verdict: str = Field(default="观察", description="建议 / 观察 / 回避")
    data_points: dict[str, Any] = Field(
        default_factory=dict,
        description="分析中用到的关键数据点",
    )


# ---------------------------------------------------------------------------
#  字段白名单
# ---------------------------------------------------------------------------

class SchemaCatalog(BaseModel):
    """从 configs/fundamental_fields/*.yaml 构建的字段白名单。

    所有 LLM 产出的列名 / WHERE 子句中的标识符都必须在此白名单内，
    否则拒绝执行 —— 防止 SQL 注入。
    """

    columns: set[str] = Field(default_factory=set, description="全部已知列名")
    table_columns: dict[str, set[str]] = Field(
        default_factory=dict,
        description="{table_name: {col1, col2, ...}}",
    )

    def allowed_columns(self) -> set[str]:
        """返回允许使用的全部列名集合。"""
        return self.columns

    def is_allowed(self, column: str) -> bool:
        """检查单个列名是否合法。"""
        return column in self.columns


# ---------------------------------------------------------------------------
#  LangGraph 共享状态
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """LangGraph 工作流中所有节点共享的 dict 状态。

    不需要 total=True —— 各节点增量填充字段。
    """

    # ---- 输入 ----
    db_path: str                        # SQLite 数据库路径
    screen_text: str                    # 纯自然语言筛选描述（.md 文件内容）
    screen_mode: str                    # 筛选模式名，用于输出文件命名

    # ---- plan_filter 产出 ----
    catalog: SchemaCatalog              # 字段白名单
    plan: FilterPlan | None             # LLM 解析出的筛选计划

    # ---- filter_and_analyze 产出 ----
    candidates: list[dict]              # 含 display_fields 的候选行（DB 一次查完）
    processed: dict[str, AnalysisReport]  # symbol → 基本面分析报告（线程池汇聚）

    # ---- review_loop 推进 ----
    review_cursor: int                  # 当前审到第几个候选
    approved: list[AnalysisReport]      # 用户确认的报告
    rejected: list[str]                 # 用户拒绝的 symbol

    # ---- finalize 产出 ----
    final_report: str | None            # 汇总报告文本（或存盘路径）
