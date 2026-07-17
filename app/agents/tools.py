"""Agent 可调用工具 —— LangChain StructuredTool 封装。

每个工具都有严格的 Pydantic args_schema，LLM 通过 function calling
自主决定何时调用哪个工具。这是从"过程式路由"到"LLM 自主决策"的核心桥梁。

工具清单（6 个）：
  screen_stocks          —— 用自然语言条件筛选全市场股票
  analyze_stock          —— 分析单只股票基本面
  list_strategies        —— 列出所有已注册策略
  refresh_data           —— 刷新数据库（股价/财报）
  save_strategy          —— 保存用户自定义策略
  translate_description  —— 将策略 NL 描述翻译为结构化条件
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ==============================================================================
#  Pydantic 参数 Schema
# ==============================================================================

class ScreenStocksInput(BaseModel):
    """筛选股票的参数。"""
    conditions: str = Field(
        description="自然语言筛选条件，如 'pe_ttm小于15，roe大于10%，pb小于1.5'"
    )
    strategy_name: str = Field(
        default="",
        description="可选策略名称（如 'graham'），留空则仅用 conditions 即时筛选",
    )


class AnalyzeStockInput(BaseModel):
    """分析单股参数。"""
    symbol: str = Field(description="6 位股票代码，如 '600519'")


class RefreshDataInput(BaseModel):
    """刷新数据参数。"""
    data_type: Literal["basic_info", "financials", "both"] = Field(
        default="both",
        description="basic_info=股价/PE/PB/市值, financials=三张财报表, both=全部",
    )


class SaveStrategyInput(BaseModel):
    """保存策略参数。"""
    name: str = Field(description="策略唯一标识（英文），如 'my_deep_value'")
    conditions: str = Field(description="筛选条件自然语言描述，如 'pe_ttm<10 且 pb<0.8 且 roe>15%'")


class TranslateDescriptionInput(BaseModel):
    """翻译策略描述参数。"""
    description: str = Field(description="策略的自然语言描述，需翻译为结构化筛选条件")


# ==============================================================================
#  工具工厂 —— 把 db_path 注入到每个工具函数
# ==============================================================================

def _format_refresh_needed(fs: object) -> str:
    """把 FreshnessStatus 转成 agent 能理解的 JSON 提示。"""
    summary = getattr(fs, "summary", "数据过期")
    needs = []
    if getattr(fs, "price_need", False):
        needs.append("basic_info")
    if getattr(fs, "financial_need", False):
        needs.append("financials")
    return json.dumps({
        "status": "needs_refresh",
        "message": f"数据需要刷新: {summary}。请先调用 refresh_data 刷新数据后重试。",
        "missing_data": needs,
    }, ensure_ascii=False)


def create_tools(db_path: str) -> list[StructuredTool]:
    """创建绑定到指定数据库路径的工具列表。

    每个工具是独立的 StructuredTool，可被 LLM 通过 function calling 调用。
    """
    db = Path(db_path)

    # ==========================================================================
    #  1. screen_stocks —— 全市场条件筛选
    # ==========================================================================

    def _screen_stocks(conditions: str, strategy_name: str = "") -> str:
        """筛选全市场股票。传入自然语言条件和可选策略名称。"""
        from app.agents.screen_workflow import run_screen
        from app.agents.sub_agents.data_agent import FreshnessStatus

        result = run_screen(
            str(db),
            strategy_name=strategy_name,
            raw_input=conditions,
            save=False,
        )

        # 数据过期 → 提示 agent 先刷新
        fs = result.get("freshness")
        if fs is not None and isinstance(fs, FreshnessStatus) and fs.needs_refresh:
            return _format_refresh_needed(fs)

        table = result.get("table", "(无结果)")
        rule_result = result.get("rule_result")

        top_hint = ""
        if rule_result and rule_result.scored:
            top5 = [v.symbol for v in rule_result.scored[:5]]
            if top5:
                top_hint = f"\n\n得分前 5: {', '.join(top5)}。可用 analyze_stock 深入分析任意一只。"

        return f"{table}{top_hint}"

    screen_tool = StructuredTool.from_function(
        func=_screen_stocks,
        name="screen_stocks",
        description=(
            "用自然语言条件筛选全市场股票。适用场景：用户说「帮我找/筛选/推荐 XX 条件的股票」。"
            "参数 conditions 是自然语言（如 'pe_ttm<15，roe>10%'），strategy_name 可选。"
            "返回时如果提示 needs_refresh，必须先用 refresh_data 刷新数据再重试。"
        ),
        args_schema=ScreenStocksInput,
    )

    # ==========================================================================
    #  2. analyze_stock —— 单股基本面分析
    # ==========================================================================

    def _analyze_stock(symbol: str) -> str:
        """分析单只股票基本面。"""
        from app.agents.analyze_workflow import run_analyze
        from app.agents.sub_agents.data_agent import FreshnessStatus

        result = run_analyze(str(db), symbol)

        fs = result.get("freshness")
        if fs is not None and isinstance(fs, FreshnessStatus) and fs.needs_refresh:
            return _format_refresh_needed(fs)

        if result.get("success"):
            return result.get("summary", f"{symbol} 分析完成但无摘要")
        return f"分析 {symbol} 失败: {result.get('reason', '未知错误')}。请确认数据库中存在该股票数据。"

    analyze_tool = StructuredTool.from_function(
        func=_analyze_stock,
        name="analyze_stock",
        description=(
            "分析单只股票的基本面（估值/盈利/分红/财务健康）。"
            "传入 6 位股票代码。适用场景：用户说「分析/看看/怎么样 + 股票代码」。"
            "返回时如果提示 needs_refresh，必须先用 refresh_data 刷新数据再重试。"
        ),
        args_schema=AnalyzeStockInput,
    )

    # ==========================================================================
    #  3. list_strategies —— 列出策略
    # ==========================================================================

    def _list_strategies() -> str:
        """列出所有已注册策略。"""
        from app.memory.strategy_memory import list_strategies as _ls
        result = _ls()
        strategies = result.get("strategies", {})
        if not strategies:
            return "当前没有注册任何策略。你可以用 save_strategy 创建自定义策略。"
        lines = [f"共 {result.get('count', 0)} 个策略:"]
        for key, info in strategies.items():
            name = info.get("name", key)
            desc = info.get("description", "")[:80]
            tags = info.get("tags", [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"  • {name} ({key}){tag_str}: {desc}")
        if result.get("default"):
            lines.append(f"\n默认策略: {result['default']}")
        return "\n".join(lines)

    list_tool = StructuredTool.from_function(
        func=_list_strategies,
        name="list_strategies",
        description=(
            "列出所有可用筛选策略。适用场景：用户问「有哪些策略/列出策略/策略列表」。"
        ),
    )

    # ==========================================================================
    #  4. refresh_data —— 刷新数据库
    # ==========================================================================

    def _refresh_data(data_type: str = "both") -> str:
        """刷新数据库数据。"""
        from app.services.data_refresher import refresh_basic_info, refresh_financials

        results: list[str] = []

        if data_type in ("basic_info", "both"):
            r = refresh_basic_info(db)
            if r.get("success"):
                results.append(f"股价数据刷新成功: {r.get('count', 0)}/{r.get('total', '?')} 只")
            else:
                results.append(f"股价数据刷新失败: {r.get('reason', '未知')}")

        if data_type in ("financials", "both"):
            r = refresh_financials(db)
            if r.get("success"):
                results.append(f"财报数据刷新成功: {r.get('count', 0)} 只")
            else:
                results.append(f"财报数据刷新失败: {r.get('reason', '未知')}")

        summary = "；".join(results) if results else "未执行任何刷新操作"
        return f"数据刷新完成。{summary}。现在可以重新执行筛选或分析了。"

    refresh_tool = StructuredTool.from_function(
        func=_refresh_data,
        name="refresh_data",
        description=(
            "刷新/更新数据库。当 screen_stocks 或 analyze_stock 返回 needs_refresh 时必须调用。"
            "data_type: basic_info(股价PE/PB/市值), financials(财报), both(全部)。"
        ),
        args_schema=RefreshDataInput,
    )

    # ==========================================================================
    #  5. save_strategy —— 保存自定义策略
    # ==========================================================================

    def _save_strategy(name: str, conditions: str) -> str:
        """保存用户自定义筛选策略。"""
        from app.agents.screen_workflow import parse_screen_conditions, extracted_to_strategy_config
        from app.memory.strategy_memory import save_strategy as _ss

        extracted = parse_screen_conditions(conditions)
        filters = extracted.get("filters", {})

        if not filters:
            return (
                f"无法从「{conditions[:60]}」中提取有效筛选规则。"
                f"请提供更明确的条件，例如：pe_ttm<10, pb<0.8, roe>15%"
            )

        config = extracted_to_strategy_config(filters, name)
        result = _ss(name, config, name)

        if result.get("success"):
            return (
                f"策略「{name}」已保存成功！"
                f"筛选条件: {conditions}。"
                f"下次可直接对 agent 说「用 {name} 策略筛选」。"
            )
        return f"保存策略失败: {result}"

    save_tool = StructuredTool.from_function(
        func=_save_strategy,
        name="save_strategy",
        description=(
            "保存用户自定义筛选策略。name 为英文标识，conditions 为自然语言筛选条件。"
            "适用场景：用户说「记住/保存/收藏 + 条件 + 策略名」。"
        ),
        args_schema=SaveStrategyInput,
    )

    # ==========================================================================
    #  6. translate_description —— 翻译策略描述
    # ==========================================================================

    def _translate_description(description: str) -> str:
        """把策略 NL 描述翻译为结构化条件。"""
        from app.agents.screen_workflow import translate_description as _td
        result = _td(description)
        return json.dumps(result, ensure_ascii=False, indent=2)

    translate_tool = StructuredTool.from_function(
        func=_translate_description,
        name="translate_description",
        description=(
            "把策略描述翻译为结构化筛选条件（min/max 约束）。"
            "适用场景：用户问「XX 策略具体有哪些条件」。"
        ),
        args_schema=TranslateDescriptionInput,
    )

    return [screen_tool, analyze_tool, list_tool, refresh_tool, save_tool, translate_tool]
