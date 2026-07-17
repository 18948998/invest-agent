"""Agent registry —— 把 agent / tool 名称映射到可调用的实现。

v2 架构说明：
  - AGENT_MAP: 顶层入口注册（MainAgent.run() 即对话循环）
  - TOOL_MAP:  保留供向后兼容，但新架构中 LLM 通过 app/agents/tools.py
               的 StructuredTool 自主发现和调用工具，不再依赖此字典。
               如需添加新工具，请在 tools.py 的 create_tools() 中注册。
"""

from __future__ import annotations

from typing import Any, Callable

from app.agents.main_agent import MainAgent
from app.agents.screen_workflow import run_screen, translate_description
from app.agents.analyze_workflow import run_analyze
from app.memory.strategy_memory import list_strategies

# 按名称查找的注册表
AGENT_MAP: dict[str, Callable[..., Any]] = {
    "main":    lambda: MainAgent().run(),
    "screen":  run_screen,
    "analyze": run_analyze,
}

# agent 可调用的工具注册表（保留向后兼容，新代码请用 app.agents.tools.create_tools）
TOOL_MAP: dict[str, dict[str, Any]] = {
    "list_strategies": {
        "fn": list_strategies,
        "description": "列出所有已注册的筛选策略",
    },
    "translate_description": {
        "fn": translate_description,
        "description": "用 LLM 将策略 NL 描述翻译成结构化筛选条件",
    },
    "refresh_basic_info": {
        "fn": lambda db_path=None: _call_refresh("basic_info", db_path),
        "description": "刷新全市场 basic_info（股价/PE/PB/市值）",
    },
    "refresh_financials": {
        "fn": lambda db_path=None: _call_refresh("financials", db_path),
        "description": "从东方财富拉取三张财报表",
    },
}


def _call_refresh(tool: str, db_path: str | None = None) -> dict[str, Any]:
    """延迟导入刷新函数，按名称调用。"""
    from pathlib import Path

    from app.services.data_refresher import refresh_basic_info, refresh_financials

    db = (
        Path(db_path)
        if db_path
        else Path(__file__).parent.parent.parent / "data" / "standardized" / "invest.db"
    )

    if tool == "basic_info":
        return refresh_basic_info(db)
    else:
        return refresh_financials(db)
