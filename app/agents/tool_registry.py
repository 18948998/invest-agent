"""ToolRegistry —— 统一工具注册与执行层。

职责：
  - 注册所有可用的 LangChain StructuredTool
  - 提供 LLM 接口（bind_tools 用）和代码接口（子 Agent 用）
  - 完全无状态，任何组件皆可使用

用法：
    from app.agents.tool_registry import ToolRegistry

    registry = ToolRegistry(db_path)
    # LLM 侧：bind_tools(registry.tools)
    llm_with_tools = llm.bind_tools(registry.tools)
    # 代码侧：直接调用
    result = registry.execute("screen_stocks", {"conditions": "pe<15"})
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ToolRegistry:
    """统一工具注册表 —— 双向接口（LLM function calling + 代码直接调用）。

    Attributes:
        tools: LangChain StructuredTool 列表（给 LLM bind_tools 用）。
        _tool_map: name → tool 快速查找字典。
    """

    def __init__(self, db_path: str) -> None:
        from app.agents.tools import create_tools

        self.db_path = db_path
        self.tools = create_tools(db_path)
        self._tool_map: dict[str, Any] = {t.name: t for t in self.tools}

        logger.info(
            "ToolRegistry 初始化完成，%d 个工具: %s",
            len(self.tools), list(self._tool_map.keys()),
        )

    # ------------------------------------------------------------------
    #  LLM 接口 —— 给 MainAgent 的 ReAct 循环用
    # ------------------------------------------------------------------

    def execute(self, name: str, args: dict[str, Any]) -> str:
        """执行单个工具调用，返回结果字符串。

        给 MainAgent 的 ReAct 循环使用 —— LLM 通过 function calling
        指定 tool name + args，这里执行并返回结果。

        Args:
            name: 工具名称（如 "screen_stocks"）。
            args: 工具参数字典。

        Returns:
            工具执行结果字符串。
        """
        tool = self._tool_map.get(name)
        if tool is None:
            return f"错误: 工具 '{name}' 不存在。可用: {list(self._tool_map.keys())}"

        try:
            result = tool.invoke(args)
            return str(result)
        except Exception as exc:
            logger.exception("工具 %s 执行失败", name)
            return (
                f"工具 {name} 执行出错: {exc}。"
                f"请尝试调整参数后重试，或使用其他工具。"
            )

    # ------------------------------------------------------------------
    #  预处理钩子 —— 留给 MainAgent 做特殊逻辑（如 needs_refresh 确认）
    # ------------------------------------------------------------------

    def needs_confirm_refresh(self, result: str) -> bool:
        """检查工具返回结果是否要求确认刷新数据。"""
        return "needs_refresh" in result
