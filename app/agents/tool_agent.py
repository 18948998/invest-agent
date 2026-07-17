"""[已废弃] ToolAgent 已迁移至 MainAgent + ToolRegistry。

v3 架构：
  - ToolAgent 的 ReAct 循环 → MainAgent._handle_with_react()
  - ToolAgent 的工具注册/执行 → ToolRegistry (tool_registry.py)
  - 对话上下文管理 → MainAgent._history

本文件保留 ToolAgent 作为 ToolRegistry 的别名，向后兼容。
新代码请直接使用 ToolRegistry。
"""

from __future__ import annotations

import warnings

from app.agents.tool_registry import ToolRegistry

warnings.warn(
    "ToolAgent 已废弃，请使用 ToolRegistry。"
    "ReAct 循环已迁移至 MainAgent。",
    DeprecationWarning,
    stacklevel=2,
)

# 向后兼容别名
ToolAgent = ToolRegistry
