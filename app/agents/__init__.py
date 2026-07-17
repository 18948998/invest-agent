"""Agent package —— MainAgent 对话中枢 + 子 Agent 工作组。

组件关系（v3）：
    MainAgent (对话中枢)
        ├── 意图识别 (intent.py)
        ├── ReAct 循环 (LLM 自主决策调什么工具)
        ├── 对话上下文管理 (_history + 压缩)
        ├── ToolRegistry     (统一工具注册 + 执行)
        │       ├── screen_stocks     → ScreenAgent
        │       ├── analyze_stock     → AnalyzeAgent
        │       ├── list_strategies   → 策略记忆层
        │       ├── refresh_data      → 数据刷新
        │       ├── save_strategy     → 策略持久化
        │       └── translate_description → NL→结构化
        └── 子 Agent 管理
                ├── ScreenAgent  (有 LLM + 任务上下文)
                └── AnalyzeAgent (有 LLM + 任务上下文)
"""

from app.agents.main_agent import MainAgent
from app.agents.tool_registry import ToolRegistry
from app.agents.tools import create_tools
from app.agents.intent import classify, classify_with_llm, Intent, Classification
from app.agents.screen_workflow import run_screen, ScreenAgent
from app.agents.analyze_workflow import run_analyze, AnalyzeAgent

__all__ = [
    "MainAgent",
    "ToolRegistry",
    "create_tools",
    "classify",
    "classify_with_llm",
    "Intent",
    "Classification",
    "run_screen",
    "run_analyze",
    "ScreenAgent",
    "AnalyzeAgent",
]
