"""主对话 agent —— 用户所有交互的唯一入口。

职责（v5）：
  1. 对话上下文管理（_history，压缩，reset）
  2. ReAct 循环 + classify_intent / list_strategies 两个 meta 工具
  3. 意图识别后路由到子 Agent（ScreenAgent / AnalyzeAgent）
  4. 自由对话（问候、闲聊）

MainAgent 有 2 个工具：
  - classify_intent：分析用户意图，路由到子 Agent
  - list_strategies：列出项目内已有策略（只读查询，不需要子 Agent）
领域工具（screen_stocks, analyze_stock 等）是子 Agent 的职责。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agents.intent import Intent, classify, classify_with_llm

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "standardized" / "invest.db"

# ==============================================================================
#  阈值常量
# ==============================================================================

DEFAULT_MAX_HISTORY_TOKENS = 80_000
DEFAULT_MAX_HISTORY_ROUNDS = 10
KEEP_RECENT_ROUNDS = 3

# ==============================================================================
#  System Prompt —— 描述 classify_intent 和 list_strategies 两个工具
# ==============================================================================

SYSTEM_PROMPT = """你是一个专业的 A 股投资研究助手。你可以使用以下两个工具：

- classify_intent：分析用户意图（筛选/分析/帮助/退出/闲聊）
- list_strategies：列出所有可用的投资策略

## 对话规则

1. **意图分析优先**：收到用户输入后，先调用 classify_intent 判断其意图。不要猜测或跳过。
2. **策略查询直接调**：当用户问"有哪些策略/列出策略/可用策略/策略列表"时，直接调用 list_strategies 获取真实数据。
3. **中文回复**：始终用中文回复，保持简洁专业。
4. **辅助说明**：当系统返回了筛选/分析结果后，请用自然语言向用户简要总结关键发现。
5. **不要臆造数据**：只引用对话中实际出现的数据。
6. **直接对话**：如果 classify_intent 返回 unknown，直接友好回复用户即可。

## 回复风格

- 筛选结果 → 简要总结发现、关键数字，提醒用户可对感兴趣的代码深入分析
- 分析结果 → 突出关键指标和结论
- 策略列表 → 如实告知有哪些策略及其描述
- 错误提示 → 说明原因 + 下一步建议"""

# 压缩上下文用的 prompt
_COMPRESSION_PROMPT = """你是一个对话摘要助手。请将以上对话历史压缩为一段简洁的摘要（200 字以内），
保留以下关键信息：
- 用户做了哪些操作（筛选了什么、分析了哪些股票）
- 得到的关键结果和结论
- 用户未完成的意图或后续需要跟进的事项

只输出摘要本身，不要加任何前缀或说明。"""


# ==============================================================================
#  classify_intent 工具
# ==============================================================================

class ClassifyIntentInput(BaseModel):
    """意图分类工具的参数。"""
    user_input: str = Field(description="用户的原始输入文本")


class MainAgent:
    """对话中枢：ReAct 循环 + classify_intent / list_strategies → 子 Agent 委托 / 自由对话。

    v5 变更：
      - ReAct 循环绑 2 个 meta 工具：classify_intent + list_strategies
      - LLM 通过 function calling 自主决定是否分析意图或查策略
      - 领域工具（screen_stocks, analyze_stock 等）完全归子 Agent
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        from app.services.llm import get_chat_model

        self.db_path = str(db_path or _DEFAULT_DB)

        # ── LLM（对话用，温度稍高以支持自然对话）──
        self._llm: Any = get_chat_model(temperature=0.3)
        logger.info("LLM 初始化成功")

        # ── 分类 LLM（独立实例，temperature=0 确保精确）──
        self._classify_llm: Any = get_chat_model(temperature=0.0)

        # ── 子 Agent（懒加载）──
        self._screen_agent: Any = None
        self._analyze_agent: Any = None

        # ── 对话上下文 ──
        self._history: list[Any] = []
        self._round_count: int = 0
        self._max_history_tokens = DEFAULT_MAX_HISTORY_TOKENS
        self._max_history_rounds = DEFAULT_MAX_HISTORY_ROUNDS

        # ── ReAct 参数 ──
        self._max_turns: int = 6

        # ── Meta 工具（classify_intent + list_strategies）──
        self._meta_tools = self._build_meta_tools()
        logger.info("MainAgent 初始化完成，%d 个 meta 工具", len(self._meta_tools))

    # ------------------------------------------------------------------
    #  构建 meta 工具
    # ------------------------------------------------------------------

    def _build_meta_tools(self) -> list[Any]:
        """构建 MainAgent 的两个工具：classify_intent + list_strategies。"""

        def _classify_intent(user_input: str) -> str:
            """分析用户意图，返回结构化分类结果。"""
            result = classify_with_llm(user_input, llm=self._classify_llm)
            return json.dumps({
                "intent": result.intent.name.lower(),
                "confidence": result.confidence,
                "symbol": result.extracted_symbol,
                "strategy_name": result.strategy_name,
            }, ensure_ascii=False)

        def _list_strategies() -> str:
            """列出所有已注册的筛选策略（只读查询，直接从策略存储读取真实数据）。"""
            from app.memory.strategy_memory import list_strategies
            result = list_strategies()
            strategies = result.get("strategies", {})
            if not strategies:
                return "当前没有任何已注册策略。"
            lines = [f"共 {result.get('count', 0)} 个策略:"]
            for key, info in strategies.items():
                name = info.get("name", key)
                desc = info.get("description", "")[:80]
                strategy_desc = info.get("strategy_desc", "")[:80]
                text_desc = strategy_desc or desc or ""
                lines.append(f"  • {name}（key: {key}）: {text_desc}")
            if result.get("default"):
                lines.append(f"默认策略: {result['default']}")
            return "\n".join(lines)

        return [
            StructuredTool.from_function(
                func=_classify_intent,
                name="classify_intent",
                description=(
                    "分析用户当前输入的意图。返回 intent 字段为: screen（筛选股票）、"
                    "screen_save（筛选并保存策略）、analyze（分析单只股票）、"
                    "help（求助）、quit（退出）、unknown（闲聊/无法识别）。"
                    "同时提取 symbol（6位股票代码）和 strategy_name（策略英文key）。"
                    "每次收到用户新输入时，如果是新的任务请求（而非只针对刚才结果的追问），"
                    "就应该调用此工具进行分析。"
                ),
                args_schema=ClassifyIntentInput,
            ),
            StructuredTool.from_function(
                func=_list_strategies,
                name="list_strategies",
                description=(
                    "列出所有可用的投资筛选策略。"
                    "当用户问'有哪些策略/列出策略/可用策略/策略列表/什么策略'时调用此工具。"
                    "返回真实策略数据，不要自己编造策略名。"
                ),
            ),
        ]

    # ------------------------------------------------------------------
    #  公开入口
    # ------------------------------------------------------------------

    def run(self) -> None:
        """启动对话循环。"""
        self._greet()
        self._run_loop()

    # ------------------------------------------------------------------
    #  对话循环
    # ------------------------------------------------------------------

    def _greet(self) -> None:
        print()
        print("╔" + "═" * 54 + "╗")
        print("║" + "  您好！我是您的投资研究助手。".center(48) + "║")
        print("║" + "".center(48) + "║")
        print("║" + "  直接告诉我你想做什么，例如：".center(43) + "║")
        print("║" + "    • 帮我筛选 pe_ttm<15 且 roe>10% 的股票".center(46) + "║")
        print("║" + "    • 分析一下 600519".center(32) + "║")
        print("║" + "    • 有哪些策略？".center(24) + "║")
        print("║" + "    • 输入 q 或 退出 结束对话".center(36) + "║")
        print("║" + "    • 输入 /reset 重置对话".center(36) + "║")
        print("╚" + "═" * 54 + "╝")

    def _run_loop(self) -> None:
        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  再见！")
                break

            if not user_input:
                continue

            if not self._pre_filter(user_input):
                break

    # ------------------------------------------------------------------
    #  预处理：退出 / 帮助 / 重置（关键词，不消耗 LLM）
    # ------------------------------------------------------------------

    def _pre_filter(self, raw_input: str) -> bool:
        c = classify(raw_input)

        if c.intent == Intent.QUIT:
            print("\n  再见！投资顺利")
            return False

        if c.intent == Intent.HELP:
            self._print_help()
            return True

        if raw_input.lower() in ("/reset", "/clear", "重置对话"):
            self.reset()
            print("\n  对话历史已清空，请说新的话题。")
            return True

        # ── 其余走 ReAct ──
        self._handle_with_react(raw_input)
        return True

    # ------------------------------------------------------------------
    #  ReAct 循环（两个工具：classify_intent + list_strategies）
    # ------------------------------------------------------------------

    def _handle_with_react(self, raw_input: str) -> None:
        """ReAct：LLM 用 classify_intent 分析意图 → 代码路由到子 Agent。"""
        # ── 初始化 system prompt ──
        if not self._history:
            self._history = [SystemMessage(content=SYSTEM_PROMPT)]

        # ── 追加用户输入 ──
        self._history.append(HumanMessage(content=raw_input))

        llm_with_tools = self._llm.bind_tools(self._meta_tools)

        # ── ReAct 循环 ──
        for turn in range(self._max_turns):
            response = llm_with_tools.invoke(self._history)
            self._history.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # 无工具调用 → 最终回复
                print(f"\n{response.content}")
                self._round_count += 1
                self._maybe_compress()
                return

            # ── 处理工具调用 ──
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
                tid = tc.get("id", "")

                if name == "classify_intent":
                    user_text = args.get("user_input", raw_input)
                    intent_data = self._do_classify(user_text)
                    intent_name = intent_data.get("intent", "unknown")
                    logger.info("classify_intent → %s (conf=%.2f)", intent_name, intent_data.get("confidence", 0))
                    print(f"\n  [Agent] 意图分析: {intent_name}")

                    # ── 路由到子 Agent ──
                    if intent_name in ("screen", "screen_save"):
                        sub_result = self._do_screen(raw_input, intent_data)
                        self._history.append(ToolMessage(
                            content=sub_result,
                            tool_call_id=tid,
                        ))
                        # 继续 ReAct —— LLM 会基于结果生成回复

                    elif intent_name == "analyze":
                        sub_result = self._do_analyze(intent_data)
                        self._history.append(ToolMessage(
                            content=sub_result,
                            tool_call_id=tid,
                        ))

                    else:
                        # unknown / help / quit → 把分类结果直接返回给 LLM
                        self._history.append(ToolMessage(
                            content=json.dumps(intent_data, ensure_ascii=False),
                            tool_call_id=tid,
                        ))

                elif name == "list_strategies":
                    from app.memory.strategy_memory import list_strategies
                    ls_result = list_strategies()
                    strategies = ls_result.get("strategies", {})
                    if not strategies:
                        result = "当前没有任何已注册策略。"
                    else:
                        lines = [f"共 {ls_result.get('count', 0)} 个策略:"]
                        for key, info in strategies.items():
                            name = info.get("name", key)
                            desc = info.get("description", "") or info.get("strategy_desc", "") or ""
                            lines.append(f"  • {name}（key: {key}）: {desc[:80]}")
                        if ls_result.get("default"):
                            lines.append(f"默认策略: {ls_result['default']}")
                        result = "\n".join(lines)
                    print(f"\n{result}")
                    self._history.append(ToolMessage(
                        content=result,
                        tool_call_id=tid,
                    ))

                else:
                    logger.warning("未知工具调用: %s", name)

        # 达到最大 ReAct 轮次 → 强制总结
        logger.warning("达到 ReAct 最大轮次 %d，强制总结", self._max_turns)
        self._history.append(HumanMessage(
            content="已达到最大工具调用次数。请根据已有所有信息直接给出最终回答，不要再调用工具。"
        ))
        final = llm_with_tools.invoke(self._history)
        self._history.append(final)
        print(f"\n{final.content}")
        self._round_count += 1
        self._maybe_compress()

    # ------------------------------------------------------------------
    #  意图分类
    # ------------------------------------------------------------------

    def _do_classify(self, user_input: str) -> dict[str, Any]:
        """调用 classify_with_llm 做 LLM 意图分类。
        
        同时检查是否已有对话上下文（追问场景），有则传更多上下文给分类器。
        """
        # 如果有历史对话，把最近一轮的摘要也传给分类器
        result = classify_with_llm(user_input, llm=self._classify_llm)
        return {
            "intent": result.intent.name.lower(),
            "confidence": result.confidence,
            "symbol": result.extracted_symbol,
            "strategy_name": result.strategy_name,
        }

    # ------------------------------------------------------------------
    #  子 Agent 委托
    # ------------------------------------------------------------------

    def _do_screen(self, raw_input: str, intent_data: dict[str, Any]) -> str:
        """执行筛选并返回格式化结果。"""
        agent = self._get_screen_agent()

        task: dict[str, Any] = {
            "action": "screen",
            "conditions": raw_input,
            "strategy_name": intent_data.get("strategy_name", ""),
            "save": intent_data.get("intent") == "screen_save",
        }

        result = agent.execute(task)
        table = result.get("table", "(无结果)")
        top_symbols = result.get("top_symbols", [])
        save_name = result.get("save_name", "")
        unsupported = result.get("unsupported", [])
        translated_summary = result.get("translated_summary", [])

        # 直接打印表格给用户
        print(f"\n{table}")

        # 返回文本摘要给 LLM，让它基于此生成自然语言回复
        parts = [f"筛选完成。"]
        if translated_summary:
            parts.append(f"\n策略条件翻译结果：")
            parts.extend(translated_summary)
        if top_symbols:
            parts.append(f"得分前5名: {', '.join(top_symbols[:5])}")
        if save_name:
            parts.append(f"策略已保存为: {save_name}")
        if unsupported:
            parts.append(f"\n注意：有 {len(unsupported)} 条规则因系统限制被跳过，已在上方标注。")
        parts.append(f"\n已展示给用户的表格:\n{table}")

        return "\n".join(parts)

    def _do_analyze(self, intent_data: dict[str, Any]) -> str:
        """执行单股分析并返回格式化结果。"""
        symbol = intent_data.get("symbol", "")
        if not symbol:
            return f"意图为 analyze 但未提取到股票代码。请询问用户要分析哪只股票。"

        agent = self._get_analyze_agent()
        result = agent.execute({"action": "analyze", "symbol": symbol})

        if not result.get("success"):
            reason = result.get("reason", "未知错误")
            return f"分析 {symbol} 失败: {reason}。请告知用户原因并建议下一步。"

        summary = result.get("summary", "")
        # 直接打印格式化摘要
        print(f"{summary}")

        return f"分析完成。{symbol} 的基本面摘要已展示给用户。请基于以下数据给出简要评价：\n{summary}"

    # ------------------------------------------------------------------
    #  上下文管理
    # ------------------------------------------------------------------

    def reset(self) -> None:
        old_rounds = self._round_count
        self._history = []
        self._round_count = 0
        if self._screen_agent:
            self._screen_agent.clear_task_state()
        if self._analyze_agent:
            self._analyze_agent.clear_task_state()
        logger.info("对话历史已重置（之前 %d 轮）", old_rounds)

    def _maybe_compress(self) -> None:
        tokens = self._estimate_tokens()
        rounds = self._round_count

        if tokens < self._max_history_tokens and rounds < self._max_history_rounds:
            return

        logger.info("触发压缩: tokens=%d, rounds=%d", tokens, rounds)
        print("\n  [*] 上下文已达上限，正在压缩历史对话...")

        try:
            self._compress()
            print(f"  [OK] 压缩完成，降至 ~{self._estimate_tokens()} token")
        except Exception as exc:
            logger.exception("压缩失败")
            print(f"  [!] 压缩失败: {exc}，将截断早期对话")
            self._truncate_fallback()

    def _compress(self) -> None:
        keep = KEEP_RECENT_ROUNDS * 2
        if len(self._history) <= 1 + keep:
            self._truncate_fallback()
            return

        old = self._history[1:-keep] if keep > 0 else self._history[1:]
        if not old:
            return

        summary = self._summarize(old)
        self._history = [self._history[0], SystemMessage(content=summary)] + self._history[-keep:]
        self._round_count = KEEP_RECENT_ROUNDS

    def _summarize(self, messages: list[Any]) -> str:
        text = ""
        for msg in messages:
            role = type(msg).__name__.replace("Message", "").lower()
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                if len(content) > 500:
                    content = content[:500] + "...[截断]"
                text += f"[{role}] {content}\n"

        try:
            response = self._llm.invoke([
                SystemMessage(content="你是一个精准的对话摘要助手。"),
                HumanMessage(content=f"总结以下对话（200字内）：\n\n{text}\n\n{_COMPRESSION_PROMPT}"),
            ])
            return response.content.strip()
        except Exception:
            return f"[摘要] 共 {len(messages)} 条历史消息。"

    def _truncate_fallback(self) -> None:
        keep = KEEP_RECENT_ROUNDS * 2
        if len(self._history) > 1 + keep:
            self._history = [self._history[0]] + self._history[-(keep):]
            self._round_count = KEEP_RECENT_ROUNDS

    def _estimate_tokens(self) -> int:
        total = 0
        for msg in self._history:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += sum(
                    len(t.get("text", "")) if isinstance(t, dict) else len(str(t))
                    for t in content
                )
        return int(total * 0.5)

    # ------------------------------------------------------------------
    #  子 Agent 懒加载
    # ------------------------------------------------------------------

    def _get_screen_agent(self) -> Any:
        if self._screen_agent is None:
            from app.agents.screen_workflow import ScreenAgent
            self._screen_agent = ScreenAgent(self.db_path)
        return self._screen_agent

    def _get_analyze_agent(self) -> Any:
        if self._analyze_agent is None:
            from app.agents.analyze_workflow import AnalyzeAgent
            self._analyze_agent = AnalyzeAgent(self.db_path)
        return self._analyze_agent

    # ------------------------------------------------------------------
    #  帮助
    # ------------------------------------------------------------------

    def _print_help(self) -> None:
        strategies_summary = "格雷厄姆价值策略（默认）"
        try:
            from app.memory.strategy_memory import list_strategies
            info = list_strategies()
            names = [v.get("name", k) for k, v in info.get("strategies", {}).items()]
            if names:
                strategies_summary = "、".join(names)
        except Exception:
            pass

        for line in [
            "",
            "  +-------------------------------------------------+",
            "  |  投资研究助手 -- 使用指南                      |",
            "  +-------------------------------------------------+",
            "  |                                                 |",
            "  |  筛选推荐                                      |",
            "  |     直接说你的条件，例如:                       |",
            "  |       [帮我找市盈率低于15的股票]               |",
            "  |       [筛选 pe<10 pb<1 的标的]                |",
            "  |       [推荐高roe低估值股票]                   |",
            f"  |     可用策略: {strategies_summary[:38]:<45}|",
            "  |                                                 |",
            "  |  单股分析                                      |",
            "  |      [分析一下 600519]                        |",
            "  |      [000001 基本面怎么样]                     |",
            "  |                                                 |",
            "  |  策略管理                                      |",
            "  |      [记住 pe<10 为超低估策略]                |",
            "  |      [有哪些策略]                              |",
            "  |                                                 |",
            "  |  数据管理 | [刷新数据]                          |",
            "  |  对话管理 | /reset 清空上下文, q 退出            |",
            "  |                                                 |",
            "  +-------------------------------------------------+",
        ]:
            print(line)
