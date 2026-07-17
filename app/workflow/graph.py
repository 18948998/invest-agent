"""LangGraph 工作流 —— 5 节点状态机。

节点：  load_config → plan_filter → filter_and_analyze → review_loop ⇄ finalize
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from .state import AgentState


# ==============================================================================
#  节点实现（当前为骨架，后续逐步填入真实逻辑）
# ==============================================================================

def load_config(state: AgentState) -> AgentState:
    """节点 ①：读取自然语言筛选文件 → screen_text；构建字段白名单。

    期望输入：  state["screen_mode"]（可选，没给就用默认配置）
    产出：      state["screen_text"]（自然语言字符串）
               state["catalog"]    （SchemaCatalog 白名单）
    """
    # TODO: 读 configs/screens/<mode>.md → screen_text
    # TODO: 读 configs/fundamental_fields/*.yaml → SchemaCatalog
    return state


def plan_filter(state: AgentState) -> AgentState:
    """节点 ②：LLM 把自然语言描述转成 FilterPlan；白名单安全校验。

    期望输入：  state["screen_text"] + state["catalog"]
    产出：      state["plan"]（FilterPlan，含 where_clause + display_fields）
    重试策略：  见 retry.py —— LLM 输出非法时自动重试 N 次
    """
    # TODO: 调 LLM 解析 screen_text → FilterPlan
    # TODO: 用 retry.validate_plan 校验，不过就重试
    return state


def filter_and_analyze(state: AgentState) -> AgentState:
    """节点 ③：边筛边出边分析 —— 核心节点。

    流程：
        1. SELECT display_fields FROM basic_info WHERE plan.where_clause → 一次性拿全部候选
        2. 遍历候选列表，对每一只：
           a. 立即用 rich 打印该股票表格
           b. 同时 submit analyst_agent 到 ThreadPoolExecutor（并发基本面分析）
        3. 遍历完等待全部 Future 完成 → 汇聚 AnalysisReport 到 state["processed"]

    期望输入：  state["db_path"] + state["plan"]
    产出：      state["candidates"]（含 display_fields 的候选行）
               state["processed"]（symbol → AnalysisReport）
               state["review_cursor"] = 0
    """
    # TODO: query_candidates(db_path, plan.where_clause, plan.display_fields)
    # TODO: with ThreadPoolExecutor(max_workers=5) as pool:
    #           for row in rows:
    #               print_table(row, fields=plan.display_fields, index=i+1)
    #               futures[pool.submit(analyst_agent_analyze, row, state)] = row["symbol"]
    #           for fut in as_completed(futures):
    #               processed[futures[fut]] = fut.result()
    state["review_cursor"] = 0
    return state


def review_loop(state: AgentState) -> AgentState:
    """节点 ④：人在回路 —— interrupt() 暂停，逐个交用户检查。

    每轮：
        1. 取 candidates[cursor] → 对应 AnalysisReport
        2. 调用 interrupt() 暂停，返回候选表格 + 分析报告
        3. 等待用户在终端输入决策：
           - "approve"        → 记入 approved
           - "reject"         → 记入 rejected
           - "revise <意见>"  → 带意见重跑 analyst_agent，再次 interrupt
        4. cursor += 1

    期望输入：  state["candidates"] + state["processed"] + state["review_cursor"]
    产出：      state["approved"] / state["rejected"] / state["review_cursor"]
    """
    candidates = state.get("candidates", [])
    processed = state.get("processed", {})
    cursor = state.get("review_cursor", 0)

    if cursor >= len(candidates):
        return state

    candidate = candidates[cursor]
    symbol = candidate.get("symbol", "unknown")
    report = processed.get(symbol)

    # 暂停，等用户输入
    decision: str = interrupt(
        {
            "index": cursor + 1,
            "total": len(candidates),
            "symbol": symbol,
            "candidate": candidate,
            "report": report.model_dump() if report else None,
            "prompt": "请输入: approve / reject / revise <修改意见>",
        }
    )

    if isinstance(decision, str) and decision.strip().lower() == "approve":
        if report:
            state.setdefault("approved", []).append(report)
    elif isinstance(decision, str) and decision.strip().lower().startswith("revise"):
        # TODO: 提取修改意见 → 带 feedback 重跑 analyst_agent → 更新 processed[symbol]
        #       然后再次 interrupt 交用户确认
        state.setdefault("approved", []).append(report) if report else None
    else:
        state.setdefault("rejected", []).append(symbol)

    state["review_cursor"] = cursor + 1
    return state


def finalize(state: AgentState) -> AgentState:
    """节点 ⑤：汇总 approved → 写入 output/ 目录 + 终端打印。

    期望输入：  state["approved"] + state["screen_mode"]
    产出：      state["final_report"]（汇总报告路径或文本）
    """
    # TODO: 把 approved 列表写入 output/screen_<mode>_<ts>.md
    # TODO: 用 rich 打印汇总表格
    return state


# ==============================================================================
#  条件路由
# ==============================================================================

def should_continue_review(state: AgentState) -> str:
    """review_loop → review_loop（还有未审）或 finalize（全部审完）。"""
    cursor = state.get("review_cursor", 0)
    total = len(state.get("candidates", []))
    return "review_loop" if cursor < total else "finalize"


# ==============================================================================
#  Graph Builder
# ==============================================================================

def build_graph() -> StateGraph:
    """构建并返回未编译的 StateGraph。"""
    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node("load_config", load_config)
    builder.add_node("plan_filter", plan_filter)
    builder.add_node("filter_and_analyze", filter_and_analyze)
    builder.add_node("review_loop", review_loop)
    builder.add_node("finalize", finalize)

    # 边
    builder.set_entry_point("load_config")
    builder.add_edge("load_config", "plan_filter")
    builder.add_edge("plan_filter", "filter_and_analyze")
    builder.add_edge("filter_and_analyze", "review_loop")

    # review_loop 自循环 或 → finalize
    builder.add_conditional_edges(
        "review_loop",
        should_continue_review,
        {
            "review_loop": "review_loop",
            "finalize": "finalize",
        },
    )
    builder.add_edge("finalize", END)

    return builder


# 编译好的全局 graph 实例，供外部直接引入
graph = build_graph().compile()
