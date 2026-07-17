"""分析工作流 —— 编排 data → factor 单股分析链路。

职责：
  - 接收单个股票代码
  - 从 basic_info + 财报表加载数据
  - 计算因子
  - 用 rich/文本格式呈现摘要

这是 MainAgent 的"分析股票"分支。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.agents.sub_agents.data_agent import load_basic_info, load_latest_financial, check_freshness, FreshnessStatus
from app.agents.sub_agents.factor_agent import FactorResult, compute_factors_with_financials

logger = logging.getLogger(__name__)


def _format_analysis_summary(
    factor_result: FactorResult,
    symbol: str,
) -> str:
    """把 FactorResult 格式化为单股分析摘要文本。"""
    if not factor_result.factor_sets:
        return f"  [!] 未找到股票 {symbol} 的数据"

    fs = factor_result.factor_sets[0]

    def f(val: Any) -> str:
        if val is None:
            return "-"
        v = float(val)
        if abs(v) >= 1e8:
            return f"{v / 1e8:.2f} 亿"
        return f"{v:.2f}"

    lines = [
        f"",
        f"  ╔══════════════════════════════════╗",
        f"  ║  {fs.name or symbol} ({symbol})",
        f"  ╠══════════════════════════════════╣",
        f"  ║  [估值]",
        f"  ║    市盈率 (PE_TTM) :  {f(fs.pe_ttm)} 倍",
        f"  ║    市净率 (PB)     :  {f(fs.pb)} 倍",
        f"  ║    市销率 (PS_TTM) :  {f(fs.ps_ttm)} 倍",
        f"  ║    总市值          :  {f(fs.market_cap)}",
        f"  ║",
        f"  ║  [盈利]",
        f"  ║    ROE            :  {f(fs.roe)}%",
        f"  ║    盈利收益率     :  {f(fs.earnings_yield) if fs.earnings_yield is not None else '-'}",
        f"  ║    EPS (基本)     :  {f(fs.eps_basic)}",
        f"  ║    每股净资产     :  {f(fs.bvps)}",
        f"  ║",
        f"  ║  [分红]",
        f"  ║    股息率         :  {f(fs.dividend_yield)}",
        f"  ║    每股股利       :  {f(fs.dividend_per_share)}",
        f"  ║    分红率         :  {f(fs.dividend_payout_ratio)}",
        f"  ║",
    ]

    if fs.debt_to_equity is not None or fs.current_ratio is not None:
        lines.append(f"  ║  [财务健康]")
        if fs.debt_to_equity is not None:
            lines.append(f"  ║    负债权益比     :  {f(fs.debt_to_equity)}")
        if fs.current_ratio is not None:
            lines.append(f"  ║    流动比率       :  {f(fs.current_ratio)}")
        if fs.total_assets is not None:
            lines.append(f"  ║    总资产         :  {f(fs.total_assets)}")

    lines.append(f"  ╚══════════════════════════════════╝")
    return "\n".join(lines)


def run_analyze(
    db_path: str | Path,
    symbol: str,
) -> dict[str, Any]:
    """执行单只股票的基本面分析。

    Args:
        db_path: SQLite 数据库路径。
        symbol:  6 位股票代码（如 "600519"）。

    Returns:
        dict:
            - factor_result: FactorResult
            - summary:       str —— 格式化文本摘要
            - symbol:        str
            - success:       bool
    """
    db = Path(db_path)

    if not symbol or len(symbol) < 6:
        return {"success": False, "symbol": symbol, "reason": "股票代码无效"}

    print(f"\n  [*] 正在分析 {symbol} 的基本面...")

    # ① 检查数据新鲜度（分析需要 basic_info + 财报）
    fs = check_freshness(str(db))
    if fs.needs_refresh:
        return {"success": False, "symbol": symbol, "freshness": fs}

    # ② 加载 basic_info
    print("  ... 加载基础数据...")
    basic_rows = load_basic_info(str(db), symbols=[symbol])
    if not basic_rows:
        print(f"  [X] 未找到股票 {symbol} 的数据")
        return {"success": False, "symbol": symbol, "reason": "basic_info 中无此股票"}

    name = basic_rows[0].get("name", symbol)
    print(f"  [OK] {name} ({symbol})")

    # ③ 加载最新财报表
    print("  ... 加载财务报表...")
    balance_rows = load_latest_financial(str(db), "balance_sheet", symbols=[symbol])
    income_rows = load_latest_financial(str(db), "income_statement", symbols=[symbol])
    print(f"  [OK] 资产负债表: {'有数据' if balance_rows else '无数据'}，利润表: {'有数据' if income_rows else '无数据'}")

    # ④ 计算因子
    factor_result = compute_factors_with_financials(basic_rows, balance_rows, income_rows)
    print(f"  [OK] 计算完成，共 {len(factor_result.factor_sets)} 个因子集")

    # ⑤ 生成摘要
    summary = _format_analysis_summary(factor_result, symbol)

    return {
        "success": True,
        "symbol": symbol,
        "name": name,
        "factor_result": factor_result,
        "summary": summary,
    }


# ==============================================================================
#  AnalyzeAgent —— 具有任务上下文的子 Agent
# ==============================================================================

class AnalyzeAgent:
    """分析子 Agent —— 自主完成单股基本面分析。

    职责：
      - 持有自己的 LLM（用于生成分析摘要）
      - 缓存任务上下文（分析结果），同一股票短期复用
      - 任务结束后释放任务上下文

    不是对话 Agent——不持有对话历史，不跟用户直接交互。

    Attributes:
        db_path:    SQLite 数据库路径。
        _llm:       领域 LLM。
        _task_state: 任务上下文缓存（symbol → 分析结果）。
    """

    def __init__(self, db_path: str | Path) -> None:
        from app.services.llm import get_chat_model

        self.db_path = str(db_path)
        self._llm = get_chat_model(temperature=0.3)
        self._task_state: dict[str, Any] = {}   # symbol → cached result
        logger.info("AnalyzeAgent 初始化完成（db=%s）", self.db_path)

    # ------------------------------------------------------------------
    #  公开接口
    # ------------------------------------------------------------------

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """执行单股分析任务。

        Args:
            task: {"action": "analyze", "symbol": "600519"}

        Returns:
            dict: {success, symbol, factor_result, summary, ...}
        """
        symbol = task.get("symbol", "")

        # ── 检查缓存 ──
        cached = self._task_state.get(symbol)
        if cached:
            logger.info("AnalyzeAgent 命中缓存: %s", symbol)
            return cached

        # ── 执行分析 ──
        result = run_analyze(self.db_path, symbol)

        # 缓存结果
        if result.get("success"):
            self._task_state[symbol] = result
            # 清理旧缓存（保留最近 5 个）
            if len(self._task_state) > 5:
                oldest = sorted(self._task_state.keys())[:-5]
                for k in oldest:
                    del self._task_state[k]

        return result

    def clear_task_state(self) -> None:
        """清空任务上下文缓存。"""
        count = len(self._task_state)
        self._task_state = {}
        if count:
            logger.info("AnalyzeAgent 任务缓存已清空（%d 条）", count)
