"""意图分类器 —— 将用户自然语言映射为可路由的意图枚举。

两档实现：
  - classify(text)       → 关键词匹配（快速、离线可用）
  - classify_with_llm()  → LLM 分类（更智能，后续接入）

意图类型：
  screen       —— 筛选推荐类（"帮我找低估值股票"）
  screen_save  —— 筛选 + 持久化为命名策略（"记住 pe<10 为超低估策略"）
  analyze      —— 单股分析类（"分析一下 600519"）
  help         —— 求助 / 询问能力
  quit         —— 退出
  unknown      —— 无法识别
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class Intent(Enum):
    SCREEN = auto()
    SCREEN_SAVE = auto()
    ANALYZE = auto()
    HELP = auto()
    QUIT = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class Classification:
    intent: Intent
    confidence: float          # 0.0 ~ 1.0
    extracted_symbol: str = ""  # analyze 类时提取出的股票代码
    strategy_name: str = ""     # screen 类时提取出的策略名称（如 "graham"）
    reason: str = ""            # 为什么分到这个意图


# ==============================================================================
#  关键词/正则规则
# ==============================================================================

# 股票代码模式：6 位数字，或以 sh/sz/ SH/SZ 开头
_STOCK_CODE_PATTERN = re.compile(
    r"(?:sh|sz|SH|SZ)?(\d{6})",
)

# 退出类关键词
_QUIT_KEYWORDS: tuple[str, ...] = (
    "退出", "再见", "quit", "bye", "q", "exit", "结束",
)

# 帮助类关键词
_HELP_KEYWORDS: tuple[str, ...] = (
    "帮助", "help", "怎么用", "使用说明", "功能", "能做什么", "h",
    "指南", "说明",
)

# 筛选推荐类关键词
_SCREEN_KEYWORDS: tuple[str, ...] = (
    "筛选", "推荐", "找", "选股", "筛选股票", "推荐股票",
    "选", "格雷厄姆", "价值投资", "低估值", "便宜",
    "有哪些", "列出", "扫描", "海选", "发现",
    "screen", "scan", "find", "search", "filter",
    "帮我找", "帮我选", "帮我筛选", "给我推荐",
    "挑", "选点", "找找",
)

# 分析类关键词
_ANALYZE_KEYWORDS: tuple[str, ...] = (
    "分析", "看看", "怎么样", "如何", "基本面",
    "财报", "估值", "研究", "深度",
    "analyze", "analysis", "research", "look",
    "介绍", "说说", "讲一下", "聊聊",
)

# 保存策略 / 起名类关键词（筛选 + 持久化）
_SCREEN_SAVE_KEYWORDS: tuple[str, ...] = (
    "记住", "保存策略", "起名", "命名", "取个名字",
    "存为策略", "收藏策略", "记为",
)


# ==============================================================================
#  策略名称 → 英文 key 的映射（用于从自然语言中提取策略名）
# ==============================================================================

_STRATEGY_KEY_MAP: dict[str, str] = {
    "格雷厄姆": "graham",
    "graham": "graham",
    "价值投资": "graham",
    "深度价值": "graham",
    "低估值": "graham",
    "防御": "defance-strategy",
    "defance": "defance-strategy",
    "防御型": "defance-strategy",
    "defance-strategy": "defance-strategy",
    "超低估": "超低估",
}


def _extract_strategy_name(text: str) -> str:
    """从用户输入中提取策略名称，返回英文 key（如 "graham"）。

    查找顺序：
      1. 硬编码关键词映射（预设策略，毫秒级）
      2. "策略N" 编号匹配 → 按注册表索引查找
      3. 策略注册表（用户自定义策略，按中文名匹配）
    """
    t = text.lower()
    # ---- 1. 预设关键词 ----
    for keyword, key in _STRATEGY_KEY_MAP.items():
        if keyword.lower() in t:
            return key

    # ---- 1.5. "策略N" / "第N个策略" 编号匹配 ----
    import re
    num_match = re.search(r'(?:第\s*)?策略\s*(\d+)', text)
    if num_match:
        try:
            n = int(num_match.group(1))
            from app.memory.strategy_memory import list_strategies
            info = list_strategies()
            strategies = info.get("strategies", {})
            keys = list(strategies.keys())
            if 1 <= n <= len(keys):
                return keys[n - 1]
        except Exception:
            pass

    # ---- 2. 用户自定义策略（按中文名遍历注册表） ----
    try:
        from app.memory.strategy_memory import list_strategies
        info = list_strategies()
        for key, entry in info.get("strategies", {}).items():
            display_name = entry.get("name", key)
            if display_name and display_name in text:
                return key
    except Exception:
        pass

    return ""


# ==============================================================================
#  分类器
# ==============================================================================

def classify(text: str) -> Classification:
    """用关键词 + 正则匹配做意图分类（离线、毫秒级）。

    Args:
        text: 用户输入的自然语言字符串。

    Returns:
        Classification
    """
    t = text.strip().lower()

    if not t:
        return Classification(Intent.UNKNOWN, 0.0, reason="空输入")

    # 1. 退出
    if t in _QUIT_KEYWORDS or any(kw in t for kw in _QUIT_KEYWORDS):
        return Classification(Intent.QUIT, 1.0, reason="命中退出关键词")

    # 2. 帮助（放在退出之后、其他之前，因为"功能"可能出现在其他场景）
    if any(kw in t for kw in _HELP_KEYWORDS):
        return Classification(Intent.HELP, 0.9, reason="命中帮助关键词")

    # 3. 提取股票代码
    code_match = _STOCK_CODE_PATTERN.search(text)
    symbol = code_match.group(1) if code_match else ""

    # 4. 判断：两种关键词都不匹配时的默认处理
    has_analyze_kw = any(kw in t for kw in _ANALYZE_KEYWORDS)
    has_screen_kw = any(kw in t for kw in _SCREEN_KEYWORDS)
    has_save_kw = any(kw in t for kw in _SCREEN_SAVE_KEYWORDS)

    # ---- 保存策略意图（记住 xx 为 yy 策略） ----
    # "记住 pe<10 pb<0.8 为超低估策略" → SCREEN_SAVE
    # "保存策略 低pe高roe" → SCREEN_SAVE
    if has_save_kw:
        return Classification(
            Intent.SCREEN_SAVE, 0.9,
            reason="命中保存策略关键词",
        )

    # 筛选词 + 无股票代码 → 一定是筛选（如 "帮我找低估值股票"）
    if has_screen_kw and not symbol:
        return Classification(
            Intent.SCREEN, 0.9,
            strategy_name=_extract_strategy_name(text),
            reason="命中筛选关键词（无股票代码）",
        )

    # 分析词 + 有股票代码 → 一定是分析（如 "分析一下600519"）
    if has_analyze_kw and symbol:
        return Classification(Intent.ANALYZE, 0.95, symbol, reason="分析关键词 + 股票代码")

    # 既有筛选词又有分析词且有股票代码 → 偏向分析（如 "找找600519怎么样"）
    if has_screen_kw and has_analyze_kw and symbol:
        return Classification(Intent.ANALYZE, 0.8, symbol, reason="分析+筛选关键词 + 股票代码")

    # 筛选词（有股票代码但无分析词）→ 筛选（如 "筛选一下银行股"）
    if has_screen_kw:
        return Classification(
            Intent.SCREEN, 0.85,
            strategy_name=_extract_strategy_name(text),
            reason="命中筛选关键词",
        )

    # 只有分析关键词（无股票代码）→ 分析意图但缺代码
    if has_analyze_kw:
        return Classification(Intent.ANALYZE, 0.6, reason="分析关键词（需提供股票代码）")

    # 只有股票代码没有关键词语境 → 默认分析
    if symbol:
        return Classification(Intent.ANALYZE, 0.8, symbol, "检测到股票代码，默认分析")

    # 5. 兜底
    return Classification(Intent.UNKNOWN, 0.0, reason="匹配不到任何意图")


def classify_with_llm(text: str, llm=None) -> Classification:
    """用 LLM 做意图分类（需提供 langchain ChatModel）。

    当 llm 未提供时退化为关键词分类。
    """
    if llm is None:
        return classify(text)

    # 动态从注册表获取策略列表
    strategies_desc = ""
    try:
        from app.memory.strategy_memory import list_strategies
        info = list_strategies()
        strategies = info.get("strategies", {})
        if strategies:
            lines = []
            for i, (key, entry) in enumerate(strategies.items(), 1):
                name = entry.get("name", key)
                desc = (entry.get("description", "") or "")[:80]
                lines.append(f"  {i}. {name}（key: {key}）: {desc}")
            strategies_desc = "\n".join(lines)
    except Exception:
        pass

    if not strategies_desc:
        strategies_desc = "- graham / 格雷厄姆 / 价值投资 / 深度价值 / 低估值"

    prompt = f"""你是一个投资助手意图分类器。分析用户输入，判断其意图。

意图类型及示例：
- screen: "帮我找市盈率低于15的股票"、"筛选格雷厄姆标的"、"推荐价值股"、"根据策略3筛选股票"、"按defance-strategy筛选"
- screen_save: "记住pe<10 pb<0.8为超低估策略"、"保存这个策略叫低估值"、"起个名字叫小盘价值"
- analyze: "分析一下600519"、"茅台的基本面怎么样"、"看看000001"
- help: "你能做什么"、"帮助"、"怎么用"
- quit: "退出"、"再见"
- unknown: 无法识别

可用策略名称（仅在 intent 为 screen/screen_save 时提取对应的 key，否则留空）：
{strategies_desc}

重要：如果用户说"策略N"（如"策略3"），请查找上面列表中编号为N的策略，提取其 key 填入 strategy_name。

用户输入: {text}

请严格按以下 JSON 格式输出（不要输出任何其他内容）：
{{"intent": "<intent>", "confidence": <0.0-1.0>, "symbol": "<提取到的6位数字股票代码，没有则为空>", "strategy_name": "<策略英文 key，如 graham 或 defance-strategy，没有则为空>"}}"""

    try:
        response = llm.invoke(prompt)
        import json
        content = response.content if hasattr(response, "content") else str(response)
        data = json.loads(content)
        intent_str = data.get("intent", "unknown")
        intent_map = {
            "screen": Intent.SCREEN,
            "screen_save": Intent.SCREEN_SAVE,
            "analyze": Intent.ANALYZE,
            "help": Intent.HELP,
            "quit": Intent.QUIT,
        }
        return Classification(
            intent=intent_map.get(intent_str, Intent.UNKNOWN),
            confidence=float(data.get("confidence", 0.5)),
            extracted_symbol=data.get("symbol", ""),
            strategy_name=data.get("strategy_name", ""),
            reason=f"LLM 分类: {intent_str}",
        )
    except Exception:
        return classify(text)  # 降级

    try:
        response = llm.invoke(prompt)
        import json
        content = response.content if hasattr(response, "content") else str(response)
        data = json.loads(content)
        intent_str = data.get("intent", "unknown")
        intent_map = {
            "screen": Intent.SCREEN,
            "screen_save": Intent.SCREEN_SAVE,
            "analyze": Intent.ANALYZE,
            "help": Intent.HELP,
            "quit": Intent.QUIT,
        }
        return Classification(
            intent=intent_map.get(intent_str, Intent.UNKNOWN),
            confidence=float(data.get("confidence", 0.5)),
            extracted_symbol=data.get("symbol", ""),
            strategy_name=data.get("strategy_name", ""),
            reason=f"LLM 分类: {intent_str}",
        )
    except Exception:
        return classify(text)  # 降级
