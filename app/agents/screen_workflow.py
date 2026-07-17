"""筛选工作流 —— 编排 data → factor → rule 全链路。

职责：
  - 从自然语言中提取结构化筛选条件（NL → dict）
  - 加载策略配置（YAML / 策略记忆层）
  - 合并用户即时条件 + 预设策略
  - 从 basic_info 加载全市场数据
  - 计算因子
  - 按规则打分排序
  - 持久化用户自定义策略
  - 用 rich 表格呈现结果

这是 MainAgent 的"推荐股票"分支。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.agents.sub_agents.data_agent import load_basic_info, check_freshness, FreshnessStatus, load_latest_financial
from app.agents.sub_agents.factor_agent import FactorResult, compute_factors, compute_factors_with_financials
from app.agents.sub_agents.rule_agent import RuleResult, apply_rules

logger = logging.getLogger(__name__)


def _load_strategy_config(
    config_path: str | Path | None = None,
    strategy_name: str = "",
) -> dict[str, Any]:
    """加载策略 YAML 配置。

    查找优先级：
      1. config_path（显式路径，最高优先级）
      2. strategy_name → 通过策略记忆层按名查找
      3. 默认：configs/strategies/graham.yaml
    """
    import importlib
    yaml = importlib.import_module("yaml")

    # ---- 1. 显式路径 ----
    if config_path:
        path = Path(config_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

    # ---- 2. 按策略名称从记忆层加载 ----
    if strategy_name:
        from app.memory.strategy_memory import load_strategy
        cfg = load_strategy(strategy_name)
        if cfg:
            logger.info("通过策略记忆层加载 '%s'", strategy_name)
            return cfg
        # 策略名指定了但没找到/空文件 → 返回空，不 fallback
        logger.warning("策略 '%s' 未找到或为空，不使用默认策略", strategy_name)
        return {}

    # ---- 3. 默认 fallback（仅在未指定 strategy_name 时）----
    default_path = Path(__file__).parent.parent.parent / "configs" / "strategies" / "graham.yaml"
    if default_path.exists():
        with default_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    return {}


def parse_screen_conditions(text: str) -> dict[str, Any]:
    """用 LLM 从用户自然语言中提取结构化筛选条件 + 保存意愿。

    示例：
        "找 pe<10 pb<0.8 的股票"
        → {"filters": {"pe_ttm": {"max": 10}, "pb": {"max": 0.8}}, "save_name": ""}

        "记住 pe<10 pb<0.8 roe>15% 为超低估策略"
        → {"filters": {"pe_ttm": {"max": 10}, "pb": {"max": 0.8}, "roe": {"min": 15}},
            "save_name": "超低估"}

    Args:
        text: 用户原始输入。

    Returns:
        {"filters": {field: {min/max}}, "save_name": str}
    """
    try:
        from app.services.llm import get_chat_model
        llm = get_chat_model(temperature=0.0)
    except Exception:
        logger.warning("LLM 不可用，无法提取筛选条件，使用纯规则降级")
        return _parse_conditions_fallback(text)

    prompt = f"""你是一个筛选条件提取器。从用户输入中提取股票筛选条件。

可用的筛选字段（每个字段可选 min 和 max 约束，单位见注释）：
- pe_ttm: 市盈率 TTM（倍）
- pb: 市净率（倍）
- roe: 净资产收益率（%，如 15 表示 15%）
- dividend_yield: 股息率（%，如 3 表示 3%）
- market_cap: 总市值（自动换算：1亿=1e8，如 "50亿" → 5000000000）
- eps_basic: 每股基本收益
- debt_to_equity: 负债权益比
- current_ratio: 流动比率

规则：
- 只输出用户明确提到的字段和条件
- 如果用户没提某个字段，不要凭空添加
- 百分号去掉，只保留数值（roe>15% → {{"roe": {{"min": 15}}}}）
- 市值单位 "亿" 需乘以 1e8（"50亿" → {{"market_cap": {{"min": 5000000000}}}}）
- 如果用户想要保存策略并起了名字，填入 save_name（纯中文名称）

用户输入: {text}

请严格按以下 JSON 格式输出（不要输出任何其他内容）：
{{"filters": {{"<字段名>": {{"min": <数值, 没有则省略此key>, "max": <数值, 没有则省略此key>}}}}, "save_name": "<用户想保存的策略名称，没有则为空字符串>"}}"""

    try:
        import json
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        # 清理可能的 markdown 包裹
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(content)
        logger.info("NL 条件提取成功: filters=%s, save_name='%s'",
                    result.get("filters", {}), result.get("save_name", ""))
        return result
    except Exception as e:
        logger.warning("LLM 条件提取失败，降级为规则提取: %s", e)
        return _parse_conditions_fallback(text)


def _parse_conditions_fallback(text: str) -> dict[str, Any]:
    """纯规则降级：用正则从文本中直接提取筛选条件。

    在最简场景下兜底，例如 "pe<10 pb<0.8"。
    """
    import re

    filters: dict[str, Any] = {}
    save_name = ""

    # 字段别名映射
    _alias_map = {
        "pe_ttm": ["pe_ttm", "pe", "市盈率"],
        "pb": ["pb", "市净率"],
        "roe": ["roe", "净资产收益率"],
        "dividend_yield": ["dividend_yield", "股息率", "分红率"],
        "market_cap": ["market_cap", "市值"],
        "eps_basic": ["eps_basic", "每股收益", "eps"],
        "debt_to_equity": ["debt_to_equity", "负债权益比", "资产负债率"],
        "current_ratio": ["current_ratio", "流动比率"],
    }

    # 匹配: 字段名 + 比较符号 + 数值（可选 % 或 亿）
    _cond_pattern = re.compile(
        r"(市盈率|pe_ttm|pe|市净率|pb|净资产收益率|roe|股息率|分红率|dividend_yield"
        r"|市值|market_cap|每股收益|eps_basic|eps|负债权益比|资产负债率|debt_to_equity"
        r"|流动比率|current_ratio)"
        r"\s*"
        r"(<|>|<=|>=|=|小于|大于|低于|高于|不超过|不小于)"
        r"\s*"
        r"(\d+\.?\d*)\s*(%|亿)?",
        re.IGNORECASE,
    )

    for m in _cond_pattern.finditer(text):
        raw_field = m.group(1).lower()
        op = m.group(2)
        val = float(m.group(3))
        unit = m.group(4) or ""

        # 映射到规范字段名
        field = ""
        for canonical, aliases in _alias_map.items():
            if raw_field in aliases:
                field = canonical
                break
        if not field:
            continue

        # 单位换算
        if unit == "%":
            pass  # 已经是百分比数值
        elif unit == "亿" and field == "market_cap":
            val *= 1e8

        entry = filters.setdefault(field, {})

        if op in (">", ">=", "大于", "高于", "不小于"):
            entry["min"] = val
        elif op in ("<", "<=", "小于", "低于", "不超过"):
            entry["max"] = val
        elif op == "=":
            entry["min"] = val
            entry["max"] = val

    # 提取保存名称：为"xxx"策略 / 记为"xxx" / 叫"xxx"
    _save_pattern = re.compile(
        r"(?:为|记为|称为|叫)\s*"
        r"[「『\"\']?"
        r"([\u4e00-\u9fff\w]{1,10}?)(?:的)?"
        r"[」』\"\']?"
        r"\s*(?:策略|条件|规则|[,，。.!！]|$)",
    )
    sm = _save_pattern.search(text)
    if sm:
        save_name = sm.group(1).strip()

    return {"filters": filters, "save_name": save_name}


def _parse_strategy_description_fallback(description: str) -> dict[str, Any]:
    """LLM 不可用时的确定性降级：用别名表 + 正则从策略描述中提取结构化条件。

    逐条拆分描述中的规则，用别名表匹配字段，提取数值和比较符。
    别名表覆盖不到的规则放入 unsupported。
    """
    import re

    from app.agents.sub_agents.rule_agent import FIELD_ALIASES_ZH

    # ---- 字段别名映射（同 rule_agent）----
    _alias_map: dict[str, set[str]] = {}
    for alias, field in FIELD_ALIASES_ZH.items():
        _alias_map.setdefault(field, set()).add(alias)

    # 需要排除的通用词（避免误匹配）—— 仅排除太短的通用词
    _exclude = {
        "eps", "dividend",
        "负债", "长期债务", "每股收益",
    }

    filters: dict[str, Any] = {}
    unsupported: list[str] = []

    # 去掉"输出要求"后方内容，按编号拆分
    desc_clean = re.split(r'\n\s*输出要求', description)[0].strip()
    rules = re.split(r'\n\s*\d+[\.、．)\s]+', desc_clean)
    rules = [r.strip() for r in rules if len(r.strip()) > 10]

    # 计算字段优先（如 pe_pb 比 pe_ttm 先匹配），行内按分号拆分
    _computed_fields = {"pe_pb"}
    _field_order = list(_computed_fields) + [f for f in _alias_map if f not in _computed_fields]

    for rule_raw in rules:
        # 一行可能有多个条件（用分号/句号分隔）
        sub_rules = re.split(r'[；;。]', rule_raw) if len(rule_raw) > 30 else [rule_raw]
        sub_rules = [s.strip() for s in sub_rules if len(s.strip()) > 10]

        for rule in sub_rules:
            matched = False

            for field in _field_order:
                aliases = _alias_map[field]
                # 排除通用短词防止误匹配
                effective_aliases = [a for a in aliases if a not in _exclude or len(a) >= 4]
                for alias in sorted(effective_aliases, key=len, reverse=True):
                    if alias not in rule and alias.lower() not in rule.lower():
                        continue

                    patterns = [
                        r'(不低于|不小于|至少|大于|高于|不应该?超过|不应?超过|不超过|不应高于|低于|小于|不大于)\s*(\d+\.?\d*)\s*(亿|万|%|倍)?',
                        r'的\s*(\d+\.?\d*)\s*(倍|亿|万)?',
                        r'(\d+\.?\d*)\s*(亿|万|倍|%)',
                    ]

                    for pat in patterns:
                        m = re.search(pat, rule)
                        if not m:
                            continue

                        groups = m.groups()
                        if len(groups) == 3:
                            op_word, val_str, unit = groups
                        elif len(groups) == 2:
                            op_word, val_str, unit = (None, groups[0], groups[1])
                        else:
                            continue

                        val = float(val_str)

                        has_usd = '美元' in rule
                        if has_usd:
                            if '亿' in rule:
                                val *= 1e8
                            val *= 7
                        elif unit == '亿':
                            val *= 1e8
                        elif unit == '万':
                            val *= 1e4

                        entry = filters.setdefault(field, {})

                        if op_word in ('不低于', '不小于', '至少', '大于', '高于'):
                            entry["min"] = val
                        elif op_word in ('不应该超过', '不应超过', '不超过', '不应高于', '小于', '低于', '不大于'):
                            entry["max"] = val
                        elif op_word is None:
                            if '不低于' in rule or '至少' in rule:
                                entry["min"] = val
                            elif '不超过' in rule or '低于' in rule or '小于' in rule:
                                entry["max"] = val
                            else:
                                entry["max"] = val  # "的15倍"→默认上限

                        matched = True
                        break

                    if matched:
                        break

                if matched:
                    break

            if not matched:
                unsupported.append(f"「{rule[:60]}…」")

    # pe_ttm > 0 的特殊处理
    if "pe_ttm" in filters and "min" not in filters["pe_ttm"]:
        filters["pe_ttm"]["min"] = 0
        # 如果有 max=15 但没 min，补 min=0
        if "max" in filters["pe_ttm"] and filters["pe_ttm"]["max"] > 0:
            filters["pe_ttm"]["min"] = 0

    return {"filters": filters, "unsupported": unsupported}


def translate_description(description: str, safety_filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """用 LLM 把自然语言策略描述翻译成结构化筛选条件。

    这是 description → hard_filters 的核心桥梁。YAML 的 screen.description
    是给人写的（语义丰富），但筛选引擎只能理解 hard_filters（结构化 min/max）。
    本函数让 LLM 阅读 NL 描述，编译出可执行的 hard_filters。

    Args:
        description:    策略 YAML 中的 screen.description 文本（自然语言）。
        safety_filters: 策略 YAML 中的 screen.hard_filters（可选安全边界）。
                        如果提供，LLM 生成的条件会被夹在安全边界之内。
                        如果为 None，不应用安全边界。

    Returns:
        {
            "filters":     {field: {min, max}, ...},    # 结构化筛选条件
            "unsupported": ["规则 X 当前系统不支持，因为...", ...],  # 无法翻译的规则
        }

    Example:
        >>> result = translate_description("pe_ttm 小于 15，pb 小于 1.5，roe 大于 10%")
        >>> result["filters"]
        {"pe_ttm": {"max": 15}, "pb": {"max": 1.5}, "roe": {"min": 10}}
    """
    safety_filters = safety_filters or {}

    try:
        from app.services.llm import get_chat_model
        llm = get_chat_model(temperature=0.0)
    except Exception:
        logger.warning("LLM 不可用，使用规则解析降级方案")
        result = _parse_strategy_description_fallback(description)
        # 后校验也适用于规则解析结果
        effective, hallucinated = _drop_hallucinated_fields(result.get("filters", {}), description)
        if hallucinated:
            result["unsupported"] = list(result.get("unsupported", [])) + hallucinated
        result["filters"] = _normalize_filters(effective)
        return result

    # ---- 构造安全边界描述 ----
    safety_desc = ""
    if safety_filters:
        lines = ["以下安全边界来自策略配置的 hard_filters，你生成的条件必须落在此范围内："]
        lines.append("（min 不能低于安全边界 min，max 不能高于安全边界 max）")
        for field, cond in safety_filters.items():
            parts = []
            if "min" in cond:
                parts.append(f"min={cond['min']}")
            if "max" in cond:
                parts.append(f"max={cond['max']}")
            lines.append(f"  - {field}: {', '.join(parts)}")
        safety_desc = "\n".join(lines)
    else:
        safety_desc = "（无安全边界约束）"

    # 动态生成字段列表（含中文名和计算字段）
    _base_fields = [
        ("pe_ttm",  "滚动市盈率（倍），中文别名：市盈率/PE"),
        ("pb",      "市净率（倍），中文别名：市净率/账面值/价格账面值比"),
        ("roe",     "净资产收益率（%），中文别名：净资产收益率/ROE"),
        ("dividend_yield", "股息率（%），中文别名：股息率/分红率"),
        ("market_cap",     "总市值（元），1亿=1e8，别名：市值/总市值。注意：销售额/营收不是市值"),
        ("eps_basic",      "基本每股收益（元），中文别名：每股收益/EPS"),
        ("debt_to_equity", "负债权益比（倍），中文别名：负债权益比/资产负债率/负债"),
        ("current_ratio",  "流动比率（倍），中文别名：流动比率/流动比/流动资产"),
        ("revenue",        "营业收入（元），来自利润表，别名：销售额/年销售额/营收。'年销售额不低于60亿美元'→先用汇率换算为元，再映射到此字段"),
        ("total_assets",   "总资产（元），来自资产负债表，别名：总资产/资产总额。'总资产不低于30亿人民币'→映射到此字段"),
    ]

    _computed_fields = [
        ("pe_pb", "PE×PB乘积 = pe_ttm × pb，中文别名：市盈率×市净率/市盈率与价格账面值之比的乘积/市盈率乘市净率"),
    ]

    base_lines = "\n".join(f"- {k}: {desc}" for k, desc in _base_fields)
    computed_lines = "\n".join(f"- {k}: {desc}" for k, desc in _computed_fields)

    prompt = f"""你是一个投资策略编译器。请逐条阅读下面的策略描述，将其中每条规则翻译成结构化筛选条件。

## 可用基础字段

{base_lines}

## 可用复合字段

{computed_lines}

## 核心规则（必须严格遵守）

### 1. 反幻觉
- 策略描述中**没有提到的字段，一律不要凭空添加**到 filters 中。
- 例如：描述只说"PE不超过15"，不要自己补 PB、ROE、股息率等任何条件。
- 不要套用任何"典型模板"或"常见策略组合"——只做逐字翻译。

### 2. 逐条对账
- 策略描述中的**每一行/每条规则都必须被处理**。
- 能被翻译的 → 放入 filters。
- 系统不支持的 → 放入 unsupported，同时写清编号和原文。
- 例如描述有 8 条规则，输出中 filters + unsupported 的条目总数应等于 8。
- 如果某条规则部分可翻译部分不支持，把它们拆开分别对待。

### 3. 复合条件用 pe_pb
- "市盈率乘市净率不超过 22.5" → {{"pe_pb": {{"max": 22.5}}}}
- "市盈率与价格账面值之比的乘积不应超过 22.5" → {{"pe_pb": {{"max": 22.5}}}}
- 不要把 pe_pb 拆成单独的 pb 约束。

### 4. 不支持的情况要明确说明
以下情况应放入 unsupported：
- 需要历史数据（如"过去10年EPS增长"、"连续20年分红"、"过去10年每年有利润"）
- 系统没有对应字段（如"总资产"——在资产负债表、不在basic_info；"年销售额"——在利润表、不是市值）
- 跨多字段的复杂比较（如"负债≤股权×2"，这需要同时查两个字段做除法判分，当前单字段 min/max 无法表达）
- 单个字段 min/max 无法表达的条件（如"流动资产至少是流动负债的两倍"= current_ratio≥2，这是可以的）

注意：部分条件虽然涉及多字段，但可以化简为单字段约束：
  - "流动资产至少是流动负债的两倍" → {{"current_ratio": {{"min": 2}}}}（流动比率本身就是流动资产/流动负债）

### 5. 条件输出格式
每个字段最多两个约束：min（下限）和 max（上限）。
- "pe_ttm 小于 15" → {{"pe_ttm": {{"max": 15}}}}
- "roe 大于 10%" → {{"roe": {{"min": 10}}}}
- "市值不低于 50 亿" → {{"market_cap": {{"min": 5000000000}}}}
- "市盈率乘市净率不超过 22.5" → {{"pe_pb": {{"max": 22.5}}}}

操作符映射：
- 小于/低于/不超过/不应高于 → max
- 大于/高于/不小于/至少 → min
- 等于/在 X 到 Y 之间 → min 和 max 都要
- "pe_ttm 大于 0" 通常表示排除亏损股 → {{"min": 0}}

单位换算：
- 市值/销售额带"亿" → 乘以 1e8（"50亿" → 5000000000）
- 带"%" → 去掉百分号只留数值（"15%" → 15）
- pe_ttm 的"倍"、pb 的"倍" → 直接写数值

## 安全边界

{safety_desc}

## 策略描述（逐条翻译）

{description}

## 输出格式

严格按以下 JSON 格式输出（不要 markdown 包裹、不要额外说明）：
{{"filters": {{"<字段名>": {{"min": <数值>, "max": <数值>}}}}, "unsupported": ["第X条「原文摘要」不支持：原因"]}}

要求：filters + unsupported 的条目总数应等于策略描述中的规则条数。每一条都要有下落。"""

    try:
        import json
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(content)

        llm_filters = result.get("filters", {})
        unsupported = result.get("unsupported", [])

        # ---- 应用安全边界 ----
        if safety_filters:
            effective = _apply_safety_bounds(llm_filters, safety_filters)
        else:
            effective = llm_filters

        effective = _normalize_filters(effective)

        # ---- 后校验：剔除廖述中未提及的字段（防御 LLM 幻觉）----
        effective, hallucinated = _drop_hallucinated_fields(effective, description)
        if hallucinated:
            unsupported = list(unsupported) + hallucinated
            # 确保控制台可见
            for h in hallucinated:
                print(f"  [!] 后校验剔除: {h}")

        logger.info("策略描述翻译完成: filters=%s, unsupported=%s", effective, unsupported)
        return {"filters": effective, "unsupported": unsupported}

    except Exception as e:
        logger.warning("LLM 翻译策略描述失败: %s", e)
        return {"filters": {}, "unsupported": [f"翻译失败: {e}"]}


def _to_num(v: Any) -> float | None:
    """将值转为 float，处理 LLM 可能输出的字符串数字。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except (ValueError, TypeError):
            logger.debug("无法将 %r 转为数字，跳过", v)
            return None
    return None


def _apply_safety_bounds(
    llm_filters: dict[str, Any],
    safety_filters: dict[str, Any],
) -> dict[str, Any]:
    """用安全边界约束 LLM 生成的条件。

    原则：
    - LLM 的 min 不能低于 safety 的 min（防止太松）
    - LLM 的 max 不能高于 safety 的 max（防止太松）
    - 如果 LLM 没生成某字段但 safety 有 → 保留 safety 底（fallback）
    """
    merged = dict(safety_filters)

    for field, llm_cond in llm_filters.items():
        safety_cond = safety_filters.get(field, {})
        merged.setdefault(field, {})

        llm_min = _to_num(llm_cond.get("min"))
        llm_max = _to_num(llm_cond.get("max"))
        safety_min = _to_num(safety_cond.get("min"))
        safety_max = _to_num(safety_cond.get("max"))

        # min: 取更严格（更大值）
        if llm_min is not None and safety_min is not None:
            merged[field]["min"] = max(llm_min, safety_min)
        elif llm_min is not None:
            merged[field]["min"] = llm_min
        elif safety_min is not None:
            merged[field]["min"] = safety_min

        # max: 取更严格（更小值）
        if llm_max is not None and safety_max is not None:
            merged[field]["max"] = min(llm_max, safety_max)
        elif llm_max is not None:
            merged[field]["max"] = llm_max
        elif safety_max is not None:
            merged[field]["max"] = safety_max

    return merged


def _normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """清洗 filters：去掉空 min/max 字段、去掉 LLM 可能乱加的负数等。"""
    cleaned: dict[str, Any] = {}
    for field, cond in filters.items():
        if not isinstance(cond, dict):
            continue
        entry: dict[str, Any] = {}
        v = _to_num(cond.get("min"))
        if v is not None:
            entry["min"] = v
        v = _to_num(cond.get("max"))
        if v is not None:
            entry["max"] = v
        if entry:
            cleaned[field] = entry
    return cleaned


def _drop_hallucinated_fields(
    filters: dict[str, Any],
    description: str,
) -> tuple[dict[str, Any], list[str]]:
    """后校验：剔除描述中未出现任何关键词的字段（防御 LLM 套用模板幻觉）。

    原理：如果原文完全没有提到"净资产收益率"/ROE，但 LLM 输出了 roe 约束，
    说明 LLM 在套用训练数据中的通用模板，应该把它揪出来移回 unsupported。
    """
    import re

    # 每个字段在原文中必须出现的关键词（至少命中一个才保留）
    _field_keywords: dict[str, list[str]] = {
        "pe_ttm":         [r"市盈率", r"\bPE\b", r"pe_ttm", r"股价.*倍", r"利润的?\d+倍", r"利润的.*倍"],
        "pb":             [r"市净率", r"账面值", r"价格账面", r"市账", r"\bPB\b", r"资产净值"],
        "roe":            [r"净资产收益", r"净资产回报", r"\bROE\b", r"资本回报率", r"股东权益回报"],
        "dividend_yield": [r"股息率", r"分红率", r"派息率", r"dividend.yield", r"现金分红"],
        "market_cap":     [r"市值", r"总市值", r"\bmarket.cap\b"],
        "revenue":        [r"营业收入", r"销售额", r"营收", r"年销售额", r"\brevenue\b", r"销售收入"],
        "total_assets":   [r"总资产", r"资产总额", r"total.assets", r"资产(?!负债)"],
        "eps_basic":      [r"每股收?益", r"\bEPS\b", r"每股利润", r"每股盈利"],
        "debt_to_equity": [r"负债.*权益", r"负债.*股权", r"资产负债", r"debt.*equity", r"长期债务"],
        "current_ratio":  [r"流动比", r"流动资产.*流动负债", r"流动负债.*流动资产", r"current.ratio"],
        "pe_pb":          [r"市盈率.*市净率|市盈率.*账面值|市净率.*市盈率|账面值.*市盈率",
                           r"市盈率.*乘|市盈率.*积|乘积|之积"],
    }

    desc_lower = description.lower()
    kept: dict[str, Any] = {}
    dropped: list[str] = []

    for field, cond in filters.items():
        keywords = _field_keywords.get(field)
        if not keywords:
            # 未知字段直接保留（不做误杀）
            kept[field] = cond
            continue

        found = any(re.search(kw, desc_lower, re.IGNORECASE) for kw in keywords)
        if found:
            kept[field] = cond
        else:
            display = _FIELD_LABEL.get(field, field)
            dropped.append(f"「{display}」原文未提及，疑似 LLM 模板幻觉，已自动剔除")

    if dropped:
        logger.warning("剔除 %d 个幻觉字段: %s", len(dropped), dropped)

    return kept, dropped


def _drop_no_data_fields(
    hard_filters: dict[str, Any],
    enriched_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """数据可用性检查：如果某字段在所有行中都是 None/0，则从筛选条件中移除。

    例如股息率在 A 股 basic_info 中普遍缺失，筛了也是 0 结果，应跳过。
    """
    from app.agents.sub_agents.rule_agent import _safe_float, COMPUTED_FIELDS

    cleaned = dict(hard_filters)
    reasons: list[str] = []

    for field in list(cleaned.keys()):
        has_data = False

        if field in COMPUTED_FIELDS:
            # 计算字段：检查其依赖的基础字段是否有数据
            meta = COMPUTED_FIELDS[field]
            dep_fields = meta.get("fields", [])
            # 所有依赖字段都有值才算有数据
            sample_count = sum(
                1 for r in enriched_rows
                if all(_safe_float(r.get(f)) is not None for f in dep_fields)
            )
            has_data = sample_count > 0
        else:
            sample_count = sum(
                1 for r in enriched_rows
                if _safe_float(r.get(field)) is not None
            )
            has_data = sample_count > 0

        if not has_data:
            display = _FIELD_LABEL.get(field, field)
            cleaned.pop(field)
            msg = f"「{display}」全市场无数据，已自动跳过该条件"
            logger.info("剔除无数据字段: %s", field)
            reasons.append(msg)

    return cleaned, reasons


def _merge_filters(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """合并筛选条件：override 完全覆盖 base 中的同名字段。

    Args:
        base:     基础条件（如 YAML hard_filters）。
        override: 用户即时条件（从 NL 提取）。

    Returns:
        合并后的条件字典。
    """
    merged = dict(base)
    for k, v in override.items():
        merged[k] = v
    return merged


def extracted_to_strategy_config(
    filters: dict[str, Any],
    save_name: str = "",
) -> dict[str, Any]:
    """把提取出的筛选条件包装为完整的策略配置（格式对齐 YAML）。

    Args:
        filters:   {pe_ttm: {max: 10}, pb: {max: 0.8}, ...}
        save_name: 策略展示名称。

    Returns:
        完整的策略配置字典。
    """
    description = save_name or "自定义即时筛选"
    return {
        "universe": {
            "market": "A股",
            "exclude_st": True,
            "min_listing_days": 365,
        },
        "screen": {
            "description": description,
            "hard_filters": filters or {},
        },
        "review": {"require_confirm": False},
    }


def _format_score_table(
    scored: list[Any],
    max_rows: int = 30,
    show_score: bool = True,
) -> str:
    """把 RuleVerdict 列表格式化为终端文本表格。"""
    if not scored:
        return "(无结果)"

    rows = scored[:max_rows]

    def fmt_float(val: Any, precision: int = 2) -> str:
        if val is None:
            return "-"
        fv = float(val)
        return f"{fv:.{precision}f}"

    if show_score:
        headers = ["排名", "代码", "名称", "总分", "PE_TTM", "PB", "ROE%", "股息率%", "市值(亿)"]
    else:
        headers = ["序号", "代码", "名称", "PE_TTM", "PB", "ROE%", "股息率%", "市值(亿)"]

    col_offset = 0 if show_score else 1  # 不显示总分列时后续列宽需调整

    lines: list[str] = []
    header_line = " │ ".join(
        f"{h:<8}" if i < 3 else f"{h:<10}" for i, h in enumerate(headers)
    )
    lines.append(header_line)
    lines.append("─┼─".join(
        "─" * 8 if i < 3 else "─" * 10 for i in range(len(headers))
    ))

    for i, v in enumerate(rows):
        factors = v.factors if hasattr(v, "factors") else {}
        pe = factors.get("pe_ttm")
        pb = factors.get("pb")
        roe = factors.get("roe")
        div_y = factors.get("dividend_yield")
        mcap = factors.get("market_cap")

        if show_score:
            cols = [
                str(i + 1),
                str(v.symbol),
                str(v.name or "")[:8],
                f"{v.total_score:.2f}",
                fmt_float(pe),
                fmt_float(pb),
                fmt_float(roe, 1),
                fmt_float(div_y, 2) if div_y is not None else "-",
                fmt_float(mcap / 1e8, 1) if mcap is not None else "-",
            ]
        else:
            cols = [
                str(i + 1),
                str(v.symbol),
                str(v.name or "")[:8],
                fmt_float(pe),
                fmt_float(pb),
                fmt_float(roe, 1),
                fmt_float(div_y, 2) if div_y is not None else "-",
                fmt_float(mcap / 1e8, 1) if mcap is not None else "-",
            ]

        line = " │ ".join(
            f"{c:<8}" if j < 3 else f"{c:<10}"
            for j, c in enumerate(cols)
        )
        lines.append(line)

    total = len(scored)
    if total > max_rows:
        lines.append(f"\n... 共 {total} 只，仅显示前 {max_rows} 只")
    else:
        lines.append(f"\n共 {total} 只")

    return "\n".join(lines)


def _build_none_notes(scored: list[Any], max_rows: int = 30) -> str:
    """收集通过筛选但存在 None 字段的股票，生成警告信息。"""
    rows_with_none: list[tuple[str, str, list[str]]] = []  # [(code, name, [field_labels])]
    for v in scored[:max_rows]:
        none_fields = []
        for rule in v.rules if hasattr(v, "rules") else []:
            if getattr(rule, "skipped", False):
                label = _FIELD_LABEL.get(rule.rule_name, rule.rule_name)
                none_fields.append(label)
        if none_fields:
            rows_with_none.append((str(v.symbol), str(v.name or "")[:8], none_fields))

    if not rows_with_none:
        return ""

    lines = ["\n⚠ 以下标的部分字段数据缺失，已跳过对应条件（其余条件通过即算通过）："]
    for code, name, fields in rows_with_none:
        lines.append(f"  {code} {name}: 缺 {', '.join(fields)}")
    return "\n".join(lines)


def run_screen(
    db_path: str | Path,
    strategy_config_path: str | Path | None = None,
    strategy_name: str = "",
    raw_input: str = "",
    save: bool = False,
    max_results: int = 500,
) -> dict[str, Any]:
    """执行完整筛选流程。

    Args:
        db_path:              SQLite 数据库路径。
        strategy_config_path: 策略 YAML 路径（优先级最高）。
        strategy_name:        策略名称（如 "graham"），通过策略记忆层按名加载。
        raw_input:            用户原始输入，用于 NL 条件提取（"" 则不提取）。
        save:                 筛选完后是否需要持久化策略。
        max_results:           最多显示/返回的候选数。

    Returns:
        dict:
            - factor_result: FactorResult
            - rule_result:   RuleResult
            - table:         str —— 格式化文本表格
            - top_symbols:   list[str]
            - save_name:     str —— 提取的策略名称（供主 agent 后续处理）
            - filters_used:  dict —— 实际使用的筛选条件
    """
    db = Path(db_path)

    # ---- 0. 检查数据新鲜度（筛选只用 basic_info）----
    fs = check_freshness(str(db))
    if fs.price_need:
        return {
            "factor_result": None, "rule_result": None, "table": "",
            "top_symbols": [], "save_name": "", "filters_used": {},
            "freshness": fs,
        }

    # ---- 1. 从自然语言提取条件 ----
    extracted = parse_screen_conditions(raw_input) if raw_input else {"filters": {}, "save_name": ""}
    extracted_filters = extracted.get("filters", {})
    extracted_save_name = extracted.get("save_name", "")

    # ---- 2. 加载策略配置 ----
    config = _load_strategy_config(strategy_config_path, strategy_name)

    # ---- 2.5. 自动翻译description → structured filters（description 为主，hard_filters 为安全边界）----
    screen_cfg = config.get("screen", {})
    strategy_desc = screen_cfg.get("description", "") if screen_cfg else ""
    safety_filters = screen_cfg.get("hard_filters", {}) if screen_cfg else {}
    unsupported: list[str] = []
    translated_summary: list[str] = []
    if strategy_desc:
        print(f"  [...] 正在翻译策略描述为筛选条件...")
        translated = translate_description(strategy_desc, safety_filters=safety_filters)
        translated_filters = translated.get("filters", {})
        unsupported = translated.get("unsupported", [])
        if translated_filters:
            config.setdefault("screen", {})["hard_filters"] = _merge_filters(
                safety_filters, translated_filters
            )
            # 逐条打印翻译出的条件
            print(f"  ┌─ 策略翻译结果 ─────────────────────────────")
            print(f"  │ 已翻译条件（{len(translated_filters)} 条）：")
            for i, (field, cond) in enumerate(translated_filters.items(), 1):
                label = _FIELD_LABEL.get(field, field)
                constraints: list[str] = []
                if "min" in cond:
                    constraints.append(str(cond["min"]))
                constraints.append("~")
                if "max" in cond:
                    constraints.append(str(cond["max"]))
                line = f"  │   {i}. {label}  {''.join(constraints)}"
                print(line)
                translated_summary.append(line.strip())
        if unsupported:
            print(f"  │")
            print(f"  │ 不支持条件（{len(unsupported)} 条）：")
            for i, u in enumerate(unsupported, 1):
                line = f"  │   {i}. {u}"
                print(line)
                translated_summary.append(line.strip())
            print(f"  └───────────────────────────────────────────")
        else:
            print(f"  └───────────────────────────────────────────")

    # ---- 3. 合并条件 ----
    if extracted_filters:
        if strategy_name or strategy_config_path:
            base_filters = config.get("screen", {}).get("hard_filters", {})
            merged_filters = _merge_filters(base_filters, extracted_filters)
            config.setdefault("screen", {})["hard_filters"] = merged_filters
            display_name = config.get("screen", {}).get("description", "自定义策略")[:40]
        else:
            config = extracted_to_strategy_config(extracted_filters, extracted_save_name)
            display_name = extracted_save_name or "自定义条件"
    else:
        # 有策略名时优先用策略名，否则用描述前40字
        display_name = strategy_name or config.get("screen", {}).get("description", "默认策略")[:40]

    # ---- 4. 确定最终保存名称 ----
    final_save_name = ""
    if save:
        final_save_name = extracted_save_name  # 可能为空，由主 agent 补问

    print(f"\n  [*] 正在按「{display_name}」筛选全市场股票...")
    if extracted_filters:
        print(f"  [+] 应用即时条件: {_format_filters(extracted_filters)}")

    # ---- 5. 加载数据（三表合并：basic_info + income_statement + balance_sheet）----
    print("  ... 加载数据...")
    basic_rows = load_basic_info(str(db))
    if not basic_rows:
        print("  [X] basic_info 表为空或数据库不存在")
        return {
            "factor_result": None, "rule_result": None, "table": "(无数据)",
            "top_symbols": [], "save_name": final_save_name, "filters_used": {},
        }
    print(f"  [OK] basic_info: {len(basic_rows)} 只")

    income_rows = load_latest_financial(str(db), "income_statement")
    print(f"  [OK] income_statement: {len(income_rows)} 只")

    balance_rows = load_latest_financial(str(db), "balance_sheet")
    print(f"  [OK] balance_sheet: {len(balance_rows)} 只")

    # ---- 6. 计算因子（合并三表）----
    factor_result = compute_factors_with_financials(basic_rows, balance_rows, income_rows)
    print(f"  [OK] 计算 {factor_result.output_count} 只股票的因子")

    # ---- 6.5. 数据可用性检查：剔除全为 None 的筛选字段 ----
    hard_filters = config.get("screen", {}).get("hard_filters", {})
    if hard_filters and factor_result.enriched_rows:
        cleaned_filters, reasons_data = _drop_no_data_fields(
            dict(hard_filters), factor_result.enriched_rows
        )
        if reasons_data:
            config.setdefault("screen", {})["hard_filters"] = cleaned_filters
            unsupported = list(unsupported) + reasons_data
            for reason in reasons_data:
                print(f"  [!] {reason}")
                translated_summary.append(reason)

    # ---- 7. 规则筛选（仅通过/不通过，不打分不排序）----
    rule_result = apply_rules(factor_result.enriched_rows, config, filter_only=True)
    print(f"  [OK] 筛选完成：{rule_result.passed_candidates} 只符合条件，共 {len(rule_result.scored)} 只")

    # ---- 8. 生成表格 ----
    table = _format_score_table(rule_result.scored, max_results, show_score=False)

    # ---- 8.5. 标注数据缺失字段 ----
    none_notes = _build_none_notes(rule_result.scored, max_results)
    if none_notes:
        table = table + "\n\n" + none_notes

    # ---- 9. 持久化策略 ----
    if save and final_save_name and extracted_filters:
        strategy_config = extracted_to_strategy_config(extracted_filters, final_save_name)
        from app.memory.strategy_memory import save_strategy as persist_strategy
        result = persist_strategy(final_save_name, strategy_config, final_save_name)
        if result.get("success"):
            print(f"  [OK] 策略「{final_save_name}」已保存，下次输入「用{final_save_name}策略筛选」即可直接使用。")

    return {
        "factor_result": factor_result,
        "rule_result": rule_result,
        "table": table,
        "top_symbols": [v.symbol for v in rule_result.scored[:max_results]],
        "save_name": final_save_name,
        "filters_used": extracted_filters,
        "unsupported": unsupported,
        "translated_summary": translated_summary,
    }


_FIELD_LABEL = {
    "pe_ttm": "PE(ttm)", "pb": "PB", "roe": "ROE(%)",
    "dividend_yield": "股息率(%)", "market_cap": "市值(元)",
    "eps_basic": "EPS", "debt_to_equity": "负债权益比", "current_ratio": "流动比率",
    "pe_pb": "PE×PB",
    "revenue": "营收(元)", "total_assets": "总资产(元)",
}


def _format_filters(filters: dict[str, Any]) -> str:
    """把筛选条件可视化为简短字符串。"""
    _label = _FIELD_LABEL
    parts: list[str] = []
    for k, v in filters.items():
        label = _label.get(k, k)
        cond_parts: list[str] = []
        if "min" in v:
            cond_parts.append(f">={v['min']}")
        if "max" in v:
            cond_parts.append(f"<={v['max']}")
        parts.append(f"{label}{','.join(cond_parts)}")
    return ", ".join(parts)


# ==============================================================================
#  ScreenAgent —— 具有任务上下文的子 Agent
# ==============================================================================

class ScreenAgent:
    """筛选子 Agent —— 自主完成筛选任务的完整生命周期。

    职责：
      - 持有自己的 LLM（用于 NL → 结构化条件、策略翻译）
      - 缓存任务上下文（中间计算结果），同一任务链复用
      - 任务结束后释放任务上下文

    不是对话 Agent——不持有对话历史，不跟用户直接交互。
    只接收结构化任务参数，返回结果字典。

    Attributes:
        db_path:  SQLite 数据库路径。
        _llm:     领域 LLM（用于 parse_screen_conditions / translate_description）。
        _task_state: 任务上下文缓存（dict，key 为 result_id）。
    """

    def __init__(self, db_path: str | Path) -> None:
        from app.services.llm import get_chat_model

        self.db_path = str(db_path)
        self._llm = get_chat_model(temperature=0.0)
        self._task_state: dict[str, Any] = {}   # result_id → 缓存数据
        logger.info("ScreenAgent 初始化完成（db=%s）", self.db_path)

    # ------------------------------------------------------------------
    #  公开接口
    # ------------------------------------------------------------------

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """执行筛选任务。

        Args:
            task: {
                "action":          "screen" | "filter_on_cached",
                "conditions":      str —— 自然语言条件，
                "strategy_name":   str —— 策略名称（可选），
                "base_result_id":  str —— 上一轮结果 ID（筛选子集场景），
                "save":            bool —— 是否持久化策略，
                "max_results":     int —— 最多返回数（默认 200），
            }

        Returns:
            dict: {
                "result_id":      str —— 本次结果 ID，供后续追筛引用，
                "factor_result":  FactorResult | None,
                "rule_result":    RuleResult | None,
                "table":          str —— 格式化表格文本，
                "top_symbols":    list[str],
                "save_name":      str,
                "filters_used":   dict,
                "freshness":      FreshnessStatus | None,
            }
        """
        action = task.get("action", "screen")
        base_result_id = task.get("base_result_id", "")

        # ── 如果有上一轮结果且是追筛，从缓存取 ──
        if action == "filter_on_cached" and base_result_id:
            cached = self._task_state.get(base_result_id)
            if cached and cached.get("scored"):
                return self._filter_on_cached(
                    cached,
                    task.get("conditions", ""),
                    task.get("max_results", 30),
                )

        # ── 全新筛选 ──
        result = run_screen(
            self.db_path,
            strategy_name=task.get("strategy_name", ""),
            raw_input=task.get("conditions", ""),
            save=task.get("save", False),
            max_results=task.get("max_results", 500),
        )

        # 缓存结果
        if result.get("rule_result") and result["rule_result"].scored:
            import uuid
            result_id = str(uuid.uuid4())[:8]
            result["result_id"] = result_id
            self._task_state[result_id] = {
                "scored": result["rule_result"].scored,
                "filters_used": result.get("filters_used", {}),
            }
            # 清理旧缓存（保留最近 5 个）
            if len(self._task_state) > 5:
                oldest = sorted(self._task_state.keys())[:-5]
                for k in oldest:
                    del self._task_state[k]
        else:
            result["result_id"] = ""

        return result

    def clear_task_state(self) -> None:
        """清空任务上下文缓存（切换话题时由 MainAgent 调用）。"""
        count = len(self._task_state)
        self._task_state = {}
        if count:
            logger.info("ScreenAgent 任务缓存已清空（%d 条）", count)

    # ------------------------------------------------------------------
    #  内部
    # ------------------------------------------------------------------

    def _filter_on_cached(
        self,
        cached: dict[str, Any],
        new_conditions: str,
        max_results: int = 500,
    ) -> dict[str, Any]:
        """基于缓存结果做二级筛选。"""
        from app.agents.sub_agents.rule_agent import apply_rules, RuleResult

        scored = cached["scored"]
        existing_filters = cached.get("filters_used", {})

        # 从新条件中提取结构化 filters
        extracted = parse_screen_conditions(new_conditions)
        new_filters = extracted.get("filters", {})

        # 合并
        merged_filters = _merge_filters(existing_filters, new_filters)

        # 用新的 merged_filters 在缓存结果上直接做筛选
        verdicts = []
        for v in scored:
            # 检查是否满足合并后的条件
            factors = v.factors if hasattr(v, "factors") else {}
            passed = True
            for field, cond in merged_filters.items():
                val = factors.get(field)
                if "min" in cond and (val is None or val < cond["min"]):
                    passed = False
                    break
                if "max" in cond and (val is not None and val > cond["max"]):
                    passed = False
                    break
            if passed:
                verdicts.append(v)

        rule_result = RuleResult(
            scored=verdicts,
            total_candidates=len(scored),
            passed_candidates=len(verdicts),
        )

        table = _format_score_table(verdicts, max_results, show_score=False)

        result_id = ""
        if verdicts:
            import uuid
            result_id = str(uuid.uuid4())[:8]
            self._task_state[result_id] = {
                "scored": verdicts,
                "filters_used": merged_filters,
            }

        return {
            "result_id": result_id,
            "factor_result": None,
            "rule_result": rule_result,
            "table": table,
            "top_symbols": [v.symbol for v in verdicts[:max_results]],
            "save_name": "",
            "filters_used": merged_filters,
            "freshness": None,
        }
