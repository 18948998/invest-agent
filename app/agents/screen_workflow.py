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

from app.agents.sub_agents.data_agent import load_basic_info, check_freshness, FreshnessStatus, load_latest_financial, load_dividend_history, load_income_statement_history
from app.agents.sub_agents.factor_agent import FactorResult, compute_factors, compute_factors_with_financials, enrich_with_historical_factors
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

    # 从财务数据字典动态生成可用字段列表
    from app.knowledge.dict import get_dict
    fd = get_dict()
    table_fields_prompt = fd.prompt_tables()
    computed_fields_prompt = fd.prompt_computed()

    prompt = f"""你是一个筛选条件提取器。从用户输入中提取股票筛选条件。

## 可用数据表与字段

{table_fields_prompt}

## 系统计算字段

{computed_fields_prompt}

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

    # 字段别名映射（从财务数据字典动态加载）
    from app.knowledge.dict import get_dict
    _alias_map = get_dict().field_aliases()

    # 匹配: 字段名 + 比较符号 + 数值（可选 % 或 亿）
    _cond_pattern = re.compile(
        r"(市盈率|pe_ttm|pe|市净率|pb|净资产收益率|roe|股息率|分红率|dividend_yield"
        r"|每股派息|每股股利|分红|派息|pretax_bonus_per_share"
        r"|连续分红|持续分红|dividend_years_count"
        r"|市值|market_cap|每股收益|eps_basic|eps|负债权益比|资产负债率|debt_to_equity"
        r"|流动比率|current_ratio"
        r"|长期债务|长期借款|long_term_borrowings"
        r"|流动资产净额|净流动资产|net_current_assets"
        r"|long_term_debt_to_net_ca_ratio)"
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

    逐条拆分描述中的规则，先匹配特殊规则（EPS增长、分红年数、二分一、每年利润等），
    再用别名表覆盖剩余规则。
    """
    import re

    # 字段别名映射（从财务数据字典动态加载）
    from app.knowledge.dict import get_dict
    _alias_map: dict[str, set[str]] = {
        field: set(aliases) for field, aliases in get_dict().field_aliases().items()
    }

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

    # ── 特殊规则（先检查）：中文分数、"每年都有利润"、"至少N年分红"等 ──
    _CHINESE_FRACTIONS = {"三分之一": 1/3, "三分之二": 2/3, "一半": 0.5, "四分之一": 0.25,
                          "一倍": 1, "两倍": 2, "三倍": 3, "五倍": 5, "十倍": 10}

    # 计算字段优先
    _computed_fields = {"pe_pb", "eps_growth_10yr_3yr_avg", "consecutive_profitable_years"}
    _field_order = list(_computed_fields) + [f for f in _alias_map if f not in _computed_fields]

    for rule_raw in rules:
        sub_rules = re.split(r'[；;。]', rule_raw) if len(rule_raw) > 40 else [rule_raw]
        sub_rules = [s.strip() for s in sub_rules if len(s.strip()) > 10]

        for rule in sub_rules:
            matched = False

            # ── 特殊匹配：EPS 增长率 ──
            if re.search(r'(每股收益|eps|每股利润).*(增长|上涨|增加|至少要|不低于)', rule, re.I):
                for word, num in _CHINESE_FRACTIONS.items():
                    if word in rule:
                        filters.setdefault("eps_growth_10yr_3yr_avg", {})["min"] = round(num, 4)
                        matched = True
                        break
                if matched:
                    continue
                # 尝试匹配数字
                m = re.search(r'(增长|上涨).*?(\d+\.?\d*)\s*(%|倍|分之一)?', rule)
                if m:
                    val = float(m.group(2))
                    if m.group(3) in ('分之一',): val = 1/val
                    filters.setdefault("eps_growth_10yr_3yr_avg", {})["min"] = val
                    matched = True
                    continue

            # ── 特殊匹配：连续分红年数 ──
            m = re.search(r'(至少有|至少|不低于|不小于?)\s*(\d+)\s*年\s*(连续|持续).*(分红|支付股息|股息|派息|股利)', rule)
            if m:
                filters.setdefault("dividend_years_count", {})["min"] = float(m.group(2))
                matched = True
                continue

            # ── 特殊匹配：每年都有利润 → 连续盈利年数 ──
            if re.search(r'(每年|每一年).*(利润|盈利|有利润|有盈利)', rule):
                m = re.search(r'(过去\s*)?(\d+)\s*年', rule)
                yr = int(m.group(2)) if m else 10
                filters.setdefault("consecutive_profitable_years", {})["min"] = float(yr)
                matched = True
                continue

            # ── 特殊匹配：过去3年平均利润的市盈率 / 股价不应高于过去3年平均利润的X倍 ──
            if re.search(r'(过去\s*)?3\s*年\s*平均\s*(利润|收益|EPS|eps)', rule) or re.search(r'(三年平均|近3年).*(利润|收益)', rule):
                m = re.search(r'(\d+\.?\d*)\s*(倍|_)', rule)
                if m:
                    val = float(m.group(1))
                    filters.setdefault("pe_3yr_avg", {})["max"] = val
                    matched = True
                    continue
                # 尝试匹配"15倍"这样的数字
                for word, num in _CHINESE_FRACTIONS.items():
                    if word in rule:
                        filters.setdefault("pe_3yr_avg", {})["max"] = float(num * 15) if num == 3 else float(num * 15)
                        matched = True
                        break
                if matched:
                    continue
                # 兜底：尝试任何数字
                m = re.search(r'(\d+)\s*倍', rule)
                if m:
                    filters.setdefault("pe_3yr_avg", {})["max"] = float(m.group(1))
                    matched = True
                    continue

            # ── 特殊匹配：流动资产 ≥ 2× 流动负债 ──
            if re.search(r'流动资产.*流动负债.*(两倍|2倍|二倍)', rule):
                filters.setdefault("current_ratio", {})["min"] = 2.0
                matched = True
                continue

            # ── 特殊匹配：负债 ≤ 2× 股权 ──
            if re.search(r'负债.*股权.*(两倍|2倍|二倍)', rule):
                filters.setdefault("debt_to_equity", {})["max"] = 2.0
                matched = True
                continue

            # ── 特殊匹配：长期债务不应超过流动资产净额 → long_term_debt_to_net_ca_ratio ≤ 1 ──
            if re.search(r'长期债务.*不超过.*流动资产净额', rule):
                filters.setdefault("long_term_debt_to_net_ca_ratio", {})["max"] = 1.0
                matched = True
                continue

            # ── 特殊匹配：行业差异化条件（工业企业…公用事业企业…），取最严格（全部 AND）──
            if re.search(r'(工业企业|公用事业)', rule):
                # 销售额
                for unit, mul in [("亿美元", 7*1e8), ("亿", 1e8), ("万", 1e4)]:
                    m = re.search(r'销售额\s*(不低于|不小于|至少|大于|高于|超过)?\s*(\d+\.?\d*)\s*' + unit, rule)
                    if m:
                        val = float(m.group(2)) * mul
                        filters.setdefault("revenue", {})["min"] = val
                        matched = True
                        break
                # 总资产
                for unit, mul in [("亿美元", 7*1e8), ("亿", 1e8), ("万", 1e4)]:
                    m = re.search(r'总资产\s*(不低于|不小于|至少|大于|高于|超过)?\s*(\d+\.?\d*)\s*' + unit, rule)
                    if m:
                        val = float(m.group(2)) * mul
                        filters.setdefault("total_assets", {})["min"] = val
                        matched = True
                        break
                # 流动比率
                for word, num in [("两倍", 2), ("2倍", 2), ("二倍", 2)]:
                    if word in rule and re.search(r'流动资产.*流动负债', rule):
                        filters.setdefault("current_ratio", {})["min"] = float(num)
                        matched = True
                        break
                # 负债 ≤ N× 股权
                m = re.search(r'负债.*(?:不应|不该|不).*(?:超过|高于)\s*(?:股权|权益|净资产)\s*(两倍|2倍|二倍)', rule)
                if m:
                    filters.setdefault("debt_to_equity", {})["max"] = 2.0
                    matched = True
                if matched:
                    continue

            # ── 特殊匹配：非ST、退市股 → 已通过 universe.exclude_st 支持 ──
            if re.search(r'(非\s*ST|退市|ST\s*股)', rule):
                matched = True
                continue

            # ── 特殊匹配：年销售额不低于 X 亿 ──
            for unit, mul in [("亿美元", 7*1e8), ("亿", 1e8), ("万", 1e4)]:
                m = re.search(r'销售额\s*(不低于|不小于|至少|大于|高于|超过)?\s*(\d+\.?\d*)\s*' + unit, rule)
                if m:
                    val = float(m.group(2)) * mul
                    filters.setdefault("revenue", {})["min"] = val
                    matched = True
                    break
            if matched:
                continue

            # ── 特殊匹配：总资产不低于 X 亿 ──
            for unit, mul in [("亿美元", 7*1e8), ("亿", 1e8), ("万", 1e4)]:
                m = re.search(r'总资产\s*(不低于|不小于|至少|大于|高于|超过)?\s*(\d+\.?\d*)\s*' + unit, rule)
                if m:
                    val = float(m.group(2)) * mul
                    filters.setdefault("total_assets", {})["min"] = val
                    matched = True
                    break
            if matched:
                continue

            # ── 通用别名匹配 ──
            for field in _field_order:
                aliases = _alias_map[field]
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
                            if '亿' in rule: val *= 1e8
                            val *= 7
                        elif unit == '亿': val *= 1e8
                        elif unit == '万': val *= 1e4

                        entry = filters.setdefault(field, {})
                        if op_word in ('不低于', '不小于', '至少', '大于', '高于'):
                            entry["min"] = val
                        elif op_word in ('不应该超过', '不应超过', '不超过', '不应高于', '小于', '低于', '不大于'):
                            entry["max"] = val
                        elif op_word is None:
                            if '不低于' in rule or '至少' in rule: entry["min"] = val
                            elif '不超过' in rule or '低于' in rule or '小于' in rule: entry["max"] = val
                            else: entry["max"] = val

                        matched = True
                        break
                    if matched: break
                if matched: break

            if not matched:
                # 跳过策略标题、非条件描述等
                if not re.search(r'(不大于|不小于|至少|不低于|不应|不超过|大于|低于|小于|高于|连续|每年|至少|不超)', rule):
                    continue
                unsupported.append(f"「{rule[:60]}…」")

    # pe_ttm > 0 的特殊处理
    if "pe_ttm" in filters and "min" not in filters["pe_ttm"]:
        filters["pe_ttm"]["min"] = 0
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

    # 从财务数据字典动态生成字段列表（外部记忆注入 LLM 上下文）
    from app.knowledge.dict import get_dict
    fd = get_dict()
    table_fields_prompt = fd.prompt_tables()
    computed_fields_prompt = fd.prompt_computed()

    prompt = f"""你是一个投资策略编译器。请逐条阅读下面的策略描述，将其中每条规则翻译成结构化筛选条件。

## 可用数据表与字段（财务数据字典）

{table_fields_prompt}

## 系统计算字段（自动计算，可直接引用）

{computed_fields_prompt}

## 核心规则（必须严格遵守）

### 0. 数据来源说明
- 所有财报字段（revenue、total_assets、debt_to_equity、current_ratio、eps_basic、roe 等）系统**仅提供最新年报数据**。
- 分红字段 pretax_bonus_per_share 来自分红送转历史表的最新记录（每股税前股利）。
- 分红字段 dividend_years_count 是系统自动计算的**连续分红年数**（从最新年份往前回溯，无间断的年份数）。如"至少20年连续支付股息"→ {{"dividend_years_count": {{"min": 20}}}}。

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
- 系统没有对应字段且无法用现有计算字段表达的条件
- 例如"管理层持股比例"、"员工人数"等非财务条件

注意以下条件系统已支持：
  - "长期债务不应超过流动资产净额" → {{"long_term_debt_to_net_ca_ratio": {{"max": 1}}}}（系统自动计算比率）
  - "流动资产至少是流动负债的两倍" → {{"current_ratio": {{"min": 2}}}}（current_ratio已支持）

### 5. 行业差异化规则处理
**核心原则：系统没有行业分类字段，遇到"工业企业…公用事业企业…"等分行业的条件，取最严格（所有分支条件都加 AND）。**
- 例如"工业企业年销售额不低于60亿；公用事业企业总资产不低于30亿" → 同时加入 revenue min=60亿 和 total_assets min=30亿
- 例如"工业企业流动比率≥2；公用事业企业负债≤2×股权" → 同时加入 current_ratio min=2 和 debt_to_equity max=2
- 例如"非ST、退市股" → 系统已通过 universe.exclude_st 自动排除ST/PT/退市股，无需输出到 filters 或 unsupported

注意：以下历史数据相关条件系统已支持：
  - "至少N年连续分红"/"连续N年支付股息"/"至少有N年连续分红记录" → {{"dividend_years_count": {{"min": N}}}}
  - "过去10年EPS增长至少达到三分之一" / "过去10年内每股收益的增长至少要达到三分之一(期初和期末使用三年平均数)" → {{"eps_growth_10yr_3yr_avg": {{"min": 0.333}}}}
  - "过去10年中每年都有利润" / "过去10年中普通股每年都有一定的利润" → {{"consecutive_profitable_years": {{"min": 10}}}}
  - "股价不应高于过去3年平均利润的15倍" / "过去3年平均利润的X倍" → {{"pe_3yr_avg": {{"max": X}}}}
  - dividend_years_count 是系统自动从分红送转历史表计算的连续分红年数（从最新年份回溯无间断）
  - eps_growth_10yr_3yr_avg 是系统自动从利润表10年历史计算的EPS增长率（期初3年均值vs期末3年均值）
  - consecutive_profitable_years 是系统自动从利润表历史数据计算的连续盈利年数
  - pe_3yr_avg 是系统自动计算的PE(3年均)，用最近3年EPS均值
  - "每股派息"/"每股股利"/"每股分红" → pretax_bonus_per_share（最新记录）
  - "长期债务不应超过流动资产净额" → {{"long_term_debt_to_net_ca_ratio": {{"max": 1}}}}（系统自动计算 long_term_borrowings ÷ (current_assets - current_liabilities)）

### 6. 条件输出格式
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
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

        def _call_llm():
            return llm.invoke(prompt)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_llm)
            try:
                response = future.result(timeout=120)  # 最多等 2 分钟
            except FutureTimeoutError:
                print(f"  ⚠ LLM 调用超时（>120s），自动降级为规则解析器...")
                logger.warning("LLM 翻译策略描述超时（>120s），降级为规则解析")
                raise TimeoutError("LLM call timed out after 120s")

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

    except (TimeoutError, Exception) as e:
        print(f"  ⚠ LLM 翻译超时/失败，降级为规则解析器，原因: {e}")
        logger.warning("LLM 翻译策略描述失败，降级为规则解析: %s", e)
        # 超时或其他异常时，使用规则解析器兜底
        result = _parse_strategy_description_fallback(description)
        effective, hallucinated = _drop_hallucinated_fields(result.get("filters", {}), description)
        if hallucinated:
            result["unsupported"] = list(result.get("unsupported", [])) + hallucinated
        result["filters"] = _normalize_filters(effective)
        print(f"  ┌─ 降级解析结果（规则解析器）─────────────")
        print(f"  │ 已翻译条件（{len(result['filters'])} 条）：")
        for i, (field, cond) in enumerate(result["filters"].items(), 1):
            label = _field_display_label(field)
            constraints: list[str] = []
            if "min" in cond:
                constraints.append(str(cond["min"]))
            constraints.append("~")
            if "max" in cond:
                constraints.append(str(cond["max"]))
            print(f"  │   {i}. {label}  {''.join(constraints)}")
        if result.get("unsupported"):
            print(f"  │")
            print(f"  │ 不支持条件（{len(result['unsupported'])} 条）：")
            for i, u in enumerate(result["unsupported"], 1):
                print(f"  │   {i}. {u}")
        print(f"  └───────────────────────────────────────────")
        return result


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

    # 从财务数据字典动态获取字段关键词（精确匹配层）
    from app.knowledge.dict import get_dict
    _field_keywords = get_dict().field_keywords()

    # 柔性正则模式（匹配自然语言描述的变体，弥补纯关键词精确匹配的不足）
    # 策略描述用的是自然语言（如"流动资产至少应该是流动负债的两倍"），
    # 而不是字段标签（如"流动比率"），所以需要 regex 模式做兜底。
    _field_patterns: dict[str, list[str]] = {
        "pe_3yr_avg": [
            r"3.*年.*平均.*利润|三年平均.*利润|过去.*3.*年.*平均.*利润",
            r"最近3年.*EPS均值|3年.*平均.*PE|平均利润的?\d+倍",
        ],
        "eps_growth_10yr_3yr_avg": [
            r"每股收益.*增长|每股利润.*增长|EPS.*增长",
            r"三年平均.*对比|期初.*期末.*三年",
        ],
        "current_ratio": [
            r"流动资产.*流动负债|流动负债.*流动资产",
            r"流动资产.*至少.*流动负债|流动负债.*两倍",
        ],
        "debt_to_equity": [
            r"负债.*股权|负债.*权益|股权.*负债|资产负债(?!表)",
            r"负债.*不应.*超过.*股权|debt.*equity"
        ],
        "long_term_debt_to_net_ca_ratio": [
            r"长期债务.*流动资产|长期债务.*不超过.*流动",
            r"长期借款.*流动资产",
        ],
        "long_term_borrowings": [
            r"长期债务|长期借款|长期负债|long.term.borrow",
        ],
        "net_current_assets": [
            r"流动资产净额|净流动资产|net.current.asset",
        ],
        "consecutive_profitable_years": [
            r"每年.*利润|普通股每年|每年.*盈利|连续盈利",
        ],
        "pe_pb": [
            r"市盈率.*市净率|市盈率.*账面值|市净率.*市盈率|账面值.*市盈率",
            r"市盈率.*乘积|市盈率.*之积|乘积|之积",
        ],
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

        # 第一层：精确关键词匹配（re.escape 处理含括号等特殊字符的标签）
        found = any(re.search(re.escape(kw), desc_lower, re.IGNORECASE) for kw in keywords)
        # 第二层：柔性正则模式匹配（处理自然语言描述，如"负债不应超过股权两倍" → debt_to_equity）
        if not found and field in _field_patterns:
            found = any(
                re.search(pat, desc_lower, re.IGNORECASE)
                for pat in _field_patterns[field]
            )
        if found:
            kept[field] = cond
        else:
            display = _field_display_label(field)
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
            display = _field_display_label(field)
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


# monetary fields that should display as 亿 unit
_MONETARY_FIELDS = {"market_cap", "revenue", "total_assets", "long_term_borrowings", "net_current_assets"}

# percentage fields (already in %, just show as-is)
_PCT_FIELDS = {"roe", "dividend_yield", "dividend_payout_ratio"}


def _resolve_field_value(factors: dict[str, Any], field: str) -> float | None:
    """Resolve a field's value from factors dict, including computed fields."""
    if field == "pe_pb":
        pe = _to_num(factors.get("pe_ttm"))
        pb = _to_num(factors.get("pb"))
        return pe * pb if pe is not None and pb is not None else None
    if field == "net_current_assets":
        tca = _to_num(factors.get("total_current_assets"))
        tcl = _to_num(factors.get("total_current_liabilities"))
        return tca - tcl if tca is not None and tcl is not None else None
    if field == "long_term_debt_to_net_ca_ratio":
        ltb = _to_num(factors.get("long_term_borrowings"))
        nca = _to_num(factors.get("net_current_assets"))
        if ltb is not None and nca is not None and nca > 0:
            return ltb / nca
        tca = _to_num(factors.get("total_current_assets"))
        tcl = _to_num(factors.get("total_current_liabilities"))
        if ltb is not None and tca is not None and tcl is not None:
            nca2 = tca - tcl
            if nca2 > 0:
                return ltb / nca2
        return None
    return _to_num(factors.get(field))


def _format_score_table(
    scored: list[Any],
    max_rows: int = 30,
    show_score: bool = True,
    filter_fields: list[str] | None = None,
) -> str:
    """把 RuleVerdict 列表格式化为终端文本表格。

    Args:
        scored:        排序后的 RuleVerdict 列表。
        max_rows:      最多显示行数。
        show_score:    是否显示评分列。
        filter_fields: 筛选条件字段列表，每字段生成一列显示对应股票的实际值。
                       未提供时使用默认列（PE/PB/ROE/股息率/市值）。
    """
    if not scored:
        return "(无结果)"

    rows = scored[:max_rows]

    def fmt_float(val: Any, precision: int = 2) -> str:
        if val is None:
            return "-"
        fv = float(val)
        return f"{fv:.{precision}f}"

    # ── build headers ──
    if filter_fields:
        field_labels = [_field_display_label(f) for f in filter_fields]
        if show_score:
            headers = ["排名", "代码", "名称", "总分"] + field_labels
        else:
            headers = ["序号", "代码", "名称"] + field_labels
    else:
        if show_score:
            headers = ["排名", "代码", "名称", "总分", "PE_TTM", "PB", "ROE%", "股息率%", "市值(亿)"]
        else:
            headers = ["序号", "代码", "名称", "PE_TTM", "PB", "ROE%", "股息率%", "市值(亿)"]

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

        if filter_fields:
            # ── dynamic columns from filter conditions ──
            vals: list[str] = []
            for field in filter_fields:
                val = _resolve_field_value(factors, field)
                if val is None:
                    vals.append("-")
                elif field in _MONETARY_FIELDS:
                    vals.append(f"{val / 1e8:.1f}")
                elif field in _PCT_FIELDS:
                    vals.append(f"{val:.1f}")
                elif field == "eps_growth_10yr_3yr_avg":
                    vals.append(f"{val:.3f}")
                else:
                    vals.append(f"{val:.2f}")

            if show_score:
                cols = [str(i + 1), str(v.symbol), str(v.name or "")[:8], f"{v.total_score:.2f}"] + vals
            else:
                cols = [str(i + 1), str(v.symbol), str(v.name or "")[:8]] + vals
        else:
            # ── fallback: hardcoded old columns ──
            pe = factors.get("pe_ttm")
            pb = factors.get("pb")
            roe = factors.get("roe")
            div_y = factors.get("dividend_yield")
            mcap = factors.get("market_cap")

            if show_score:
                cols = [
                    str(i + 1), str(v.symbol), str(v.name or "")[:8],
                    f"{v.total_score:.2f}",
                    fmt_float(pe), fmt_float(pb), fmt_float(roe, 1),
                    fmt_float(div_y, 2) if div_y is not None else "-",
                    fmt_float(mcap / 1e8, 1) if mcap is not None else "-",
                ]
            else:
                cols = [
                    str(i + 1), str(v.symbol), str(v.name or "")[:8],
                    fmt_float(pe), fmt_float(pb), fmt_float(roe, 1),
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
                label = _field_display_label(rule.rule_name)
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
        if fs.price_count == 0:
            # 表为空，确实需要先刷新
            return {
                "factor_result": None, "rule_result": None, "table": "",
                "top_symbols": [], "save_name": "", "filters_used": {},
                "freshness": fs,
            }
        # 数据过期但表有数据：后台刷新，本次继续用旧数据
        print(f"  [!] 行情数据已过期（{fs.price_reason}），后台刷新中，本次使用缓存数据")
        from app.services.data_refresher import refresh_basic_info
        import threading
        threading.Thread(target=refresh_basic_info, args=(str(db),), daemon=True).start()

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
            print(f"  ┌─ 策略翻译结果（via LLM）──────────────────")
            print(f"  │ 已翻译条件（{len(translated_filters)} 条）：")
            for i, (field, cond) in enumerate(translated_filters.items(), 1):
                label = _field_display_label(field)
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

    # ---- 5. 加载数据（四表合并：basic_info + income_statement + balance_sheet + dividend_history）----
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

    dividend_rows = load_dividend_history(str(db))
    print(f"  [OK] dividend_history: {len(dividend_rows)} 条")

    # ---- 6. 计算因子（合并四表）----
    factor_result = compute_factors_with_financials(basic_rows, balance_rows, income_rows, dividend_rows)
    print(f"  [OK] 计算 {factor_result.output_count} 只股票的因子")

    # ---- 6.3. 加载历史利润表数据并计算历史趋势因子 ---- 
    income_history = load_income_statement_history(str(db))
    if income_history:
        factor_result = enrich_with_historical_factors(factor_result, income_history)
        sample_count = sum(1 for fs in factor_result.factor_sets if fs.eps_growth_10yr_3yr_avg is not None)
        print(f"  [OK] 历史趋势因子: eps_growth_10yr（{sample_count} 只有效数据）")

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

    # ---- 8. 生成表格（列 = 筛选条件字段）----
    hard_filters = config.get("screen", {}).get("hard_filters", {})
    filter_fields = list(hard_filters.keys()) if hard_filters else None
    table = _format_score_table(rule_result.scored, max_results, show_score=False, filter_fields=filter_fields)

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
    "dividend_yield": "股息率(%)", "pretax_bonus_per_share": "每股股利(元)",
    "dividend_years_count": "连续分红(年)", "market_cap": "市值(元)",
    "eps_basic": "EPS", "debt_to_equity": "负债权益比", "current_ratio": "流动比率",
    "pe_pb": "PE×PB",
    "revenue": "营收(元)", "total_assets": "总资产(元)",
    "eps_growth_10yr_3yr_avg": "EPS10年增长", "consecutive_profitable_years": "连续盈利(年)",
    "pe_3yr_avg": "PE(3年均)",
    "long_term_borrowings": "长期债务(元)", "net_current_assets": "流动资产净额(元)",
    "long_term_debt_to_net_ca_ratio": "长期债务/流动资产净额",
}


def _field_display_label(field: str) -> str:
    """获取字段的展示标签：优先用内置简写，兜底查财务数据字典。"""
    if field in _FIELD_LABEL:
        return _FIELD_LABEL[field]
    # 兜底：从财务数据字典查找 label_zh
    try:
        from app.knowledge.dict import get_dict
        labels = get_dict().field_labels()
        if field in labels:
            return labels[field]
    except Exception:
        pass
    return field


def _format_filters(filters: dict[str, Any]) -> str:
    """把筛选条件可视化为简短字符串。"""
    parts: list[str] = []
    for k, v in filters.items():
        label = _field_display_label(k)
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
