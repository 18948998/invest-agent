"""财务数据字典加载器。

从 configs/knowledge/financial_data_dict.yaml 加载，提供：
- prompt_tables():     生成"五张表及全部字段"prompt 片段
- prompt_computed():   生成"系统计算字段及公式"prompt 片段
- prompt_concepts():   生成"财务概念速查"prompt 片段
- prompt_full():       以上三部分合并
- field_aliases():     返回 {规范字段名: [别名列表]}，供降级解析器用
- field_keywords():    返回 {规范字段名: [关键词列表]}，供幻觉检测用
- field_labels():      返回 {规范字段名: 中文标签}
- find_field(keyword): 反查字段在哪个表
- find_concept(keyword): 反查财务概念
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class FinancialDict:
    """会计词典。一次加载 YAML，按需生成 prompt 片段或字段映射。"""

    def __init__(self, yaml_path: str | Path) -> None:
        raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        self.tables: dict[str, dict] = raw.get("tables", {})
        self.computed_fields: list[dict] = raw.get("computed_fields", [])
        self.concepts: list[dict] = raw.get("concepts", [])

    # ------------------------------------------------------------------
    #  Prompt 生成 —— 注入 LLM 上下文
    # ------------------------------------------------------------------

    def prompt_tables(self) -> str:
        """生成 prompt 片段：五张数据表及全部字段。

        ScreenAgent 在 translate_description() 中调用，
        AnalyzeAgent 在生成分析摘要时调用。
        """
        lines: list[str] = []

        for table_name, meta in self.tables.items():
            lines.append(f"### {table_name} —— {meta['name_zh']}")
            lines.append(f"> {meta.get('description', '')}")

            note = meta.get("note", "")
            if note:
                lines.append(f"> ⚠ {note}")

            lines.append("")
            lines.append("| 字段名 | 中文名 | 单位 | 说明 |")
            lines.append("|--------|--------|------|------|")

            for f in meta.get("fields", []):
                unit = f.get("unit") or ""
                desc = f.get("description", "")
                lines.append(f"| `{f['name']}` | {f['label_zh']} | {unit} | {desc} |")

            # 别名
            aliases_parts: list[str] = []
            for f in meta.get("fields", []):
                if f.get("aliases"):
                    aliases_parts.append(
                        f"`{f['name']}` → {' / '.join(f['aliases'])}"
                    )
            if aliases_parts:
                lines.append(f"\n字段别名：{'；'.join(aliases_parts)}")

            lines.append("")

        return "\n".join(lines)

    def prompt_computed(self) -> str:
        """生成 prompt 片段：系统自动计算字段及公式。"""
        lines: list[str] = [
            "## 系统计算字段（系统自动计算，LLM 可直接引用，无需指定数据来源）",
            "",
            "| 字段名 | 中文名 | 计算公式 | 单位 | 说明 |",
            "|--------|--------|----------|------|------|",
        ]

        for cf in self.computed_fields:
            lines.append(
                f"| `{cf['name']}` | {cf['label_zh']} | "
                f"{cf['formula']} | {cf.get('output_unit', '')} | "
                f"{cf.get('description', '')} |"
            )

        lines.append("")
        for cf in self.computed_fields:
            if cf.get("aliases"):
                lines.append(f"- `{cf['name']}` ⇄ {' / '.join(cf['aliases'])}")

        return "\n".join(lines)

    def prompt_concepts(self) -> str:
        """生成 prompt 片段：常用财务概念 → 公式 → 来源表。"""
        lines: list[str] = [
            "## 财务概念速查（用户提到的说法 → 对应字段/公式）",
            "",
            "| 用户可能说 | 公式 | 数据来源表 |",
            "|-----------|------|-----------|",
        ]

        for c in self.concepts:
            aliases = " / ".join(c.get("aliases", [])[:3])
            label = f"{c['name']}（{aliases}）" if aliases else c["name"]
            lines.append(
                f"| {label} | {c.get('formula', '')} | "
                f"{c.get('source_table', '')} |"
            )

        return "\n".join(lines)

    def prompt_full(self) -> str:
        """生成完整 prompt 片段（表字段 + 计算字段 + 概念）。"""
        return "\n\n".join([
            self.prompt_tables(),
            self.prompt_computed(),
            self.prompt_concepts(),
        ])

    # ------------------------------------------------------------------
    #  反向查询 —— 根据中文名/别名反查字段或概念
    # ------------------------------------------------------------------

    def find_field(self, keyword: str) -> dict | None:
        """根据中文名或别名反查字段所属表和完整信息。

        Returns:
            {"table": "balance_sheet", "field": {...}} 或 None
        """
        kw = keyword.strip()

        # 基础字段
        for table_name, meta in self.tables.items():
            for f in meta.get("fields", []):
                if f["name"] == kw or f["label_zh"] == kw:
                    return {"table": table_name, "field": f}
                if kw in f.get("aliases", []):
                    return {"table": table_name, "field": f}

        # 计算字段
        for cf in self.computed_fields:
            if cf["name"] == kw or cf["label_zh"] == kw:
                return {"table": "(computed)", "field": cf}
            if kw in cf.get("aliases", []):
                return {"table": "(computed)", "field": cf}

        return None

    def find_concept(self, keyword: str) -> dict | None:
        """根据中文名或别名反查财务概念。"""
        kw = keyword.strip()
        for c in self.concepts:
            if c["name"] == kw or kw in c.get("aliases", []):
                return c
        return None

    # ------------------------------------------------------------------
    #  映射表 —— 供降级解析器、幻觉检测等使用
    # ------------------------------------------------------------------

    def field_aliases(self) -> dict[str, list[str]]:
        """返回 {规范字段名: [别名列表]}。

        合并基础字段和计算字段的别名，供 _parse_conditions_fallback 降级解析器用。
        """
        result: dict[str, list[str]] = {}

        for meta in self.tables.values():
            for f in meta.get("fields", []):
                if f.get("aliases"):
                    result[f["name"]] = list(f["aliases"])

        for cf in self.computed_fields:
            if cf.get("aliases"):
                result[cf["name"]] = list(cf["aliases"])

        return result

    def field_keywords(self) -> dict[str, list[str]]:
        """返回 {规范字段名: [关键词列表]}。

        用于 _drop_hallucinated_fields 中的字段合法性校验（每一步 LLM 产出后都要校验）。
        关键词从 label_zh + aliases + name 自动聚合。
        """
        result: dict[str, list[str]] = {}

        for meta in self.tables.values():
            for f in meta.get("fields", []):
                keywords = {f["label_zh"], f["name"]}
                keywords.update(f.get("aliases", []))
                result[f["name"]] = list(keywords)

        for cf in self.computed_fields:
            keywords = {cf["label_zh"], cf["name"]}
            keywords.update(cf.get("aliases", []))
            result[cf["name"]] = list(keywords)

        return result

    def field_labels(self) -> dict[str, str]:
        """返回 {规范字段名: 中文标签}。"""
        result: dict[str, str] = {}

        for meta in self.tables.values():
            for f in meta.get("fields", []):
                result[f["name"]] = f["label_zh"]

        for cf in self.computed_fields:
            result[cf["name"]] = cf["label_zh"]

        return result


# ------------------------------------------------------------------
#  单例
# ------------------------------------------------------------------

_DICT_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "knowledge"
    / "financial_data_dict.yaml"
)


@lru_cache(maxsize=1)
def get_dict() -> FinancialDict:
    """获取财务数据字典单例。启动时首次调用加载 YAML，之后返回缓存。"""
    return FinancialDict(_DICT_PATH)
