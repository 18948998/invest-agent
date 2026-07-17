"""统一模型层 —— 所有 LLM 调用的唯一入口。

用法：
    from app.services.llm import get_chat_model

    llm = get_chat_model()               # 从环境变量自动配置
    llm = get_chat_model(temperature=0.7)  # 部分覆盖

    response = llm.invoke("你好")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ==============================================================================
#  环境变量键名
# ==============================================================================

ENV_API_KEY = "LLM_API_KEY"
ENV_BASE_URL = "LLM_BASE_URL"
ENV_MODEL_NAME = "LLM_MODEL_NAME"
ENV_TEMPERATURE = "LLM_TEMPERATURE"
ENV_MAX_TOKENS = "LLM_MAX_TOKENS"
ENV_TIMEOUT = "LLM_TIMEOUT"

# ==============================================================================
#  配置模型
# ==============================================================================


@dataclass
class LLMConfig:
    """LLM 连接配置，支持任意 OpenAI 兼容 API（OpenAI / DeepSeek / Qwen / Ollama 等）。"""

    api_key: str = ""
    base_url: str | None = None          # None = 使用提供商默认地址
    model_name: str = "deepseek-v4-pro"    # 默认模型，可覆盖
    temperature: float = 0.0              # 结构化任务建议 0.0
    max_tokens: int = 2048
    timeout: int = 60                     # 秒

    @classmethod
    def from_env(cls, **overrides: Any) -> LLMConfig:
        """从环境变量构建配置，overrides 可部分覆盖任意字段。"""
        temp_str = os.getenv(ENV_TEMPERATURE, "")
        max_tok_str = os.getenv(ENV_MAX_TOKENS, "")
        timeout_str = os.getenv(ENV_TIMEOUT, "")

        base = cls(
            api_key=os.getenv(ENV_API_KEY, ""),
            base_url=os.getenv(ENV_BASE_URL) or "https://api.deepseek.com",
            model_name=os.getenv(ENV_MODEL_NAME, "deepseek-v4-pro"),
            temperature=float(temp_str) if temp_str else 0.0,
            max_tokens=int(max_tok_str) if max_tok_str else 2048,
            timeout=int(timeout_str) if timeout_str else 60,
        )
        # 应用覆盖
        for k, v in overrides.items():
            if hasattr(base, k) and v is not None:
                setattr(base, k, v)
        return base

    def validate(self) -> None:
        """校验必填字段，缺失时抛出明确错误。"""
        if not self.api_key:
            raise ValueError(
                f"LLM API key 未设置。请设置环境变量 {ENV_API_KEY}，"
                f"或通过 LLMConfig.from_env(api_key='your-key') 传入。"
            )
        if not self.model_name:
            raise ValueError("LLM model_name 不能为空。")


# ==============================================================================
#  全局单例
# ==============================================================================

_chat_model_instance: Any = None
_current_config: LLMConfig | None = None


def create_chat_model(**overrides: Any) -> Any:
    """创建新的 ChatModel 实例（不缓存）。

    Args:
        **overrides: 覆盖 LLMConfig 的任意字段（api_key, base_url, model_name 等）。

    Returns:
        langchain ChatModel 实例。

    Raises:
        ImportError: 未安装 langchain-openai。
        ValueError: 配置校验失败。
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "需要 langchain-openai 包。请执行: pip install langchain-openai"
        )

    config = LLMConfig.from_env(**overrides)
    config.validate()

    kwargs: dict[str, Any] = {
        "model": config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
    }

    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url

    logger.info(
        "创建 ChatModel: model=%s base_url=%s temp=%.1f",
        config.model_name,
        config.base_url or "(default)",
        config.temperature,
    )
    return ChatOpenAI(**kwargs)


def get_chat_model(**overrides: Any) -> Any:
    """获取全局共享的 ChatModel 实例（懒加载 + 单例）。

    首次调用时从环境变量创建并缓存，后续直接返回缓存实例。
    如需重新配置，先调用 reset_chat_model()。
    """
    global _chat_model_instance, _current_config

    if _chat_model_instance is None or overrides:
        config = LLMConfig.from_env(**overrides)
        if _chat_model_instance is not None and config != _current_config:
            # 配置变化了，重建
            _chat_model_instance = None
        if _chat_model_instance is None:
            _chat_model_instance = create_chat_model(**overrides)
            _current_config = LLMConfig.from_env(**overrides)

    return _chat_model_instance


def reset_chat_model() -> None:
    """重置全局单例，下次调用 get_chat_model() 会重新创建。"""
    global _chat_model_instance, _current_config
    _chat_model_instance = None
    _current_config = None


# ==============================================================================
#  工具函数
# ==============================================================================

def is_available() -> bool:
    """检查 LLM 是否可用（配置完整 + 包已安装）。

    不会触发 API 网络调用，仅做本地检查。
    """
    try:
        import langchain_openai  # noqa: F401
    except ImportError:
        return False

    try:
        LLMConfig.from_env().validate()
        return True
    except ValueError:
        return False


# ==============================================================================
#  Prompt 模板
# ==============================================================================

INTENT_CLASSIFY_PROMPT = """你是一个投资助手意图分类器。分析用户输入，判断其意图。

意图类型及示例：
- screen: "帮我找市盈率低于15的股票"、"筛选格雷厄姆标的"、"推荐价值股"
- screen_save: "筛选并保存低市净率股票"、"把价值股存到我的策略里"、"保存筛选结果"
- analyze: "分析一下600519"、"茅台的基本面怎么样"、"看看000001"
- help: "你能做什么"、"帮助"、"怎么用"
- quit: "退出"、"再见"
- unknown: 无法识别

用户输入: {text}

请严格按以下 JSON 格式输出（不要输出任何其他内容）：
{{"intent": "<intent>", "confidence": <0.0-1.0>, "symbol": "<提取到的6位数字股票代码，没有则为空>"}}"""


STOCK_ANALYSIS_PROMPT = """你是一位资深价值投资者，请对以下股票的基本面数据进行分析。

股票信息：
- 代码: {symbol}
- 名称: {name}

估值指标：
- PE_TTM: {pe_ttm} 倍
- PB: {pb} 倍
- PS_TTM: {ps_ttm} 倍
- 总市值: {market_cap} 亿

盈利指标：
- ROE: {roe}%
- 盈利收益率(1/PE): {earnings_yield}
- EPS(基本): {eps_basic}
- 每股净资产: {bvps}

分红指标：
- 股息率: {dividend_yield}%
- 每股股利: {dividend_per_share}
- 分红率: {dividend_payout_ratio}%

财务健康：
- 负债权益比: {debt_to_equity}
- 流动比率: {current_ratio}
- 总资产: {total_assets} 亿

请从以下角度给出分析：
1. 估值水平（便宜/合理/偏贵）
2. 盈利能力
3. 分红回报
4. 财务健康度
5. 综合建议（建议关注 / 值得观察 / 暂时回避）

请用中文回答，保持专业、简洁，控制在 300 字以内。"""
