"""Knowledge module —— agent 共享的外部会计知识记忆。

提供 FinancialDict 单例，从 configs/knowledge/financial_data_dict.yaml 加载，
生成 prompt 片段供 ScreenAgent 和 AnalyzeAgent 注入 LLM 上下文。
"""

from app.knowledge.dict import FinancialDict, get_dict

__all__ = ["FinancialDict", "get_dict"]
