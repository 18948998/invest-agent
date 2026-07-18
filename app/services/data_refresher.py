"""数据刷新 —— 两套模式：

  1. **Tool 模式**（agent 显式调用，同步阻塞）：
     - refresh_basic_info()  —— 腾讯行情 → basic_info 表 → valid_codes.json
     - refresh_financials()  —— valid_codes.json → 东方财富 → 三张财报表（含分红送转，dividend.yaml 已纳入 financial track）

  2. **后台自动模式**（已有数据但过期，异步线程）：
     - ensure_data_fresh() —— 检查新鲜度 → 启动后台线程刷新

  链路：
    basic_info   → 腾讯行情接口 (batch=400) → normalize → save → _cache_valid_codes
    financials   → 从 valid_codes.json 读代码 → 东方财富接口逐只拉三张表

原则：
  - Tool 模式：数据为空时 agent 主动调用，同步等待结果
  - 后台模式：已有数据过期时自动后台刷新，不阻塞当前查询
  - 同一 track 正在刷新时不重复启动（防重入）
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.storage.sqlite_repo import DataFreshness, check_data_freshness, record_update

logger = logging.getLogger(__name__)

# 防重入：同一 track 正在刷新时不重复启动
_refresh_lock = threading.Lock()
_refreshing: set[str] = set()

# 腾讯行情接口单次可查询 400 只
_TENCENT_BATCH_SIZE = 400
# 财报逐只查询，每批 20 只（控制并发强度）
_FINANCIAL_BATCH_SIZE = 20

_VALID_CODES_FILE = "data/valid_codes.json"
_FIELD_CONFIG_DIR = Path("configs/fundamental_fields")


# ==============================================================================
#  内部工具
# ==============================================================================

def _get_symbols_for_price_refresh(db_path: Path) -> list[str]:
    """从 basic_info 表读取现有股票代码，用于增量刷新。"""
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute("SELECT DISTINCT symbol FROM basic_info").fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


def _get_symbols_for_financial_refresh(db_path: Path) -> list[str]:
    """读取有效股票代码用于财报刷新。

    优先从 valid_codes.json（basic_info 刷新后自动生成）读取；
    若文件不存在则从 basic_info 表兜底。
    """
    if os.path.exists(_VALID_CODES_FILE):
        try:
            data = json.loads(Path(_VALID_CODES_FILE).read_text(encoding="utf-8"))
            codes = data.get("codes", [])
            if isinstance(codes, list) and len(codes) > 0:
                logger.debug("从 valid_codes.json 读取 %d 只有效代码", len(codes))
                return codes
        except Exception:
            logger.warning("valid_codes.json 读取失败，降级为 basic_info 表")

    return _get_symbols_for_price_refresh(db_path)


# ==============================================================================
#  Tool: 同步刷新（agent 显式调用，阻塞等待结果）
# ==============================================================================

def refresh_basic_info(
    db_path: str | Path,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    """同步刷新 basic_info 表（股价/估值）。

    链路：腾讯行情接口 → normalize → validate → save → _cache_valid_codes → _enrich_price_metrics
    完成后 valid_codes.json 自动更新。

    Args:
        db_path: SQLite 数据库路径。
        symbols:  要刷新的股票代码列表，默认从 basic_info 表读取现有代码。

    Returns:
        {"success": True, "count": N, "db_path": str}
        {"success": False, "reason": str}
    """
    from app.pipeline.ingest import run_price_ingest

    track = "price"
    db = Path(db_path)

    with _refresh_lock:
        if track in _refreshing:
            return {"success": False, "reason": "basic_info 正在刷新中，请稍后再试"}
        _refreshing.add(track)

    try:
        target_symbols = symbols or _get_symbols_for_price_refresh(db)
        if not target_symbols:
            # 表为空 → 首次全量采集：生成全量 A 股代码
            from app.services.fundamental_data import list_a_share_symbols
            target_symbols = list_a_share_symbols()
            logger.info("basic_info 表为空，生成全量 A 股代码 %d 只", len(target_symbols))

        logger.info("tool: refresh_basic_info 开始（腾讯行情），%d 只...", len(target_symbols))
        print(f"\n  [refresh_basic_info] 正在用腾讯行情刷新 {len(target_symbols)} 只股票...")

        result = run_price_ingest(
            symbols=target_symbols,
            source="akshare",
            field_config_dir=_FIELD_CONFIG_DIR,
            db_path=db,
            batch_size=_TENCENT_BATCH_SIZE,
        )

        record_update(db, "price", result.price_success_count, "OK")

        msg = (
            f"basic_info 刷新完成: {result.price_success_count}/{result.symbol_count} 只有效股价"
        )
        logger.info("tool: %s", msg)
        print(f"  [refresh_basic_info] {msg}")
        return {
            "success": True,
            "count": result.price_success_count,
            "total": result.symbol_count,
            "db_path": str(db),
        }

    except Exception as exc:
        logger.exception("refresh_basic_info 失败")
        return {"success": False, "reason": str(exc)}
    finally:
        with _refresh_lock:
            _refreshing.discard(track)



def refresh_financials(
    db_path: str | Path,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    """同步刷新三张财报表（balance_sheet / income_statement / cash_flow_statement）。

    链路：从 valid_codes.json 读有效代码 → 东方财富接口逐只拉取

    Args:
        db_path: SQLite 数据库路径。
        symbols:  要刷新的股票代码列表，默认从 valid_codes.json 读取。

    Returns:
        {"success": True, "count": N, "db_path": str}
        {"success": False, "reason": str}
    """
    from app.pipeline.ingest import run_financial_ingest

    track = "financial"
    db = Path(db_path)

    with _refresh_lock:
        if track in _refreshing:
            return {"success": False, "reason": "财报正在刷新中，请稍后再试"}
        _refreshing.add(track)

    try:
        target_symbols = symbols or _get_symbols_for_financial_refresh(db)
        if not target_symbols:
            # 无 valid_codes.json 且 basic_info 为空 → 生成全量代码兜底
            from app.services.fundamental_data import list_a_share_symbols
            target_symbols = list_a_share_symbols()
            logger.info("无有效股票代码，生成全量 A 股代码 %d 只作为兜底", len(target_symbols))

        logger.info("tool: refresh_financials 开始（东方财富），%d 只...", len(target_symbols))
        print(f"\n  [refresh_financials] 正在用东方财富拉取 {len(target_symbols)} 只财报+分红...")

        run_financial_ingest(
            symbols=target_symbols,
            source="akshare",
            field_config_dir=_FIELD_CONFIG_DIR,
            db_path=db,
            max_periods=1,
            batch_size=_FINANCIAL_BATCH_SIZE,
        )

        record_update(db, "financial", len(target_symbols), "OK")
        # dividend_history 已在 run_financial_ingest 中随财报一同拉取
        # （dividend.yaml 的 statement_type=financial_statement，load_specs_by_track("financial") 会自动包含）
        record_update(db, "dividend", len(target_symbols), "OK")

        msg = f"财报&分红刷新完成: {len(target_symbols)} 只"
        logger.info("tool: %s", msg)
        print(f"  [refresh_financials] {msg}")
        return {
            "success": True,
            "count": len(target_symbols),
            "db_path": str(db),
        }

    except Exception as exc:
        logger.exception("refresh_financials 失败")
        return {"success": False, "reason": str(exc)}
    finally:
        with _refresh_lock:
            _refreshing.discard(track)


# ==============================================================================
#  后台自动模式（已有数据过期时，异步线程静默更新）
# ==============================================================================

def _bg_refresh_price(db_path: Path) -> None:
    """后台线程入口，委托给同步 tool。"""
    refresh_basic_info(db_path)


def _bg_refresh_financial(db_path: Path) -> None:
    """后台线程入口，委托给同步 tool。"""
    refresh_financials(db_path)



def ensure_data_fresh(db_path_str: str) -> DataFreshness:
    """检查数据新鲜度，必要时启动后台刷新线程。

    每次 data_agent 查询前调用，确保用户看到的数据不会太旧。
    后台刷新不阻塞当前查询 —— 返回旧数据，新数据下次查询生效。

    对于「表为空」的情况，不启动后台线程 —— 由 data_agent._check_freshness
    决定是否同步调用 refresh_basic_info / refresh_financials。

    Returns:
        DataFreshness: 包含各 track 是否过期、上次更新时间等信息。
    """
    db_path = Path(db_path_str)
    freshness = check_data_freshness(db_path)

    # 只在有数据且过期时启动后台线程（无数据时应走 tool 同步刷新）
    if freshness.is_price_stale and db_path.exists():
        if _get_symbols_for_price_refresh(db_path):
            t = threading.Thread(
                target=_bg_refresh_price,
                args=(db_path,),
                daemon=True,
                name="refresh-price",
            )
            t.start()
            logger.info(
                "basic_info 已过期（上次更新: %s），启动后台刷新。",
                freshness.price_last_update or "从未",
            )

    if freshness.is_financial_stale and db_path.exists():
        if _get_symbols_for_financial_refresh(db_path):
            t = threading.Thread(
                target=_bg_refresh_financial,
                args=(db_path,),
                daemon=True,
                name="refresh-financial",
            )
            t.start()
            logger.info(
                "财报已过期（上次更新: %s），启动后台刷新。",
                freshness.financial_last_update or "从未",
            )

    return freshness
