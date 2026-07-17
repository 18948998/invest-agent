"""后台数据刷新 —— agent 工具调用时检查新鲜度，过期则在后台线程静默更新。

原则：
- 前端查询不阻塞（立即返回当前数据）
- 刷新在后台线程运行，完成后静默写库
- basic_info 超过 1 天 → 刷新；财报超过 1 个月 → 刷新
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from app.storage.sqlite_repo import DataFreshness, check_data_freshness

logger = logging.getLogger(__name__)

# 防重入：同一 track 正在刷新时不重复启动
_refresh_lock = threading.Lock()
_refreshing: set[str] = set()


def _get_all_symbols_from_db(db_path: Path) -> list[str]:
    """从 basic_info 表读取当前库中所有股票代码。"""
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute("SELECT DISTINCT symbol FROM basic_info").fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


def _refresh_price(db_path: Path) -> None:
    """后台刷新 basic_info（股价/估值）。"""
    from app.pipeline.ingest import run_price_ingest

    track = "price"
    with _refresh_lock:
        if track in _refreshing:
            return
        _refreshing.add(track)

    try:
        symbols = _get_all_symbols_from_db(db_path)
        if not symbols:
            logger.info("basic_info 表无数据，跳过刷新。")
            return
        logger.info("后台开始刷新 basic_info，%d 只股票...", len(symbols))
        run_price_ingest(
            symbols=symbols,
            source="akshare",
            field_config_dir=Path("configs/fundamental_fields"),
            db_path=db_path,
            batch_size=20,
        )
        logger.info("basic_info 后台刷新完成。")
    except Exception:
        logger.exception("basic_info 后台刷新失败")
    finally:
        with _refresh_lock:
            _refreshing.discard(track)


def _refresh_financial(db_path: Path) -> None:
    """后台刷新三张财报。"""
    from app.pipeline.ingest import run_financial_ingest

    track = "financial"
    with _refresh_lock:
        if track in _refreshing:
            return
        _refreshing.add(track)

    try:
        symbols = _get_all_symbols_from_db(db_path)
        if not symbols:
            logger.info("财报表无数据，跳过刷新。")
            return
        logger.info("后台开始刷新财报，%d 只股票...", len(symbols))
        run_financial_ingest(
            symbols=symbols,
            source="akshare",
            field_config_dir=Path("configs/fundamental_fields"),
            db_path=db_path,
            max_periods=1,
            batch_size=20,
        )
        logger.info("财报后台刷新完成。")
    except Exception:
        logger.exception("财报后台刷新失败")
    finally:
        with _refresh_lock:
            _refreshing.discard(track)


def ensure_data_fresh(db_path_str: str) -> DataFreshness:
    """检查数据新鲜度，必要时启动后台刷新线程。

    agent 工具每次查询前调用此函数，确保用户看到的数据不会太旧。
    刷新不阻塞当前查询 —— 返回的是旧数据，但后台新数据下次查就有了。

    Returns:
        DataFreshness: 包含各 track 是否过期、上次更新时间等信息。
    """
    db_path = Path(db_path_str)
    freshness = check_data_freshness(db_path)

    if freshness.is_price_stale:
        t = threading.Thread(
            target=_refresh_price,
            args=(db_path,),
            daemon=True,
            name="refresh-price",
        )
        t.start()
        logger.info(
            "basic_info 数据已过期（上次更新: %s），已启动后台刷新。",
            freshness.price_last_update or "从未",
        )

    if freshness.is_financial_stale:
        t = threading.Thread(
            target=_refresh_financial,
            args=(db_path,),
            daemon=True,
            name="refresh-financial",
        )
        t.start()
        logger.info(
            "财报数据已过期（上次更新: %s），已启动后台刷新。",
            freshness.financial_last_update or "从未",
        )

    return freshness
