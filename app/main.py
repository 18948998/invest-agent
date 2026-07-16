from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from app.pipeline.ingest import run_fundamental_ingest, run_price_ingest, run_financial_ingest
from app.services.fundamental_data import (
    default_symbols,
    list_a_share_symbols,
    list_main_gem_star_symbols,
)
from app.storage.sqlite_repo import record_update, get_all_update_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run invest-agent data ingestion.")
    parser.add_argument(
        "--source",
        default="akshare",
        help="数据源名称。支持：akshare(默认)、mock(样例数据)。",
    )
    parser.add_argument(
        "--symbols",
        default="all",
        help="股票代码列表，或使用 'all'（全A股）/'main_gem_star'（主板+创业板+科创板）。",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Optional cap on how many A-share symbols to ingest when using --symbols all.",
    )
    parser.add_argument(
        "--max-periods",
        type=int,
        default=1,
        help="How many latest report periods to keep for each financial statement table.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="How many symbols to fetch/process per batch.",
    )
    parser.add_argument(
        "--field-config-dir",
        default="configs/fundamental_fields",
        help="Directory containing dataset field-definition YAML files.",
    )
    parser.add_argument(
        "--db-path",
        default="D:/invest-agent-db/fundamental.db",
        help="SQLite path for processed data storage.",
    )
    parser.add_argument(
        "--track",
        choices=["price", "financial", "all"],
        default="all",
        help="数据轨道：price(股价/估值)、financial(三张财报)、all(全部，默认)。",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        default=False,
        help="只显示各 track 上次更新时间，不抓取数据。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # ---- history mode: just show last update times ----
    if args.history:
        records = get_all_update_history(Path(args.db_path))
        if not records:
            print("尚无更新记录。")
            return
        print("各轨道上次更新时间：")
        for rec in records:
            print(f"  track={rec['track']}, 更新于={rec['updated_at']}, "
                  f"股票数={rec['symbols_count']}, 状态={rec['status']}")
        return

    # ---- symbols resolution ----
    symbols_option = args.symbols.strip().lower()
    if symbols_option == "all":
        symbols = list_a_share_symbols(max_symbols=args.max_symbols)
    elif symbols_option == "main_gem_star":
        symbols = list_main_gem_star_symbols(max_symbols=args.max_symbols)
    else:
        symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
        if args.max_symbols is not None:
            symbols = symbols[: args.max_symbols]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if args.source.strip().lower() == "mock":
        print("【注意】当前使用的是 mock 样例数据源，仅用于演示/调试，不代表真实市场数据，请勿用于投资决策。")

    # ---- route to track ----
    if args.track == "price":
        summary = run_price_ingest(
            symbols=symbols,
            source=args.source,
            field_config_dir=Path(args.field_config_dir),
            db_path=Path(args.db_path),
            batch_size=args.batch_size,
        )
    elif args.track == "financial":
        summary = run_financial_ingest(
            symbols=symbols,
            source=args.source,
            field_config_dir=Path(args.field_config_dir),
            db_path=Path(args.db_path),
            max_periods=args.max_periods,
            batch_size=args.batch_size,
        )
    else:
        summary = run_fundamental_ingest(
            symbols=symbols,
            source=args.source,
            field_config_dir=Path(args.field_config_dir),
            db_path=Path(args.db_path),
            max_periods=args.max_periods,
            batch_size=args.batch_size,
        )

    # ---- record update timestamp ----
    record_update(
        db_path=Path(args.db_path),
        track=summary.track,
        symbols_count=len(symbols),
        status="OK",
    )

    print("invest-agent 数据入库完成。")
    print(f"数据轨道：{summary.track}")
    print(f"开始时间：{now}")
    print(f"数据源：{summary.source}")
    print(f"股票数量：{len(symbols)}")
    if summary.price_success_rate is not None:
        pct = summary.price_success_rate * 100
        actual = summary.dataset_summaries[0].row_count if summary.dataset_summaries else 0
        print(f"股价获取：{summary.price_success_count}/{actual} 只有效 ({pct:.1f}%)，候选 {summary.symbol_count} 只")
    print(f"保留财报期数：{args.max_periods}")
    print(f"批大小：{args.batch_size}")
    print(f"数据库：{summary.db_path}")
    print("数据集概览：")
    for item in summary.dataset_summaries:
        status = "正常" if item.validation.is_valid else "有缺失"
        print(
            f"- {item.dataset_name_zh} ({item.dataset_id}): "
            f"行数={item.row_count}, 批次={item.batch_count}, 状态={status}"
        )
        if item.validation.required_violations:
            print(f"  必填字段缺失：{item.validation.required_violations}")
        if item.sample_row:
            preview_keys = list(item.sample_row.keys())[:4]
            preview = {key: item.sample_row[key] for key in preview_keys}
            print(f"  示例：{preview}")


if __name__ == "__main__":
    main()

