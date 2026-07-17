"""Run data-source ingestion: fetch, normalize, validate, persist."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

from app.config.loader import load_fundamental_field_specs, load_specs_by_track
from app.config.schema import DatasetSpec
from app.data.normalizer import normalize_records
from app.data.validator import ValidationResult, validate_records
from app.services.fundamental_data import fetch_raw_dataset, save_valid_codes_from_rows, is_main_gem_star_symbol
from app.storage.sqlite_repo import save_dataset, get_latest_eps_batch


@dataclass(frozen=True, slots=True)
class DatasetIngestSummary:
    dataset_id: str
    dataset_name_zh: str
    row_count: int
    batch_count: int
    validation: ValidationResult
    sample_row: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class IngestRunSummary:
    db_path: Path
    source: str
    track: str
    dataset_summaries: list[DatasetIngestSummary]
    symbol_count: int
    price_success_count: int
    price_success_rate: float | None  # 0.0~1.0，仅当 track=price 时有效，否则 None


def _field_label_map(spec) -> dict[str, str]:
    return {field.name: field.label_zh for field in spec.fields}


def _preview_fields(dataset_id: str, spec) -> list[str]:
    preferred = {
        "basic_info": ["symbol", "name", "current_price", "pe_ttm", "pb", "market_cap"],
        "balance_sheet": ["symbol", "name", "report_date", "announce_date", "total_assets", "total_liabilities"],
        "income_statement": ["symbol", "name", "report_date", "announce_date", "net_profit", "revenue"],
        "cash_flow_statement": ["symbol", "name", "report_date", "announce_date", "net_cash_flow_from_operating_activities"],
    }.get(dataset_id, [field.name for field in spec.fields])
    label_map = _field_label_map(spec)
    return [label_map.get(name, name) for name in preferred]


def _to_display_row(row: dict[str, object], spec) -> dict[str, object]:
    label_map = _field_label_map(spec)
    return {label_map.get(key, key): value for key, value in row.items()}


def _to_display_violations(violations: dict[str, int], spec) -> dict[str, int]:
    label_map = _field_label_map(spec)
    return {label_map.get(key, key): value for key, value in violations.items()}


def _print_saved_rows(dataset_id: str, batch_index: int, total_batches: int, rows: list[dict[str, object]], spec) -> None:
    dataset_label = {
        "basic_info": "基础信息表",
        "balance_sheet": "资产负债表",
        "income_statement": "利润表",
        "cash_flow_statement": "现金流量表",
    }.get(dataset_id, dataset_id)
    print(f"【已落库】{dataset_label} 第 {batch_index}/{total_batches} 批，共 {len(rows)} 条")
    keys = _preview_fields(dataset_id, spec)
    label_map = _field_label_map(spec)
    for row in rows:
        preview = ", ".join(f"{key}={row.get(name)}" for name, key in label_map.items() if key in keys)
        print(f"  - {preview}")


def _enrich_eps_from_db(db_path: Path, raw_records: list[dict[str, object]], source: str) -> None:
    """Read eps_basic / eps_ttm from income_statement table and merge into raw records.

    This avoids making extra API calls to the income statement endpoint
    just to fill two fields in basic_info.
    """
    symbols = [str(r.get("股票代码", r.get("symbol", ""))) for r in raw_records]
    symbol_set = set(symbols)
    if not symbol_set:
        return
    eps_map = get_latest_eps_batch(db_path, list(symbol_set))
    for row in raw_records:
        sym = str(row.get("股票代码", row.get("symbol", "")))
        eps_entry = eps_map.get(sym, {})
        # akshare source_keys: 基本每股收益 → eps_basic, 每股收益TTM → eps_ttm
        # 直接用 DB 值覆盖（None 时保留原值，DB 有值时用 DB）
        if eps_entry.get("eps_basic") is not None:
            row["基本每股收益"] = eps_entry["eps_basic"]
        if eps_entry.get("eps_ttm") is not None:
            row["每股收益TTM"] = eps_entry["eps_ttm"]


def _enrich_price_metrics(db_path: Path, _unused: int) -> None:
    """用财报数据回填 basic_info 的缺失股票名称。

    PE / PB / 市值直接使用腾讯行情 API 的原始返回值，不再用财报反算覆盖。
    原因：eps_basic 可能是单季报数据，price / eps_basic 会严重高估 PE。
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""\
                UPDATE basic_info SET name = (
                    SELECT s.name FROM income_statement s
                    WHERE s.symbol = basic_info.symbol LIMIT 1
                ) WHERE name IS NULL""")
            conn.commit()
            updated = conn.total_changes
    except sqlite3.OperationalError:
        updated = 0
    if updated:
        print(f"【指标计算】已补充 stock name，{updated} 条更新。")


def _cache_valid_codes(db_path: Path, symbols: set[str] | None = None) -> None:
    """将有效 A 股代码写入持久化文件（覆盖写入，不会累积）。

    Args:
        db_path: 数据库路径（仅用于兼容旧调用，symbols 非空时不使用）。
        symbols:  本次采集得到的有股价的股票代码集合。若为空则从 basic_info 表兜底提取。
    """
    codes: list[str]
    if symbols:
        codes = sorted(s for s in symbols if s and is_main_gem_star_symbol(s))
    else:
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT symbol FROM basic_info WHERE current_price IS NOT NULL"
                ).fetchall()
            codes = sorted(
                r[0] for r in rows if r[0] and is_main_gem_star_symbol(str(r[0]))
            )
        except Exception:
            return

    save_valid_codes_from_rows([{"股票代码": c} for c in codes])


def _run_price_parallel(
    symbols: list[str],
    source: str,
    spec: DatasetSpec,
    db_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    """双线程：拉取线程逐批获取原始数据，写库线程逐批标准化+落盘。

    生产线程按 batch_size 分批调 API，拿到原始数据直接丢队列；
    消费线程从队列取出、标准化、校验、写库。
    """
    q: Queue = Queue(maxsize=3)
    total_batches = max(1, (len(symbols) + batch_size - 1) // batch_size)
    results: dict[str, Any] = {
        "total_row_count": 0,
        "price_success_count": 0,
        "valid_symbols": set(),  # 本次采集的有股价 symbol，用于重建 valid_codes.json
        "batch_count": 0,
        "sample_row": None,
        "total_batches": total_batches,
        "total_required_violations": {},
    }

    def fetch_worker() -> None:
        for start in range(0, len(symbols), batch_size):
            batch_syms = symbols[start : start + batch_size]
            raw = fetch_raw_dataset(
                dataset_id=spec.meta.dataset_id,
                symbols=batch_syms,
                source=source,
                max_periods=1,
            )
            if spec.meta.dataset_id == "basic_info":
                _enrich_eps_from_db(db_path, raw, source)
            q.put(raw)
        q.put(None)  # 终止信号

    def save_worker() -> None:
        batch_no = 0
        while True:
            raw = q.get()
            if raw is None:
                break
            batch_no += 1
            normalized = normalize_records(spec, raw, source=source)
            if spec.meta.dataset_id == "basic_info":
                results["price_success_count"] += sum(
                    1 for r in normalized if r.get("current_price") is not None
                )
                # 收集本次采集的有效 symbol（用于重建 valid_codes.json）
                for r in normalized:
                    sym = r.get("symbol")
                    if sym and r.get("current_price") is not None:
                        results["valid_symbols"].add(sym)
            validation = validate_records(spec, normalized)
            save_dataset(db_path=db_path, dataset_spec=spec, rows=normalized, replace=False)
            _print_saved_rows(spec.meta.dataset_id, batch_no, total_batches, normalized, spec)

            results["total_row_count"] += len(normalized)
            if results["sample_row"] is None and normalized:
                results["sample_row"] = _to_display_row(normalized[0], spec)
            for key, count in validation.required_violations.items():
                results["total_required_violations"][key] = \
                    results["total_required_violations"].get(key, 0) + count
        results["batch_count"] = batch_no

    t1 = Thread(target=fetch_worker, daemon=True)
    t2 = Thread(target=save_worker, daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # 用本次采集结果重建 valid_codes.json（不从 DB 读，避免旧数据混入）
    _cache_valid_codes(db_path, symbols=results["valid_symbols"])

    # 全部落库后，从财报数据补充 name / PE / PB / 市值
    _enrich_price_metrics(db_path, results["price_success_count"])
    return results


def _run_ingest(
    symbols: list[str],
    source: str,
    specs: list[DatasetSpec],
    db_path: Path,
    track: str,
    max_periods: int = 1,
    batch_size: int = 20,
) -> IngestRunSummary:
    """Core ingestion pipeline for a list of DatasetSpecs."""
    summaries: list[DatasetIngestSummary] = []
    price_success_count = 0
    is_price_track = any(s.meta.dataset_id == "basic_info" for s in specs) and len(specs) == 1

    for spec in specs:
        total_row_count = 0
        total_required_violations: dict[str, int] = {}
        total_rows_checked = 0
        sample_row: dict[str, object] | None = None
        batch_count = 0

        # -------- basic_info：双线程，拉取与写库并行 --------
        if spec.meta.dataset_id == "basic_info":
            r = _run_price_parallel(symbols, source, spec, db_path, batch_size)
            total_row_count = r["total_row_count"]
            total_rows_checked = total_row_count
            price_success_count = r["price_success_count"]
            batch_count = r["batch_count"]
            sample_row = r["sample_row"]
            total_batches = r["total_batches"]
            total_required_violations = r["total_required_violations"]

        # -------- 其他数据集：保持原有分批逻辑 --------
        else:
            effective_batch = max(batch_size, 1)
            total_batches = max(1, (len(symbols) + effective_batch - 1) // effective_batch)

            for start in range(0, len(symbols), effective_batch):
                batch_symbols = symbols[start : start + effective_batch]
                raw_records = fetch_raw_dataset(
                    dataset_id=spec.meta.dataset_id,
                    symbols=batch_symbols,
                    source=source,
                    max_periods=max_periods,
                )
                normalized = normalize_records(spec, raw_records, source=source)
                batch_validation = validate_records(spec, normalized)
                save_dataset(
                    db_path=db_path,
                    dataset_spec=spec,
                    rows=normalized,
                    replace=False,
                )
                _print_saved_rows(spec.meta.dataset_id, batch_count + 1, total_batches, normalized, spec)

                batch_count += 1
                total_row_count += len(normalized)
                total_rows_checked += batch_validation.total_rows
                if sample_row is None and normalized:
                    sample_row = _to_display_row(normalized[0], spec)

                for key, count in batch_validation.required_violations.items():
                    total_required_violations[key] = total_required_violations.get(key, 0) + count

        validation = ValidationResult(
            dataset_id=spec.meta.dataset_id,
            total_rows=total_rows_checked,
            required_violations=_to_display_violations(total_required_violations, spec),
        )

        summaries.append(
            DatasetIngestSummary(
                dataset_id=spec.meta.dataset_id,
                dataset_name_zh=spec.meta.dataset_name_zh,
                row_count=total_row_count,
                batch_count=batch_count,
                validation=validation,
                sample_row=sample_row,
            )
        )

    denominator = price_success_count if is_price_track and price_success_count else None
    rate = (price_success_count / total_row_count) if is_price_track and total_row_count else None
    return IngestRunSummary(
        db_path=db_path,
        source=source,
        track=track,
        dataset_summaries=summaries,
        symbol_count=len(symbols),
        price_success_count=price_success_count,
        price_success_rate=rate,
    )


def run_price_ingest(
    symbols: list[str],
    source: str,
    field_config_dir: Path,
    db_path: Path,
    batch_size: int = 20,
) -> IngestRunSummary:
    """股价/估值等高频数据入库（basic_info）。"""
    specs = load_specs_by_track(field_config_dir, track="price")
    return _run_ingest(
        symbols=symbols,
        source=source,
        specs=specs,
        db_path=db_path,
        track="price",
        max_periods=1,
        batch_size=batch_size,
    )


def run_financial_ingest(
    symbols: list[str],
    source: str,
    field_config_dir: Path,
    db_path: Path,
    max_periods: int = 1,
    batch_size: int = 20,
) -> IngestRunSummary:
    """三张财报数据入库（balance_sheet + income_statement + cash_flow_statement）。"""
    specs = load_specs_by_track(field_config_dir, track="financial")
    return _run_ingest(
        symbols=symbols,
        source=source,
        specs=specs,
        db_path=db_path,
        track="financial",
        max_periods=max_periods,
        batch_size=batch_size,
    )


def run_fundamental_ingest(
    symbols: list[str],
    source: str,
    field_config_dir: Path,
    db_path: Path,
    max_periods: int = 1,
    batch_size: int = 20,
) -> IngestRunSummary:
    """全量数据入库（兼容旧接口，同时处理 price + financial 两条 track）。"""
    specs = load_fundamental_field_specs(field_config_dir)
    return _run_ingest(
        symbols=symbols,
        source=source,
        specs=specs,
        db_path=db_path,
        track="all",
        max_periods=max_periods,
        batch_size=batch_size,
    )






