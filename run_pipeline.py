"""全量数据采集：腾讯行情 → 有效代码 → 东方财富财报"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

ROOT = Path(__file__).parent
DB = ROOT / "data" / "standardized" / "invest.db"
CONFIG_DIR = ROOT / "configs" / "fundamental_fields"

print("=" * 60)
print("  全量数据采集")
print("=" * 60)

# Step 1: 获取全量股票代码
from app.services.fundamental_data import list_a_share_symbols

all_codes = list_a_share_symbols()
print(f"\n  [Step 1/3] 股票代码: {len(all_codes)} 只\n")

# Step 2: 腾讯接口拉 basic_info（批量400只，自动识别有效代码存 valid_codes.json）
print("  [Step 2/3] 腾讯行情 → basic_info\n")
from app.pipeline.ingest import run_price_ingest

price_result = run_price_ingest(
    symbols=all_codes,
    source="akshare",
    field_config_dir=CONFIG_DIR,
    db_path=DB,
    batch_size=400,
)
print(f"\n  basic_info 完成: {price_result.price_success_count}/{len(all_codes)} 只有效股价")

# Step 3: 读取有效代码，拉财报
codes_data = json.loads(Path("data/valid_codes.json").read_text(encoding="utf-8"))
valid_codes = codes_data["codes"]
print(f"\n  [Step 3/3] 东方财富财报: {len(valid_codes)} 只\n")
from app.pipeline.ingest import run_financial_ingest

fin_result = run_financial_ingest(
    symbols=valid_codes,
    source="akshare",
    field_config_dir=CONFIG_DIR,
    db_path=DB,
    max_periods=1,
    batch_size=20,
)

# 验证
import sqlite3
print("\n" + "=" * 60)
print("  采集结果")
print("=" * 60)
with sqlite3.connect(str(DB)) as conn:
    for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        cnt = conn.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
        print(f"  {t[0]}: {cnt} 行")
print("\n  完成！运行: python -m app.main")
