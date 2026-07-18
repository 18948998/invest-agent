"""Fetch raw fundamental datasets."""

from __future__ import annotations

import time
from datetime import date
from typing import Any

import requests

# ---------------------------------------------------------------------------
# mock data
# ---------------------------------------------------------------------------

def _mock_row(dataset_id: str, symbol: str) -> dict[str, Any]:
	seed = sum(ord(ch) for ch in symbol)
	base_cash = 2_000_000_000 + (seed % 900) * 1_000_000
	base_profit = 400_000_000 + (seed % 500) * 1_000_000
	report_date = "2025-12-31"

	if dataset_id == "balance_sheet":
		total_assets = base_cash * 4.0
		total_liabilities = total_assets * (0.35 + (seed % 10) / 100)
		return {
			"股票代码": symbol, "股票简称": f"样例{symbol[-2:]}",
			"报告期": report_date, "公告日期": "2026-03-31",
			"货币资金": base_cash, "应收账款": base_cash * 0.3,
			"存货": base_cash * 0.2, "流动资产合计": base_cash * 1.8,
			"固定资产": base_cash * 1.4, "无形资产": base_cash * 0.25,
			"商誉": base_cash * 0.08, "资产总计": total_assets,
			"短期借款": base_cash * 0.15, "流动负债合计": total_liabilities * 0.55,
			"长期借款": total_liabilities * 0.2, "负债合计": total_liabilities,
			"股本": 1_200_000_000, "未分配利润": base_profit * 3,
			"归属于母公司股东权益合计": total_assets - total_liabilities,
		}
	if dataset_id == "income_statement":
		revenue = base_profit * (5 + (seed % 3)); net_profit = base_profit
		return {
			"股票代码": symbol, "股票简称": f"样例{symbol[-2:]}",
			"报告期": report_date, "公告日期": "2026-03-31",
			"营业总收入": revenue * 1.02, "营业收入": revenue,
			"营业成本": revenue * 0.58, "销售费用": revenue * 0.08,
			"管理费用": revenue * 0.06, "研发费用": revenue * 0.05,
			"财务费用": revenue * 0.02, "营业利润": net_profit * 1.15,
			"利润总额": net_profit * 1.2, "所得税费用": net_profit * 0.2,
			"净利润": net_profit, "归属于母公司股东的净利润": net_profit * 0.97,
			"扣除非经常性损益后的净利润": net_profit * 0.9,
			"基本每股收益": round(0.7 + (seed % 20) / 20, 4),
			"稀释每股收益": round(0.68 + (seed % 20) / 21, 4),
		}
	if dataset_id == "cash_flow_statement":
		ocf = base_profit * 1.25; capex = base_profit * 0.35
		return {
			"股票代码": symbol, "股票简称": f"样例{symbol[-2:]}",
			"报告期": report_date, "公告日期": "2026-03-31",
			"经营活动产生的现金流量流入小计": ocf * 2.5,
			"经营活动产生的现金流量流出小计": ocf * 1.5,
			"经营活动产生的现金流量净额": ocf,
			"购建固定资产、无形资产和其他长期资产支付的现金": capex,
			"投资活动产生的现金流量净额": -capex * 1.4,
			"筹资活动产生的现金流量净额": -ocf * 0.15,
			"现金及现金等价物净增加额": ocf - capex,
			"期末现金及现金等价物余额": base_cash + (ocf - capex),
		}
	if dataset_id == "basic_info":
		price = round(10 + (seed % 80) * 0.8, 2)
		eps = round(0.7 + (seed % 15) * 0.12, 4)
		return {
			"股票代码": symbol, "股票简称": f"样例{symbol[-2:]}",
			"市场": "A股", "行业": "制造业", "上市日期": "2008-06-18",
			"最新价": price, "市盈率TTM": round(price / max(eps, 0.01), 2),
			"市净率": round(0.9 + (seed % 25) * 0.12, 2),
			"市销率TTM": round(0.8 + (seed % 20) * 0.2, 2),
			"总市值": round(price * 1_200_000_000, 2),
			"基本每股收益": eps, "每股收益TTM": round(eps * 1.05, 4),
			"每股净资产": round(4 + (seed % 30) * 0.25, 4),
			"每股派息": round(0.05 + (seed % 10) * 0.04, 4),
			"分红率": round(0.15 + (seed % 30) / 100, 4),
			"股息率": round(0.01 + (seed % 12) / 200, 4),
			"净资产收益率": round(0.08 + (seed % 20) / 100, 4),
		}
	raise ValueError(f"Unsupported dataset_id: {dataset_id}")

# ---------------------------------------------------------------------------
# symbol list – 有效期码缓存，每月刷新
# ---------------------------------------------------------------------------

_VALID_FILE = "data/valid_codes.json"


def list_a_share_symbols(max_symbols: int | None = None) -> list[str]:
	codes = _load_valid_codes()
	if codes:
		print(f"【股票列表】有效代码 {len(codes)} 只（来自 data/valid_codes.json）", flush=True)
	else:
		codes = _generate_a_share_codes()
		print(f"【股票列表】全量生成 {len(codes)} 只（首次或文件过期）", flush=True)
	return codes[:max_symbols] if max_symbols else codes


def save_valid_codes_from_rows(rows: list[dict[str, Any]]) -> None:
	"""从行情结果中提取 A 股代码，去重持久化。"""
	codes = sorted({r["股票代码"] for r in rows if is_main_gem_star_symbol(str(r["股票代码"]))})
	if len(codes) < 1000:
		return
	import json, os
	os.makedirs("data", exist_ok=True)
	with open(_VALID_FILE, "w", encoding="utf-8") as f:
		json.dump({"codes": codes, "count": len(codes), "updated": _fmt_date(date.today().isoformat())}, f, ensure_ascii=False)
	print(f"【股票列表】有效代码已保存 {len(codes)} 只 → data/valid_codes.json", flush=True)


def _load_valid_codes() -> list[str] | None:
	import json, os
	if not os.path.exists(_VALID_FILE):
		return None
	mtime = os.path.getmtime(_VALID_FILE)
	if time.time() - mtime > 86400 * 30:  # 30 天过期
		return None
	with open(_VALID_FILE, encoding="utf-8") as f:
		data = json.load(f)
	codes = data.get("codes", [])
	return codes if isinstance(codes, list) and len(codes) > 500 else None


def _generate_a_share_codes() -> list[str]:
	codes: list[str] = []
	# 上海主板 A 股（仅 600-605）
	for base in range(600000, 606000): codes.append(str(base))
	# 科创板
	for base in range(688000, 690000): codes.append(str(base))
	# 深圳主板 A 股（仅 000-003）
	for base in range(1, 4000): codes.append(f"{base:06d}")
	# 创业板
	for base in range(300001, 302000): codes.append(str(base))
	# 过滤非 A 股（B 股 9xxx、新三板 4/8 等）
	codes = [c for c in codes if is_main_gem_star_symbol(c)]
	return codes

def list_main_gem_star_symbols(max_symbols: int | None = None) -> list[str]:
	f = [s for s in list_a_share_symbols() if is_main_gem_star_symbol(s)]
	return f[:max_symbols] if max_symbols else f

def is_main_gem_star_symbol(s: str) -> bool:
	return s.startswith(("000","001","002","003","300","301","600","601","603","605","688","689"))

# ---------------------------------------------------------------------------
# fetch_raw_dataset
# ---------------------------------------------------------------------------

def fetch_raw_dataset(dataset_id: str, symbols: list[str], source: str = "akshare",
                      max_periods: int = 1) -> list[dict[str, Any]]:
	if source == "mock":
		return [_mock_row(dataset_id, s) for s in symbols]
	if dataset_id == "basic_info":
		return _fetch_basic_info_baostock(symbols)
	return _fetch_financial_direct(symbols, max_periods, dataset_id)

# ---------------------------------------------------------------------------
# basic_info – 新浪/腾讯行情 :: 双源降级
# ---------------------------------------------------------------------------

_BATCH = 400


def _fetch_basic_info_baostock(symbols: list[str]) -> list[dict[str, Any]]:
	total = len(symbols)
	print(f"【行情】{total} 只，每批 {_BATCH}，腾讯优先", flush=True)
	all_rows: list[dict[str, Any]] = []
	batch_no = 0

	for start in range(0, total, _BATCH):
		batch = symbols[start : start + _BATCH]
		batch_no += 1

		rows = _fetch_tencent(batch)
		if not rows:
			rows = _fetch_sina(batch)

		all_rows.extend(rows)
		print(f"  第 {batch_no} 批 {len(rows)} 只，累计 {len(all_rows)}", flush=True)
		if not rows:
			time.sleep(0.5)

	print(f"【行情】共 {len(all_rows)} 只", flush=True)
	return all_rows


def _fetch_sina(symbols: list[str]) -> list[dict[str, Any]]:
	codes = ",".join(f"{'sh' if s.startswith('6') else 'sz'}{s}" for s in symbols)
	try:
		r = requests.get("https://hq.sinajs.cn/list=" + codes,
			headers={"Referer": "https://finance.sina.com.cn"}, timeout=10)
		r.encoding = "gbk"
		return _parse_sina(r.text)
	except Exception:
		return []


def _parse_sina(text: str) -> list[dict[str, Any]]:
	rows = []
	for line in text.strip().split("\n"):
		line = line.strip()
		if not line or '=""' in line:
			continue
		# var hq_str_sh600519="名称,今开,昨收,当前价,..."
		try:
			raw = line.split("hq_str_")[1].split("=")[0]
		except IndexError:
			continue
		code = raw[2:] if raw[:2] in ("sh", "sz") else raw
		fields = line.split('="')[1].rstrip('";').split(",")
		if len(fields) < 4:
			continue
		rows.append(_build_row(code, fields[0], _to_float(fields[3])))
	return rows


def _fetch_tencent(symbols: list[str]) -> list[dict[str, Any]]:
	codes = ",".join(f"{'sh' if s.startswith('6') else 'sz'}{s}" for s in symbols)
	try:
		r = requests.get("http://qt.gtimg.cn/q=" + codes, timeout=10)
		r.encoding = "gbk"
		return _parse_tencent(r.text)
	except Exception:
		return []


def _parse_tencent(text: str) -> list[dict[str, Any]]:
	rows = []
	for line in text.strip().split("\n"):
		line = line.strip()
		if not line or '=""' in line:
			continue
		# v_sh600519="1~名~代码~现价~昨收~今开~...~PE(39)~...~市值(44)~PB(46)"
		try:
			raw = line.split("_")[1].split("=")[0]
		except IndexError:
			continue
		code = raw[2:]
		fields = line.split('="')[1].rstrip('";').split("~")
		if len(fields) < 4:
			continue
		pe = _to_float(fields[39]) if len(fields) > 39 else None
		pb = _to_float(fields[46]) if len(fields) > 46 else None
		market_cap_raw = _to_float(fields[44]) if len(fields) > 44 else None
		market_cap = market_cap_raw * 1e8 if market_cap_raw is not None else None  # 腾讯API返回亿→转为元
		rows.append(_build_row(code, fields[1], _to_float(fields[3]), pe=pe, pb=pb, market_cap=market_cap))
	return rows


def _build_row(code: str, name, price, pe=None, pb=None, market_cap=None):
	mkt = "上交所" if code.startswith("6") else "深交所"
	return {
		"股票代码": code, "股票简称": name if name else None, "市场": mkt,
		"行业": None, "上市日期": None, "最新价": price,
		"市盈率TTM": pe, "市净率": pb, "市销率TTM": None, "总市值": market_cap,
		"基本每股收益": None, "每股收益TTM": None, "每股净资产": None,
		"每股派息": None, "分红率": None, "股息率": None, "净资产收益率": None,
	}

# ---------------------------------------------------------------------------
# financial – datacenter-web per symbol
# ---------------------------------------------------------------------------

_H = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36","Referer":"https://data.eastmoney.com/"}
_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_REPORTS = {"balance_sheet":"RPT_F10_FINANCE_GBALANCE","income_statement":"RPT_F10_FINANCE_GINCOME","cash_flow_statement":"RPT_F10_FINANCE_GCASHFLOW","dividend":"RPT_SHAREBONUS_DET"}

def _fetch_financial_direct(symbols, max_periods, ds_id):
	"""串行拉取财报（保留兼容，新代码请用 fetch_one_financial_record）。"""
	report = _REPORTS[ds_id]; mapper = _MAPPERS[ds_id]; total = len(symbols)
	rows = []
	for i, sym in enumerate(symbols):
		if (i+1)%30==0 or i==total-1: print(f"  {ds_id} {i+1}/{total}", flush=True)
		try:
			data = _api_fetch(sym, report, max_periods)
			# 只保留年报和半年报，丢弃一季报/三季报
			data = [d for d in data if d.get("REPORT_TYPE") in ("年度报告", "半年度报告")]
			rows.extend(mapper(data, sym))
		except Exception: continue
	return rows


def fetch_one_financial_record(symbol: str, ds_id: str, max_periods: int) -> list[dict[str, Any]]:
	"""拉取单只股票的财报原始数据并经 mapper 映射。

	线程安全：所有状态均为函数局部变量，可被多线程并发调用。

	Args:
		symbol:      股票代码（如 '600519'）。
		ds_id:       数据集 ID（balance_sheet / income_statement / cash_flow_statement）。
		max_periods: 最多拉取期数。

	Returns:
		映射后的行列表，为空时返回 []。
	"""
	report = _REPORTS[ds_id]
	mapper = _MAPPERS[ds_id]
	data = _api_fetch(symbol, report, max_periods)
	# 只保留年报：A 股会计准则规定财年截止日为 12-31，因此年报 report_date 后缀必为 -12-31
	# 注意：report_date 是报告期，不是发布日（announce_date 才是次年 3~4 月发布）
	data = [d for d in data if (
		str(d.get("REPORT_TYPE", "")) == "年度报告" or
		(d.get("REPORT_DATE") and "-12-31" in str(d["REPORT_DATE"]))
	)]
	return mapper(data, symbol)


def fetch_one_dividend_record(symbol: str) -> list[dict[str, Any]]:
	"""拉取单只股票全量历史分红送转记录（1990 年起）。

	与财报接口使用同一个 datacenter-web base URL，但参数不同：
	- 用 startDate/endDate 控制时间范围而非 pageSize 限制期数
	- 排序字段使用 PLAN_NOTICE_DATE

	线程安全：所有状态均为函数局部变量，可被多线程并发调用。

	Args:
		symbol: 股票代码（如 '600519'）。

	Returns:
		映射后的分红行列表，为空时返回 []。
	"""
	data = _api_fetch_dividend(symbol)
	if not data:
		return []
	mapper = _MAPPERS["dividend"]
	return mapper(data, symbol)


def _api_fetch_dividend(symbol: str) -> list[dict[str, Any]]:
	"""从东方财富数据中心拉取单只股票全量历史分红明细。

	使用 reportName=RPT_SHAREBONUS_DET，通过 startDate/endDate
	获取所有历史记录（最大 pageSize=500，覆盖全量）。"""
	secu = f"{symbol}.SH" if symbol.startswith(("6", "9")) else f"{symbol}.SZ"
	params = {
		"reportName": _REPORTS["dividend"],
		"columns": "ALL",
		"filter": f'(SECUCODE="{secu}")',
		"pageNumber": "1",
		"pageSize": "500",
		"sortTypes": "-1",
		"sortColumns": "PLAN_NOTICE_DATE",
		"startDate": "1990-01-01",
		"endDate": "2099-12-31",
	}
	for attempt in range(3):
		try:
			r = requests.get(_URL, params=params, headers=_H, timeout=15)
			r.raise_for_status()
			d = r.json().get("result", {}).get("data")
			return d if isinstance(d, list) else []
		except Exception:
			if attempt < 2:
				time.sleep(1 + attempt)
	return []

def _api_fetch(symbol, report_name, N):
	secu = f"{symbol}.SH" if symbol.startswith(("6","9")) else f"{symbol}.SZ"
	for attempt in range(3):
		try:
			r = requests.get(_URL, params={"reportName":report_name,"columns":"ALL","filter":f'(SECUCODE="{secu}")',"pageNumber":"1","pageSize":str(N),"sortTypes":"-1","sortColumns":"REPORT_DATE"}, headers=_H, timeout=15)
			r.raise_for_status()
			d = r.json().get("result",{}).get("data")
			return d if isinstance(d,list) else []
		except Exception:
			if attempt<2: time.sleep(1+attempt)
	return []

_REPORT_TYPE_MAP = {"004": "年度报告"}

def _get_report_type(r):
	"""根据 report_date 后缀判断报告类型（A 股年报 report_date 始终为 YYYY-12-31）。"""
	rt = r.get("REPORT_TYPE")
	if rt: return rt
	rd = str(r.get("REPORT_DATE", ""))
	if "-12-31" in rd:
		return "年报"
	return None

_SHARE_MAP: dict[str, float] = {}

def _parse_float(v):
	if v is None or str(v) in ("nan",""): return None
	return float(v)

_MAPPERS = {
"balance_sheet": lambda rows,sym: [{"股票代码":r.get("SECURITY_CODE") or sym,"股票简称":r.get("SECURITY_NAME_ABBR"),"报告期":_fmt_date(r.get("REPORT_DATE")),"公告日期":_fmt_date(r.get("NOTICE_DATE")),"报告类型":_get_report_type(r),"货币资金":r.get("MONETARYFUNDS"),"应收账款":r.get("ACCOUNTS_RECE"),"存货":r.get("INVENTORY"),"流动资产合计":r.get("TOTAL_CURRENT_ASSETS"),"固定资产":r.get("FIXED_ASSET"),"无形资产":r.get("INTANGIBLE_ASSET"),"商誉":r.get("GOODWILL"),"资产总计":r.get("TOTAL_ASSETS"),"短期借款":r.get("SHORT_LOAN"),"流动负债合计":r.get("TOTAL_CURRENT_LIAB"),"长期借款":r.get("LONG_LOAN"),"负债合计":r.get("TOTAL_LIABILITIES"),"股本":r.get("SHARE_CAPITAL"),"未分配利润":r.get("UNASSIGN_RPOFIT"),"归属于母公司股东权益合计":r.get("TOTAL_PARENT_EQUITY")} for r in rows],
"income_statement": lambda rows,sym: [{"股票代码":r.get("SECURITY_CODE") or sym,"股票简称":r.get("SECURITY_NAME_ABBR"),"报告期":_fmt_date(r.get("REPORT_DATE")),"公告日期":_fmt_date(r.get("NOTICE_DATE")),"报告类型":_get_report_type(r),"营业总收入":r.get("TOTAL_OPERATE_INCOME"),"营业收入":r.get("OPERATE_INCOME"),"营业成本":r.get("OPERATE_COST"),"销售费用":r.get("SALE_EXPENSE"),"管理费用":r.get("MANAGE_EXPENSE"),"研发费用":r.get("RESEARCH_EXPENSE"),"财务费用":r.get("FINANCE_EXPENSE"),"营业利润":r.get("OPERATE_PROFIT"),"利润总额":r.get("TOTAL_PROFIT"),"所得税费用":r.get("INCOME_TAX"),"净利润":r.get("NETPROFIT"),"归属于母公司股东的净利润":r.get("PARENT_NETPROFIT"),"扣除非经常性损益后的净利润":r.get("DEDUCT_PARENT_NETPROFIT"),"基本每股收益":r.get("BASIC_EPS"),"稀释每股收益":r.get("DILUTED_EPS")} for r in rows],
"cash_flow_statement": lambda rows,sym: [{"股票代码":r.get("SECURITY_CODE") or sym,"股票简称":r.get("SECURITY_NAME_ABBR"),"报告期":_fmt_date(r.get("REPORT_DATE")),"公告日期":_fmt_date(r.get("NOTICE_DATE")),"报告类型":_get_report_type(r),"经营活动产生的现金流量流入小计":r.get("TOTAL_OPERATE_INFLOW"),"经营活动产生的现金流量流出小计":r.get("TOTAL_OPERATE_OUTFLOW"),"经营活动产生的现金流量净额":r.get("NETCASH_OPERATE"),"购建固定资产、无形资产和其他长期资产支付的现金":r.get("CONSTRUCT_LONG_ASSET"),"投资活动产生的现金流量净额":r.get("NETCASH_INVEST"),"筹资活动产生的现金流量净额":r.get("NETCASH_FINANCE"),"现金及现金等价物净增加额":r.get("CCE_ADD"),"期末现金及现金等价物余额":r.get("END_CCE")} for r in rows],
"dividend": lambda rows,sym: [{"股票代码":r.get("SECURITY_CODE") or sym,"股票简称":r.get("SECURITY_NAME_ABBR"),"分配方案":r.get("IMPL_PLAN_PROFILE"),"税前每股股利":r.get("PRETAX_BONUS_RMB"),"送转股比例":r.get("BONUS_IT_RATIO"),"分配进度":r.get("ASSIGN_PROGRESS"),"预案公告日":_fmt_date(r.get("PLAN_NOTICE_DATE")),"股权登记日":_fmt_date(r.get("EQUITY_RECORD_DATE")),"除权除息日":_fmt_date(r.get("EX_DIVIDEND_DATE")),"实施公告日":_fmt_date(r.get("NOTICE_DATE")),"报告期":_fmt_date(r.get("REPORT_DATE"))} for r in rows],
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt_date(v):
	if v is None or str(v) in ("nan",""): return None
	s = str(v)
	if len(s)>=10: return s[:10]
	if s.isdigit() and len(s)==8: return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
	return s

def _to_float(v, div=1.0):
	if v is None or str(v) in ("nan","","None"): return None
	try: return float(str(v).replace(",","").replace("%","")) / (div if div!=1.0 else 1.0)
	except: return None

def default_symbols(): return ["600519","000858","600036","601318","000333"]
def default_as_of_date(): return date.today().isoformat()
