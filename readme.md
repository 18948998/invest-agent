# invest-agent

一个根据价值投资逻辑的选股Agent

## 快速开始（Windows PowerShell）

在项目根目录执行一条命令，即可创建 `.venv`、安装依赖并启动项目：

```powershell
.\bootstrap.ps1
```

如果 PowerShell 在当前会话中阻止脚本执行，请先运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\bootstrap.ps1
```

## 自定义启动命令

你可以在安装依赖后传入自定义运行命令：

```powershell
.\bootstrap.ps1 -RunCommand "python -m app.main"
```

## 手动安装（可选）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m app.main
```

## 数据源处理与入库（新）

当前 `main` 已支持执行数据源拉取(默认 eastmoney 真实数据) -> 按 `configs/fundamental_fields/*.yaml` 字段定义标准化 -> 写入 SQLite。

默认运行：

```powershell
python -m app.main
```

默认会抓取**全部 A 股股票**，并对三张财务报表仅保留最近 `1` 期，避免首跑压力过大。
同时按默认 `--batch-size 20` 分批读取和处理，直到当前股票集合全部处理完成。

指定参数示例：

```powershell
python -m app.main --source mock --symbols "600519,000858,600036" --db-path "D:/invest-agent-db/fundamental.db"
```

抓取全部 A 股，但限制为前 200 只股票、最近 2 期财报：

```powershell
python -m app.main --symbols all --max-symbols 200 --max-periods 2 --db-path "D:/invest-agent-db/fundamental_all.db"
```

仅抓取 A 股主板 + 创业板 + 科创板：

```powershell
python -m app.main --symbols main_gem_star --max-periods 1 --batch-size 20
```

按 20 条一批（默认）可显式写成：

```powershell
python -m app.main --symbols all --batch-size 20
```

只获取股票价格
```powershell
python -m app.main --track price --symbols all~
```

