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
