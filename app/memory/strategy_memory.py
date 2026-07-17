"""策略记忆层 —— 对筛选策略的持久化 / 按名查找 / 注册。

策略文件统一存放在 configs/strategies/ 下，注册表在 data/agent_memory/strategy_registry.json。

暴露三个可被 agent 调用的 tool 函数：
  - list_strategies()   → 列出所有可用策略
  - load_strategy(name)  → 按名称加载策略配置
  - save_strategy(name, config) → 持久化用户自定义策略
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 注册表路径
_REGISTRY_ROOT = Path(__file__).parent.parent.parent / "data" / "agent_memory"
_REGISTRY_FILE = _REGISTRY_ROOT / "strategy_registry.json"

# 策略文件目录
_STRATEGIES_DIR = Path(__file__).parent.parent.parent / "configs" / "strategies"


# ==============================================================================
#  注册表读写
# ==============================================================================

def _read_registry() -> dict[str, Any]:
    """读取策略注册表，不存在则返回空结构。"""
    if _REGISTRY_FILE.exists():
        try:
            with _REGISTRY_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("策略注册表损坏，将重建: %s", e)
    return {"strategies": {}, "default": ""}


def _write_registry(data: dict[str, Any]) -> None:
    """写入策略注册表。"""
    _REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
    with _REGISTRY_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ==============================================================================
#  公开 API（可被 agent 当作 tool 调用）
# ==============================================================================

def list_strategies() -> dict[str, Any]:
    """列出所有已注册的筛选策略。

    Returns:
        dict: {"strategies": {...}, "default": "...", "count": N}
    """
    registry = _read_registry()
    return {
        "strategies": registry.get("strategies", {}),
        "default": registry.get("default", ""),
        "count": len(registry.get("strategies", {})),
    }


def load_strategy(name: str) -> dict[str, Any]:
    """按名称加载策略配置。

    查找顺序：
      1. 策略注册表 → 找到 file 字段指定的 YAML 路径
      2. configs/strategies/{name}.yaml 直接加载
      3. 都找不到则返回空字典

    Args:
        name: 策略名称（如 "graham"、"格雷厄姆价值"）

    Returns:
        dict: 策略 YAML 配置内容
    """
    import importlib
    yaml = importlib.import_module("yaml")

    # ---- 第一步：查注册表 ----
    registry = _read_registry()
    strategies = registry.get("strategies", {})

    # 支持按英文 key 和中文 name 两种方式查找
    entry = None
    for key, val in strategies.items():
        if key == name or val.get("name", "") == name:
            entry = val
            break

    if entry:
        config_path = Path(entry.get("file", ""))
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                logger.info("从注册表加载策略 '%s' → %s", name, config_path)
                return yaml.safe_load(f) or {}

    # ---- 第二步：直接按名称在 strategies/ 目录查找 ----
    direct_path = _STRATEGIES_DIR / f"{name}.yaml"
    if direct_path.exists():
        logger.info("按文件名直接加载策略 '%s' → %s", name, direct_path)
        with direct_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    logger.warning("策略 '%s' 未找到（注册表和文件系统均无）", name)
    return {}


def load_strategy_config_path(name: str) -> Path | None:
    """按名称查找策略文件路径（不解析 YAML，只返回路径）。

    用于向后兼容 screen_workflow 的 strategy_config_path 参数。
    """
    registry = _read_registry()
    strategies = registry.get("strategies", {})

    for key, val in strategies.items():
        if key == name or val.get("name", "") == name:
            config_path = Path(val.get("file", ""))
            if config_path.exists():
                return config_path

    direct_path = _STRATEGIES_DIR / f"{name}.yaml"
    if direct_path.exists():
        return direct_path

    return None


def save_strategy(name: str, config: dict[str, Any], description: str = "") -> dict[str, Any]:
    """持久化用户自定义筛选策略。

    1. 把配置写入 configs/strategies/{name}.yaml
    2. 更新策略注册表

    Args:
        name:        策略英文标识（如 "my_low_pe"）
        config:      策略配置字典（与 strategy.graham.yaml 结构对齐）
        description: 策略中文描述

    Returns:
        dict: {"success": bool, "name": str, "file": str}
    """
    import importlib
    yaml = importlib.import_module("yaml")

    _STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

    # 写 YAML
    file_path = _STRATEGIES_DIR / f"{name}.yaml"
    with file_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # 更新注册表
    registry = _read_registry()
    registry["strategies"][name] = {
        "name": description or name,
        "description": description or config.get("screen", {}).get("description", name),
        "file": str(file_path),
        "tags": config.get("tags", []),
        "created_at": str(date.today()),
    }
    # 如果是第一个策略，设为默认
    if not registry.get("default"):
        registry["default"] = name
    _write_registry(registry)

    logger.info("策略 '%s' 已保存 → %s", name, file_path)
    return {"success": True, "name": name, "file": str(file_path)}


# ==============================================================================
#  初始化（模块导入时自动注册内置策略）
# ==============================================================================

def init_builtin_strategies() -> None:
    """扫描 configs/strategies/ 目录，把未注册的 YAML 文件自动注册到注册表。"""
    if not _STRATEGIES_DIR.exists():
        return

    registry = _read_registry()
    strategies = registry.get("strategies", {})
    modified = False

    for yaml_file in sorted(_STRATEGIES_DIR.glob("*.yaml")):
        import importlib
        yaml = importlib.import_module("yaml")

        name = yaml_file.stem  # 文件名去掉 .yaml 后缀
        if name in strategies:
            continue  # 已注册，跳过

        # 尝试从 YAML 中提取描述
        try:
            with yaml_file.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        description = (
            cfg.get("screen", {}).get("description", "")
            or cfg.get("meta", {}).get("description", "")
            or name
        )

        strategies[name] = {
            "name": name,
            "description": description[:100],
            "file": str(yaml_file),
            "tags": [],
            "created_at": str(date.today()),
        }
        modified = True
        logger.info("自动注册策略 '%s'", name)

    if modified:
        if not registry.get("default"):
            registry["default"] = list(strategies.keys())[0]
        _write_registry(registry)


# 模块加载时自动注册内置策略
try:
    init_builtin_strategies()
except Exception:
    pass
