"""
配置管理 — 从 JSON 加载配置
"""
import json
from pathlib import Path
from typing import Any, Dict

from src.utils.exceptions import ConfigError
from src.utils.logger import logger

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings:
    """应用配置单例"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self, config_path: str = None) -> None:
        """加载配置"""
        if config_path is None:
            config_path = ROOT_DIR / "config" / "app.json"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        self._loaded = True
        logger.info(f"已加载配置: {config_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔（如 'dialog.model'）"""
        if not self._loaded:
            self.load()
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """设置配置值，支持点号分隔（如 'dialog.model'）"""
        if not self._loaded:
            self.load()
        keys = key.split(".")
        target = self._data
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    def save(self, config_path: str = None) -> None:
        """保存配置到文件"""
        if config_path is None:
            config_path = ROOT_DIR / "config" / "app.json"
        else:
            config_path = Path(config_path)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        logger.info(f"配置已保存: {config_path}")

    @property
    def data(self) -> Dict:
        if not self._loaded:
            self.load()
        return self._data


# 全局单例
settings = Settings()
