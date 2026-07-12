"""
统一日志配置 — 支持文件轮转、模块级 logger、从配置动态加载
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class ModuleNameFormatter(logging.Formatter):
    """为日志增加更短的模块名显示"""

    def format(self, record: logging.LogRecord) -> str:
        if record.name.startswith("MyDearElysia"):
            if record.name == "MyDearElysia":
                record.module_name = "app"
            else:
                record.module_name = record.name.split(".")[-1]
        else:
            record.module_name = record.name
        return super().format(record)


def _build_formatter(fmt: str, datefmt: str | None = None) -> logging.Formatter:
    return ModuleNameFormatter(fmt, datefmt=datefmt)


def setup_logger(name: str = "MyDearElysia", level: int = logging.INFO) -> logging.Logger:
    """配置并返回 logger 实例"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    logger.disabled = False

    # 控制台输出：简洁、利于观察
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(
        _build_formatter(
            "[%(asctime)s] %(levelname)-5s | %(module_name)-12s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(console)

    # 文件输出（带轮转，最大 5MB，保留 3 份）
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        _build_formatter("[%(asctime)s] %(levelname)-8s | %(module_name)-20s | %(message)s")
    )
    logger.addHandler(file_handler)

    # 降低第三方库噪音
    for noisy_name in ("urllib3", "httpx", "requests", "PIL", "faster_whisper"):
        logging.getLogger(noisy_name).setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取子模块 logger（复用全局配置）"""
    return logging.getLogger(f"MyDearElysia.{name}")


def configure_from_settings():
    """从配置重新配置日志（在 settings.load() 之后调用）"""
    try:
        from src.config.settings import settings
        cfg = settings.get("logging")
        if not cfg:
            return

        name = "MyDearElysia"
        level = getattr(logging, cfg.get("level", "INFO").upper(), logging.INFO)
        file_level = getattr(logging, cfg.get("file_level", "DEBUG").upper(), logging.DEBUG)
        max_bytes = cfg.get("max_bytes", 5 * 1024 * 1024)
        backup_count = cfg.get("backup_count", 3)

        logger = logging.getLogger(name)
        logger.setLevel(min(level, file_level))
        logger.propagate = False
        logger.handlers.clear()

        # 控制台
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(
            _build_formatter(
                cfg.get("console_format", "[%(asctime)s] %(levelname)-5s | %(module_name)-12s | %(message)s"),
                datefmt=cfg.get("console_datefmt", "%H:%M:%S"),
            )
        )
        logger.addHandler(console)

        # 文件（带轮转）
        log_dir = Path(cfg.get("dir", "logs"))
        log_dir.mkdir(exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / cfg.get("filename", "app.log"),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(
            _build_formatter(cfg.get("file_format", "[%(asctime)s] %(levelname)-8s | %(module_name)-20s | %(message)s"))
        )
        logger.addHandler(file_handler)
    except Exception:
        pass  # 配置失败时保持默认


logger = setup_logger()
