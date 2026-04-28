"""
common/config.py
统一日志单例 + 通用参数解析基础
"""
import logging
import os


class LoggerSingleton:
    """全局唯一 Logger（单例），所有任务共用。"""
    _instance = None
    _logger = None

    def __new__(cls, name: str = "CardiacAI", log_file: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)
            fmt = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            # 控制台 handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            logger.addHandler(ch)
            # 文件 handler（可选）
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                fh = logging.FileHandler(log_file)
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(fmt)
                logger.addHandler(fh)
            cls._logger = logger
        return cls._instance

    # 转发常用方法
    def info(self, msg):    self.__class__._logger.info(msg)
    def debug(self, msg):   self.__class__._logger.debug(msg)
    def warning(self, msg): self.__class__._logger.warning(msg)
    def error(self, msg):   self.__class__._logger.error(msg)
