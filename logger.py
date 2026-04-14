import logging
import sys
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Форматтер с цветами для консоли"""
    
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # Формат: [ВРЕМЯ] УРОВЕНЬ: Сообщение
    format_str = "[%(asctime)s] %(levelname)-8s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)

def get_logger(name="CareerAgent"):
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # 1. Обработчик для консоли (с цветами)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(CustomFormatter())
        
        # 2. Обработчик для файла (простой текст)
        file_handler = logging.FileHandler(f"logs_session.log", encoding="utf-8")
        file_fmt = logging.Formatter("[%(asctime)s] %(name)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(file_fmt)

        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)
        
    return logger

# Создаем глобальный экземпляр для удобного импорта
logger = get_logger()