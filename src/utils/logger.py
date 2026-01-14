import os
import yaml
import logging
import logging.config


class LoggerSetup:
    """
    Утилита для настройки логгера из YAML файла.
    """

    @staticmethod
    def setup(config_path: str = "config/logging.yaml", default_level=logging.INFO):
        """
        Загружает конфиг логгера. Если файла нет, использует базовую настройку.
        """
        if os.path.exists(config_path):
            with open(config_path, "rt") as f:
                try:
                    config = yaml.safe_load(f.read())
                    logging.config.dictConfig(config)
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Logging configured via {config_path}")
                    return
                except Exception as e:
                    print(
                        "Error in Logging Configuration. "
                        f"Using default configs. Error: {e}"
                    )
        else:
            print(
                f"Logging config file not found at {config_path}. Using basic config."
            )

        logging.basicConfig(level=default_level)


def get_logger(name: str) -> logging.Logger:
    """
    Функция для импорта в других файлах:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    """
    # Используем префикс RAGService, чтобы настройки из yaml применялись к этому логгеру
    return logging.getLogger(f"RAGService.{name}")
