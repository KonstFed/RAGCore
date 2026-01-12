from abc import ABC
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
import os
from src.utils.logger import LoggerSetup, get_logger


class BaseService(ABC):
    """
    Базовый класс сервиса, отвечающий за инициализацию конфигурации.
    """

    def __init__(self, config_path: str = "config/deployment_config.yaml"):
        config_dir = os.path.dirname(config_path) or "configs"
        logging_conf_path = os.path.join(config_dir, "logging.yaml")
        LoggerSetup.setup(logging_conf_path)

        self.logger = get_logger(self.__class__.__name__)

        load_dotenv(override=True)
        self._cfg = self._load_config(config_path)

    def _load_config(self, config_path: str) -> DictConfig:
        """
        Загружает конфигурацию с помощью OmegaConf.
        С приоритетом: CLI args > Environment Vars > Config File > Defaults
        """
        self.logger.debug(f"Loading configuration from {config_path}")
        if not os.path.exists(config_path):
            config_path = os.path.join(os.getcwd(), config_path)

        if os.path.exists(config_path):
            file_conf = OmegaConf.load(config_path)
        else:
            self.logger.warning("Config file not found, using defaults.")
            file_conf = OmegaConf.create()

        prefix = "RAG_"
        env_vars_list = []
        for key, value in os.environ.items():
            if key.startswith(prefix):
                clean_key = key[len(prefix) :]
                conf_key = clean_key.replace("__", ".").lower()
                env_vars_list.append(f"{conf_key}={value}")
        env_conf = OmegaConf.from_dotlist(env_vars_list)

        config = OmegaConf.merge(file_conf, env_conf)
        OmegaConf.set_readonly(config, True)
        return config

    @property
    def config(self) -> DictConfig:
        return self._cfg
