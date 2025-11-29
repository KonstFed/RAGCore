from abc import ABC
from omegaconf import OmegaConf, DictConfig
import os
from src.utils.logger import LoggerSetup, get_logger
from src.utils.validator import APISchemaValidator
from typing import Any, Dict, List, Tuple


class BaseService(ABC):
    """
    Базовый класс сервиса, отвечающий за инициализацию конфигурации.
    """
    def __init__(self, config_path: str = "config/deployment_config.yaml"):
        config_dir = os.path.dirname(config_path) or "configs"
        logging_conf_path = os.path.join(config_dir, "logging.yaml")
        LoggerSetup.setup(logging_conf_path)

        self.logger = get_logger(self.__class__.__name__)

        self._cfg = self._load_config(config_path)

        # TODO удалить, решили использовать pydantic
        validator_path = self._cfg.paths.openapi_spec
        if os.path.exists(validator_path):
            self.validator = APISchemaValidator(validator_path, self.logger)
            self.logger.info("Schema Validator initialized.")
        else:
            self.logger.warning(f"OpenAPI spec not found at {spec_path}. Validation disabled.")
            self.validator = None

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

        env_conf = OmegaConf.from_dotlist([
            f"{k}={v}" for k, v in os.environ.items() if k.startswith("RAG_")
        ])

        config = OmegaConf.merge(file_conf, env_conf)
        OmegaConf.set_readonly(config, True)
        return config

    def validate_input(self, data: Dict[str, Any], schema_name: str): # TODO удалить, решили использовать pydantic
        """Метод-хелпер для вызова валидации из дочерних классов"""
        if self.validator:
            self.validator.validate_data(data, schema_name)
        else:
            self.logger.debug(f"Skipping validation for {schema_name}")

    @property
    def config(self) -> DictConfig:
        return self._cfg
