import os
from src.enrichment.data_enrichment import DataEnrichment
from src.search.search_engine import SearchEngine
# TODO: from src.agent import CodeAgent
from src.core.schemas import (
    IndexRequest,
    IndexJobResponse,
    QueryRequest,
    QueryResponse
)
from typing import Any, Dict, List, Tuple
from src.utils.validator import APISchemaValidator
from src.utils.logger import get_logger


class Assistant:
    def __init__(
        self,
        service_cfg_path: str = "configs/deployment_config.yaml",
        swagger_path: str = "api/api.yaml"
    ):
        self.logger = get_logger(self.__class__.__name__)

        self.enrichment = DataEnrichment(service_cfg_path)
        self.searcher = SearchEngine(service_cfg_path)
        # self.agent = CodeAgent(service_cfg_path)

        self.validator = APISchemaValidator(swagger_path, self.logger)  # TODO удалить, решили использовать pydantic

    async def index(self, request: Dict[str, Any]) -> IndexJobResponse:
        "Функция индексации репозитория с GitHub."
        self.validator.validate_data(request, "IndexRequest") # TODO удалить, решили использовать pydantic
        response = await self.enrichment.run_indexing_pipeline(IndexRequest(**request))
        return response

    async def query(self, request: Dict[str, Any]) -> QueryResponse:
        "Функция генерации ответа на вопрос пользователя."
        self.validator.validate_data(request, "QueryRequest")  # TODO удалить, решили использовать pydantic
        response = await self.searcher.predict(QueryRequest(**request))
        return response

