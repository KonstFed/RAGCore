import os
from src.enrichment.data_enrichment import DataEnrichment
from src.search.search_engine import SearchEngine
# TODO: from src.agent import CodeAgent
from src.core.schemas import (
    IndexRequest,
    IndexJobResponse,
    IndexConfig,
    QueryRequest,
    QueryResponse,
    SearchConfig
)
from typing import Any, Dict, List, Tuple
from src.utils.logger import get_logger


class Assistant:
    def __init__(
        self,
        service_cfg_path: str = "configs/deployment_config.yaml"
    ):
        self.logger = get_logger(self.__class__.__name__)

        self.enrichment = DataEnrichment(service_cfg_path)
        self.searcher = SearchEngine(service_cfg_path)
        # self.agent = CodeAgent(service_cfg_path)

    async def index(self, request: Dict[str, Any], config: Dict[str, Any]) -> IndexJobResponse:
        "Функция индексации репозитория с GitHub."
        response = await self.enrichment.run_indexing_pipeline(
            IndexRequest(**request),
            IndexConfig(**config)
        )
        return response

    async def query(self, request: Dict[str, Any], config: Dict[str, Any]) -> QueryResponse:
        "Функция генерации ответа на вопрос пользователя."
        response = await self.searcher.predict(
            QueryRequest(**request),
            SearchConfig(**config)
        )
        return response

