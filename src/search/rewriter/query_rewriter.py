from typing import Any, Dict, List, Tuple
from omegaconf import DictConfig
from src.core.schemas import QueryRequest, SearchConfig
from src.core.llm import LLMClient
from src.utils.logger import get_logger


class QueryRewriter:
    """
    Класс переписывания запроса для улучшения качества поиска.
    Использует LLM для переформулирования.
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.logger = get_logger(self.__class__.__name__)
        pass # TODO инициализация и коннект к LLMClient

    async def pipeline(self, request: QueryRequest, config: SearchConfig) -> QueryRequest:
        self.logger.info(f"Run QueryRewriter pipeline for request_id={request.meta.request_id}.")
        if not config or \
           not config.query_rewriter or \
           not config.query_rewriter.enabled:
            return request

        config = config.query_rewriter

        original_query = request.query.messages[-1].content

        # TODO вызов LLM для переформулировки
        self.logger.info(f"Successful finished QueryRewriter pipeline for request_id={request.meta.request_id}.")
        return request
