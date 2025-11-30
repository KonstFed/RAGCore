from typing import Any, Dict, List, Tuple
from src.core.schemas import QueryRequest, SearchConfig
from src.core.llm import LLMClient
from src.utils.logger import get_logger


class QueryRewriter:
    """
    Класс переписывания запроса для улучшения качества поиска.
    Использует LLM для переформулирования.
    """
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        pass # TODO инициализация и коннект к LLMClient

    async def pipeline(self, request: QueryRequest, config: SearchConfig) -> QueryRequest:
        self.logger.info("Run QueryRewriter pipeline.")
        if not config or \
           not config.query_rewriter or \
           not config.query_rewriter.enabled:
            return request

        config = config.query_rewriter

        original_query = request.query.messages[-1].content

        # TODO вызов LLM для переформулировки
        self.logger.info("Successful finished QueryRewriter pipeline.")
        return request
