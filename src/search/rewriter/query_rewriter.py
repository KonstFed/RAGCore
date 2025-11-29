from typing import Any, Dict, List, Tuple
from src.core.schema import QueryRequest
from src.core.llm import LLMClient


class QueryRewriter:
    """
    Класс переписывания запроса для улучшения качества поиска.
    Использует LLM для переформулирования.
    """
    def __init__():
        pass # TODO инициализация и коннект к LLMClient

    def pipeline(self, request: QueryRequest) -> QueryRequest:
        if not request.search_config or \
           not request.search_config.query_rewriter or \
           not request.search_config.query_rewriter.enabled:
            return request

        config = request.search_config.query_rewriter

        original_query = request.query.messages[-1].content

        # TODO вызов LLM для переформулировки

        return request
