import random
from src.core.schemas import QueryRequest, SearchConfig
from typing import Any, Dict, List, Tuple
from src.utils.logger import get_logger


class Reranker:
    """
    Класс переранжирования (Reranking).
    Сортирует найденные чанки по релевантности с помощью Cross-Encoder модели.
    """
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        pass # TODO реализовать иницализацию и подключение к модели реранкера

    async def pipeline(self, request: QueryRequest, config: SearchConfig) -> QueryRequest:
        self.logger.info("Run reranker pipeline.")
        if not request.query.sources:
            return request

        if not config or \
           not config.reranker or \
           not config.reranker.enabled:
            return request

        config = config.reranker
        query = request.query.messages[-1].content

        # TODO реализовать поход в reranker модель

        for chunk in request.query.sources:
            if chunk.reranker_relevance_score is None:
                chunk.reranker_relevance_score = random.random()

        # Сортировка по score (по убыванию)
        request.query.sources.sort(
            key=lambda x: x.reranker_relevance_score or 0.0, 
            reverse=True
        )

        # Фильтрация по порогу
        request.query.sources = [
            c for c in request.query.sources 
            if (c.reranker_relevance_score or 0.0) >= config.threshold
        ]

        # Обрезка по Top-K
        request.query.sources = request.query.sources[:config.top_k]

        self.logger.info("Successful finished reranker pipeline.")
        return request
