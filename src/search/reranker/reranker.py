import random
from src.core.schema import QueryRequest
from typing import Any, Dict, List, Tuple


class Reranker:
    """
    Класс переранжирования (Reranking).
    Сортирует найденные чанки по релевантности с помощью Cross-Encoder модели.
    """
    def __init__(self):
        pass # TODO реализовать иницализацию и подключение к модели реранкера

    def pipeline(self, request: QueryRequest) -> QueryRequest:
        if not request.query.sources:
            return request

        if not request.search_config or \
           not request.search_config.reranker or \
           not request.search_config.reranker.enabled:
            return request

        config = request.search_config.reranker
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

        return request
