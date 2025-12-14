import re
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import DictConfig
from src.core.db import VectorDBClient
from src.core.schemas import (
    QueryRequest,
    SearchConfig,
    Chunk,
    FilterNode
)
from src.utils.logger import get_logger


class Retriever:
    """
    Класс поиска чанков (Retrieval) и расширения контекста (Expansion).
    """
    def __init__(self, cfg: DictConfig):
        self.logger = get_logger(self.__class__.__name__)
        pass # TODO реализовать коннект к VectorDBClient

    def retrieval(self, request: QueryRequest, config: SearchConfig) -> QueryRequest:
        """
        Выполняет поиск релевантных чанков в базе данных.
        """
        self.logger.info("Run retriever search.")
        if request.query.sources is None:
            request.query.sources = []

        if not config or not config.retriever:
            return request

        retriever_config = config.retriever
        query_text = request.query.messages[-1].content

        # TODO поход в VectorDBClient

        if config.filtering and config.filtering.enabled:
            request.query.sources = self._apply_filtering(
                request.query.sources,
                config.filtering.filter
            )

        self.logger.info("Successful finished retriever search.")
        return request

    def expansion(self, request: QueryRequest, config: SearchConfig) -> QueryRequest:
        """
        Расширяет найденные чанки (добавляет строки кода до и после).
        """
        self.logger.info("Run context expansion.")
        if not config or \
           not config.context_expansion or \
           not config.context_expansion.enabled:
            return request

        config = config.context_expansion
        if not request.query.sources:
            return request

        # TODO реализовать запрос в VectorDBClient на подтягивание соседних чанков
        self.logger.info("Successful finished context expansion.")
        return request

    def _apply_filtering(self, chunks: List[Chunk], filter_node: Optional[FilterNode]) -> List[Chunk]:
        if not filter_node or not chunks:
            return chunks

        filtered_chunks = []
        for chunk in chunks:
            if self._evaluate_filter(chunk, filter_node):
                filtered_chunks.append(chunk)
        return filtered_chunks

    def _evaluate_filter(self, chunk: Chunk, node: FilterNode) -> bool:

        if hasattr(node, 'operator') and hasattr(node, 'values') and isinstance(node.values, list):
            results = [self._evaluate_filter(chunk, sub_node) for sub_node in node.values]
            if node.operator == "and":
                return all(results)
            elif node.operator == "or":
                return any(results)
            return False

        elif hasattr(node, 'name') and hasattr(node, 'value'):
            chunk_val = getattr(chunk.metadata, node.name, None)
            if chunk_val is None:
                return False
            return self._compare(chunk_val, node.operator, node.value)

        return True

    def _compare(self, actual, op, expected) -> bool:
        if op == "eq": return actual == expected
        if op == "neq": return actual != expected
        if op == "gt": return actual > expected
        if op == "gte": return actual >= expected
        if op == "lt": return actual < expected
        if op == "lte": return actual <= expected
        if op == "in": return actual in expected
        if op == "contains": return expected in str(actual)
        if op == "wildcard": return re.match(str(expected).replace("*", ".*"), str(actual))
        return False
