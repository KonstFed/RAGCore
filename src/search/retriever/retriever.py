import re
from typing import Any, Dict, List, Tuple
from src.core.db import VectorDBClient
from src.core.schema import (
    QueryRequest,
    Chunk,
    FilterNode
)


class Retriever:
    """
    Класс поиска чанков (Retrieval) и расширения контекста (Expansion).
    """
    def __init__(self):
        pass # TODO реализовать коннект к VectorDBClient

    def retrieval(self, request: QueryRequest) -> QueryRequest:
        """
        Выполняет поиск релевантных чанков в базе данных.
        """
        if request.query.sources is None:
            request.query.sources = []

        if not request.search_config or not request.search_config.retriever:
            return request

        config = request.search_config.retriever
        query_text = request.query.messages[-1].content

        # TODO поход в VectorDBClient

        if request.search_config.filtering and request.search_config.filtering.enabled:
            request.query.sources = self._apply_filtering(
                request.query.sources,
                request.search_config.filtering.filter
            )

        return request

    def expansion(self, request: QueryRequest) -> QueryRequest:
        """
        Расширяет найденные чанки (добавляет строки кода до и после).
        """
        if not request.search_config or \
           not request.search_config.context_expansion or \
           not request.search_config.context_expansion.enabled:
            return request

        config = request.search_config.context_expansion
        if not request.query.sources:
            return request

        # TODO реализовать запрос в VectorDBClient на подтягивание соседних чанков

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
