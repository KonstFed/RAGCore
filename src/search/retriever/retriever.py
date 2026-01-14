from typing import Any, Dict, List, Literal, Union

from omegaconf import DictConfig

from src.core.db import VectorDBClient
from src.core.embedder import EmbeddingModel
from src.core.schemas import (
    Chunk,
    ChunkMetadata,
    FilterCondition,
    FilterGroup,
    FilterNode,
    QueryRequest,
    SearchConfig,
)
from src.utils.logger import get_logger


class Retriever:
    """
    Класс поиска чанков (Retrieval) и расширения контекста (Expansion).
    """

    def __init__(self, cfg: DictConfig):
        self.logger = get_logger(self.__class__.__name__)
        self.vector_db = VectorDBClient(cfg)
        self.embedder = EmbeddingModel(cfg)
        self.collection_name = cfg.database.collection_name
        # NOTE: храним разные репозитории в одной коллекции и фильтруем по repo_url.

    def retrieval(self, request: QueryRequest, config: SearchConfig) -> QueryRequest:
        """
        Выполняет поиск релевантных чанков в базе данных.
        """
        self.logger.info(
            f"Run retriever search for request_id={request.meta.request_id}."
        )

        if request.query.sources is None:
            request.query.sources = []

        if not config or not config.retriever:
            return request

        retriever_config = config.retriever
        query_text = request.query.messages[-1].content

        query_vector = self.embedder.embed_query([query_text])[0]

        # Always scope search to the requested repo_url
        # (otherwise sources may come from other repos in the same collection)
        must_conditions: List[Dict[str, Any]] = [
            {"key": "repo_url", "match": {"value": str(request.repo_url)}}
        ]

        user_filter = None
        if config.filtering and config.filtering.enabled and config.filtering.filter:
            user_filter = self._convert_to_qdrant_filter(config.filtering.filter)
            self.logger.debug(f"User QDrant filter (raw): {user_filter}")

        qdrant_filter: Dict[str, Any] = {"must": must_conditions}
        if user_filter:
            # If user_filter already looks like a root filter, merge it;
            # else treat it as a single condition.
            if any(k in user_filter for k in ("must", "should", "must_not")):
                if isinstance(user_filter.get("must"), list):
                    qdrant_filter["must"].extend(user_filter["must"])
                elif user_filter.get("must") is not None:
                    qdrant_filter["must"].append(user_filter["must"])

                if isinstance(user_filter.get("should"), list):
                    qdrant_filter["should"] = user_filter["should"]

                if isinstance(user_filter.get("must_not"), list):
                    qdrant_filter["must_not"] = user_filter["must_not"]
            else:
                qdrant_filter["must"].append(user_filter)

        collection_name = self.collection_name

        try:
            search_result = self.vector_db.search(
                collection_name=collection_name,
                vector=query_vector,
                top_k=retriever_config.size,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        except Exception as e:
            self.logger.error(f"Error during QDrant search: {e}")
            return request

        found_chunks = []
        if "result" in search_result:
            if len(search_result["result"]) == 0:
                self.logger.warning(
                    f"No chunks found for request_id={request.meta.request_id} in repo {request.repo_url}."
                )
                return request

            for item in search_result["result"]:
                score = item.get("score", 0.0)
                payload = item.get("payload", {})

                # Optional score thresholding (keeps obvious noise out)
                # if retriever_config.threshold and score < retriever_config.threshold:
                #     continue

                try:
                    meta = ChunkMetadata(
                        chunk_id=item.get("id"),  # или payload.get("chunk_id")
                        filepath=payload.get("filepath"),
                        repo_url=payload.get("repo_url"),
                        start_line_no=payload.get("start_line_no"),
                        end_line_no=payload.get("end_line_no"),
                        language=payload.get("language"),
                        chunk_size=payload.get("chunk_size"),
                        line_count=payload.get("line_count"),
                        node_count=payload.get("node_count"),
                    )

                    chunk = Chunk(
                        content=payload.get("content", ""),
                        metadata=meta,
                        retrieval_relevance_score=score,
                    )
                    found_chunks.append(chunk)
                except Exception as parse_e:
                    self.logger.info(
                        f"Failed to parse chunk from DB response: {parse_e}"
                        f"for request_id={request.meta.request_id}."
                    )

        request.query.sources = found_chunks
        self.logger.info(
            f"Successful finished retriever search with {len(found_chunks)} chunks "
            f"for request_id={request.meta.request_id}."
        )
        return request

    def expansion(self, request: QueryRequest, config: SearchConfig) -> QueryRequest:
        """
        Расширяет найденные чанки (добавляет строки кода до и после).
        """
        if (
            not config
            or not config.context_expansion
            or not config.context_expansion.enabled
        ):
            return request

        config = config.context_expansion
        if not request.query.sources:
            return request

        self.logger.info(
            f"Run context expansion for request_id={request.meta.request_id}."
        )

        expanded_sources = []

        for chunk in request.query.sources:
            expanded_sources.append(chunk)

            if config.before_chunk > 0:
                prev_chunks = self._fetch_neighbors(
                    chunk=chunk, direction="before", count=config.before_chunk
                )
                for pc in reversed(prev_chunks):
                    expanded_sources.insert(expanded_sources.index(chunk), pc)

            if config.after_chunk > 0:
                next_chunks = self._fetch_neighbors(
                    chunk=chunk, direction="after", count=config.after_chunk
                )
                expanded_sources.extend(next_chunks)

        unique_sources = self._deduplicate_chunks(expanded_sources)
        request.query.sources = unique_sources

        self.logger.info(
            f"Successful finished context expansion."
            f"Total chunks: {len(request.query.sources)}"
        )
        return request

    def _fetch_neighbors(
        self, chunk: Chunk, direction: Literal["before", "after"], count: int
    ) -> List[Chunk]:
        """
        Ищет соседние чанки в том же файле.
        """
        file_filter = {"key": "filepath", "match": {"value": chunk.metadata.filepath}}
        repo_filter = (
            {"key": "repo_url", "match": {"value": chunk.metadata.repo_url}}
            if chunk.metadata.repo_url
            else None
        )

        range_condition = {}

        if direction == "before":
            # Ищем чанки, где end_line_no < текущего start_line_no
            # Сортируем по убыванию (desc), чтобы взять ближайший сверху
            range_condition = {
                "key": "end_line_no",
                "range": {"lt": chunk.metadata.start_line_no},
            }
            # QDrant scroll не поддерживает сортировку напрямую так, как search.
            # Но мы можем использовать фильтр.
            # Для точного порядка "ближайший сосед"
            # в QDrant нужен order_by (доступен в v1.10+).
            # Если версия старая, можно запросить больше и отсортировать в коде.
        else:
            range_condition = {
                "key": "start_line_no",
                "range": {"gt": chunk.metadata.end_line_no},
            }

        must_conditions = [file_filter, range_condition]
        if repo_filter:
            must_conditions.insert(0, repo_filter)
        qdrant_filter = {"must": must_conditions}

        try:
            result = self.vector_db.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=count + 2,
            )
        except Exception as e:
            self.logger.warning(
                f"Error when call QDrant scroll for expand chunks: {e}."
            )
            return []

        neighbors = []
        if "result" in result and "points" in result["result"]:
            points = result["result"]["points"]

            for pt in points:
                payload = pt.get("payload", {})
                try:
                    meta = ChunkMetadata(
                        chunk_id=pt.get("id"),
                        filepath=payload.get("filepath"),
                        start_line_no=payload.get("start_line_no"),
                        end_line_no=payload.get("end_line_no"),
                        language=payload.get("language"),
                        chunk_size=payload.get("chunk_size"),
                    )
                    neighbors.append(
                        Chunk(content=payload.get("content", ""), metadata=meta)
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Error when parse chunk from QDrant scroll: {e}."
                    )
                    continue

        # Локальная сортировка, так как Scroll не гарантирует порядок по start_line_no
        if direction == "before":
            neighbors.sort(key=lambda x: x.metadata.end_line_no, reverse=True)
        else:
            neighbors.sort(key=lambda x: x.metadata.start_line_no)

        return neighbors[:count]

    def _deduplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Удаляет дубликаты чанков по ID."""
        seen = set()
        unique = []
        for c in chunks:
            cid = str(c.metadata.chunk_id)
            if cid not in seen:
                seen.add(cid)
                unique.append(c)
        return unique

    def _convert_to_qdrant_filter(
        self, node: Union[FilterNode, FilterGroup, FilterCondition]
    ) -> Dict[str, Any]:
        """
        Рекурсивно преобразует FilterNode в структуру QDrant Filter.
        """
        if isinstance(node, FilterGroup) or (
            hasattr(node, "operator") and node.operator in ["and", "or"]
        ):
            clauses = [self._convert_to_qdrant_filter(child) for child in node.values]

            if node.operator == "and":
                return {"must": clauses}
            elif node.operator == "or":
                return {"should": clauses}

        if isinstance(node, FilterCondition) or (
            hasattr(node, "name") and hasattr(node, "value")
        ):
            key = node.name
            val = node.value
            op = node.operator

            if op == "eq":
                return {"key": key, "match": {"value": val}}
            elif op == "neq":
                return {"must_not": [{"key": key, "match": {"value": val}}]}
            elif op == "in":
                return {
                    "key": key,
                    "match": {"any": val if isinstance(val, list) else [val]},
                }
            elif op in ["gt", "gte", "lt", "lte"]:
                range_op = {"gt": "gt", "gte": "gte", "lt": "lt", "lte": "lte"}.get(op)
                return {"key": key, "range": {range_op: val}}
            elif op == "contains":
                return {"key": key, "match": {"text": str(val)}}
            elif op == "wildcard":
                return {"key": key, "match": {"value": val}}

        return {}
