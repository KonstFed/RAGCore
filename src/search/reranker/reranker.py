import random
from datetime import datetime
from omegaconf import DictConfig
from src.core.schemas import QueryRequest, QueryResponse, SearchConfig, RerankerConfig
from typing import Any, Dict, List, Tuple, Union
from src.utils.logger import get_logger


class Reranker:
    """
    Класс переранжирования (Reranking).
    Сортирует найденные чанки по релевантности с помощью Cross-Encoder модели.
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = cfg.reranker.model_name
        self.url = cfg.reranker.url
        self.api_key = cfg.reranker.api_key
        self.top_k = cfg.reranker.top_k
        self.threshold = cfg.reranker.threshold
        self.fallback_message = cfg.reranker.fallback_message
        self.timeout = cfg.reranker.timeout

    async def pipeline(self, request: QueryRequest, config: SearchConfig) -> Union[QueryRequest, QueryResponse]:
        self.logger.info(f"Run reranker pipeline for request_id={request.meta.request_id}.")

        if not config or not config.reranker or not config.reranker.enabled:
            self.logger.warning(f"Finished reranker pipeline because no config or enabled=False for request_id={request.meta.request_id}.")
            return request

        if not request.query.sources:
            self.logger.warning(f"Finished reranker pipeline because no chunks for request_id={request.meta.request_id}.")
            return request

        config = config.reranker
        query = request.query.messages[-1].content
        documents_text = [chunk.content for chunk in request.query.sources]

        status_code, response_json = self._rerank(query_text, documents_text, config)
        if status_code != 200:
            self.logger.warning(
                f"Reranker API returned {status_code} for request_id={request.meta.request_id}. "
                f"Skipping rerank step."
            )
            return request

        filtered_sources = []

        if "results" in response_json:
            for item in response_json["results"]:
                idx = item["index"]
                score = item["relevance_score"]

                current_threshold = config.threshold if config.threshold is not None else self.threshold

                if score < current_threshold:
                    continue

                chunk = request.query.sources[idx]

                chunk.reranker_relevance_score = score
                filtered_sources.append(chunk)

        if not filtered_sources:
            response_dict = {
                "meta": {
                    "request_id": request.meta.request_id,
                    "start_datetime": datetime.now(), # будет перезаписано
                    "start_datetime": datetime.now(),
                    "status": "done"
                },
                "status": "preprocessor_filtering",
                "messages": request.query.messages,
                "answer": self.fallback_message,
                "sources": filtered_sources,
                "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }
            self.logger.warning(f"Reranker filtered out all the chunks for request_id={request.meta.request_id}.")
            return QueryResponse(**response_dict)

        request.query.sources = filtered_sources

        self.logger.info(
            f"Successful reranking. "
            f"kept {len(filtered_sources)}/{len(documents_text)} chunks "
            f"for request_id={request.meta.request_id}."
        )
        return request

    def _rerank(self, query: str, documents: List[str], config: RerankerConfig) -> Tuple[int, Dict[str, Any]]:
        """
        Отправляет запрос к Jina Reranker API.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": config.model_name or self.model_name,
            "query": query,
            "top_n": config.top_k or self.top_k,
            "documents": documents,
            "return_documents": False
        }
        try:
            response = requests.post(
                self.url,
                headers=headers,
                data=json.dumps(data),
                timeout=self.timeout
            )
            return response.status_code, response.json()
        except Exception as e:
            self.logger.error(f"Error during Reranker API call: {e}.")
            return 500, {}
