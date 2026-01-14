from src.enrichment.data_enrichment import DataEnrichment
from src.search.search_engine import SearchEngine

# TODO: from src.agent import CodeAgent
from src.core.schemas import (
    IndexRequest,
    IndexJobResponse,
    IndexConfig,
    QueryRequest,
    QueryResponse,
    SearchConfig,
    DeleteResponse,
)
from typing import Any, Dict
from src.utils.logger import get_logger
from src.utils.github import resolve_full_github_url
from functools import lru_cache


@lru_cache(maxsize=512)
def _cached_url_resolver(url: str) -> tuple[str, str, str, str]:
    return resolve_full_github_url(url)


class Assistant:
    def __init__(self, service_cfg_path: str = "configs/deployment_config.yaml"):
        self.logger = get_logger(self.__class__.__name__)

        self.enrichment = DataEnrichment(service_cfg_path)
        self.searcher = SearchEngine(service_cfg_path)
        # self.agent = CodeAgent(service_cfg_path)

    async def index(
        self, request: Dict[str, Any], config: Dict[str, Any]
    ) -> IndexJobResponse:
        "Функция индексации репозитория с GitHub."
        response = await self.enrichment.run_indexing_pipeline(
            IndexRequest(**request), IndexConfig(**config)
        )
        return response

    async def query(
        self, request: Dict[str, Any], config: Dict[str, Any]
    ) -> QueryResponse:
        "Функция генерации ответа на вопрос пользователя."
        _, _, base_url, commit_hash = _cached_url_resolver(request["repo_url"])
        url = f"{base_url}/tree/{commit_hash}"
        request["repo_url"] = url
        response = await self.searcher.predict(
            QueryRequest(**request), SearchConfig(**config)
        )
        return response

    async def delete_index(self, request: Dict[str, Any]) -> DeleteResponse:
        "Функция удаления индекса репозитория."
        index_request = IndexRequest(**request)
        index_response = await self.enrichment.loader.clone_repository(index_request)
        if index_response.meta.status == "error":
            return DeleteResponse(
                repo_url=index_request.repo_url,
                success=False,
                meta=index_response.meta,
                message=index_response.job_status.description_error,
            )
        response = await self.enrichment.delete_repo_index(index_response.repo_url)
        return response
