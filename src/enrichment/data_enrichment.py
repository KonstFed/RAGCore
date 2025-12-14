import os
from datetime import datetime
from omegaconf import DictConfig
from src.core.service import BaseService
from src.core.schemas import IndexRequest, IndexConfig, IndexJobResponse, Chunk
from src.enrichment.loader import LoaderConnecter
from src.enrichment.parser import RepoParser
from src.core.embedder import EmbeddingModel
from typing import Any, Dict, List, Tuple


class DataEnrichment(BaseService):
    """
    Класс отвечает за процесс индексации (`/index`).
    Пайплайн: Clone -> AST Chunking -> Embedding -> Vector DB Upsert.
    """
    def __init__(self, config_path: str = "configs/deployment_config.yaml"):
        super().__init__(config_path)

        self.loader = self._init_loader(self.config)
        self.parser = self._init_parser(self.config)
        self.vectorizer = self._init_vectorizer(self.config)

        self.logger.info("DataEnrichment service initialized.")

    def _init_loader(self, config: DictConfig) -> LoaderConnecter:
        return LoaderConnecter(config)

    def _init_parser(self, config: DictConfig) -> RepoParser:
        return RepoParser(config)

    def _init_vectorizer(self, config: DictConfig) -> EmbeddingModel:
        return EmbeddingModel(config)

    async def run_indexing_pipeline(self, request: IndexRequest, config: IndexConfig) -> IndexJobResponse:
        """
        Основной метод, запускающий пайплайн обработки репозитория.
        В микросервисной архитектуре этот метод вызывается внутри Worker-процесса.
        """
        self.logger.info(f"Starting indexing job: {request.meta.request_id} for repo: {request.repo_url}")

        start_time = datetime.now()
        index_response = request

        try:
            index_response = await self.loader.clone_repository(request)
            if index_response.meta.status == "error":
                self._finalize_response(index_response, start_time)

            index_response, chunks = self.parser.pipeline(config, index_response)

            index_response, vectors = await self.vectorizer.vectorize(chunks, config, index_response)
            if index_response.meta.status == "error":
                self._finalize_response(index_response, start_time)

            index_response = await self.loader.save_vectors(vectors, index_response)
            if index_response.meta.status == "error":
                self._finalize_response(index_response, start_time)

            return self._finalize_response(index_response, start_time)

        except Exception as e:
            self.logger.exception(f"Critical error in job {request.meta.request_id}")

        return IndexJobResponse(**response)

    def _finalize_response(
        self,
        response: IndexJobResponse,
        start_time: datetime
    ) -> IndexJobResponse:
        """
        Вспомогательный метод для обновления метаданных перед возвратом ответа.
        Гарантирует, что request_id совпадает и проставляет время выполнения.
        """
        end_time = datetime.now()

        response.meta.start_datetime = start_time
        response.meta.end_datetime = end_time
        response.meta.status = "done" if not response.meta.status else response.meta.status

        self.logger.info(
            f"Job {response.meta.request_id} completed. "
            f"Status: {response.job_status.status}. "
            f"Duration: {(end_time - start_time).total_seconds():.2f}s"
        )
        return response
