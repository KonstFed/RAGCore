import os

from src.core.service import BaseService
from src.core.schemas import IndexRequest, IndexJobResponse, Chunk
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

        self.loader = self._init_loader()
        self.parser = self._init_parser()
        self.vectorizer = self._init_vectorizer()

        self.logger.info("DataEnrichment service initialized.")

    def _init_loader(self) -> LoaderConnecter:
        return LoaderConnecter

    def _init_parser(self) -> RepoParser:
        return RepoParser

    def _init_vectorizer(self) -> EmbeddingModel:
        return EmbeddingModel

    async def run_indexing_pipeline(self, request: IndexRequest) -> IndexJobResponse:
        """
        Основной метод, запускающий пайплайн обработки репозитория.
        В микросервисной архитектуре этот метод вызывается внутри Worker-процесса.
        """
        self.logger.info(f"Starting indexing job: {request.meta.request_id} for repo: {request.repo_url}")

        response = {
            "job_id": request.meta.request_id,
            "status": "queued"
        }

        try:
            repo_path = self.loader.clone_repository(request)

            chunks = self.parser.pipeline(repo_path, request)

            vectors = await self.vectorizer.vectorize(chunks, request)

            response = await self.loader.save_vectors(vectors)

            self.logger.info(f"Job {request.meta.request_id} completed successfully.")

            response = {
                "job_id": request.meta.request_id,
                "status": "processing"
            }
        except Exception as e:
            self.logger.exception(f"Critical error in job {request.meta.request_id}")

        return IndexJobResponse(**response)
