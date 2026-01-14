from datetime import datetime
from omegaconf import DictConfig
from src.core.service import BaseService
from src.core.schemas import (
    IndexRequest,
    IndexConfig,
    IndexJobResponse,
    MetaResponse,
    DeleteResponse,
)
from src.enrichment.loader import LoaderConnecter
from src.enrichment.parser import RepoParser
from src.core.embedder import EmbeddingModel
import uuid


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

    async def run_indexing_pipeline(
        self, request: IndexRequest, config: IndexConfig
    ) -> IndexJobResponse:
        """
        Основной метод, запускающий пайплайн обработки репозитория.
        В микросервисной архитектуре этот метод вызывается внутри Worker-процесса.
        """
        msg = (
            "Starting indexing job: {request.meta.request_id} for "
            "repo: {request.repo_url}"
        )
        self.logger.info(msg)

        start_time = datetime.now()
        index_response = request

        index_response = await self.loader.clone_repository(request)
        if index_response.meta.status == "error":
            return self._finalize_response(index_response, start_time)

        if self.loader.is_repo_indexed(index_response.repo_url):
            self.logger.info(
                f"Repo {index_response.repo_url} already indexed. Skipping indexing."
            )
            index_response.job_status.description_error = (
                "Repository already indexed. Skipping indexing."
            )
            return self._finalize_response(index_response, start_time)

        index_response, chunks = self.parser.pipeline(config, index_response)

        index_response, vectors = await self.vectorizer.vectorize(
            chunks, index_response
        )
        if index_response.meta.status == "error":
            return self._finalize_response(index_response, start_time)

        index_response = await self.loader.save_vectors(vectors, index_response)
        if index_response.meta.status == "error":
            return self._finalize_response(index_response, start_time)

        return self._finalize_response(index_response, start_time)

    def _finalize_response(
        self, response: IndexJobResponse, start_time: datetime
    ) -> IndexJobResponse:
        """
        Вспомогательный метод для обновления метаданных перед возвратом ответа.
        Гарантирует, что request_id совпадает и проставляет время выполнения.
        """
        end_time = datetime.now()

        response.meta.start_datetime = start_time
        response.meta.end_datetime = end_time
        response.meta.status = (
            "done" if not response.meta.status else response.meta.status
        )

        self.logger.info(
            f"Job {response.meta.request_id} completed. "
            f"Status: {response.job_status.status}. "
            f"Duration: {(end_time - start_time).total_seconds():.2f}s"
        )
        return response

    async def delete_repo_index(
        self, repo_url: str, request_id: uuid.UUID = None
    ) -> DeleteResponse:
        """
        Удаляет индекс репозитория из векторной базы данных.

        Args:
            repo_url: URL репозитория для удаления
            request_id: Опциональный ID запроса (если не указан, будет сгенерирован)

        Returns:
            DeleteResponse с результатом операции
        """
        if request_id is None:
            request_id = uuid.uuid4()

        start_time = datetime.now()
        self.logger.info(f"Starting deletion job: {request_id} for repo: {repo_url}")

        try:
            success = self.loader.delete_repo_vectors(repo_url)
            end_time = datetime.now()

            message = (
                f"Successfully deleted vectors for repository {repo_url}"
                if success
                else f"Failed to delete vectors for repository {repo_url}"
            )

            response = DeleteResponse(
                repo_url=repo_url,
                success=success,
                meta=MetaResponse(
                    request_id=request_id,
                    start_datetime=start_time,
                    end_datetime=end_time,
                    status="done" if success else "error",
                ),
                message=message,
            )

            self.logger.info(
                f"Deletion job {request_id} completed. "
                f"Success: {success}. "
                f"Duration: {(end_time - start_time).total_seconds():.2f}s"
            )

            return response

        except Exception as e:
            end_time = datetime.now()
            self.logger.error(f"Error in deletion job {request_id}: {e}", exc_info=True)

            return DeleteResponse(
                repo_url=repo_url,
                success=False,
                meta=MetaResponse(
                    request_id=request_id,
                    start_datetime=start_time,
                    end_datetime=end_time,
                    status="error",
                ),
                message=f"Error deleting repository index: {str(e)}",
            )
