import os
import shutil
import tempfile
import uuid
import git
from datetime import datetime
from omegaconf import DictConfig
from src.core.db import VectorDBClient
from src.core.schemas import (
    IndexRequest,
    IndexJobResponse,
    IndexJobStatus,
    MetaResponse
)
from typing import List, Dict, Any
from src.utils.logger import get_logger


class LoaderConnecter:
    """
    Отвечает за взаимодействие с внешними источниками данных (Git)
    и хранилищами (Vector DB).
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.download_path = cfg.paths.temp_repo_storage
        self.collection_name = cfg.database.collection_name
        self.vector_db_client = VectorDBClient(cfg)
        self.createdir(self.download_path)

    async def clone_repository(self, request: IndexRequest) -> IndexJobResponse:
        """
        Клонирует репозиторий во временную директорию.
        Возвращает путь к склонированной папке.
        """
        repo_url = str(request.repo_url)
        branch = request.branch

        temp_dir = tempfile.mkdtemp(prefix=f"repo_{request.meta.request_id}_", dir=self.download_path)

        self.logger.info(f"Cloning {repo_url} (branch: {branch}) to {temp_dir} for request_id={request.meta.request_id}.")

        try:
            git.Repo.clone_from(
                url=repo_url,
                to_path=temp_dir,
                branch=branch,
                depth=1
            )
            self.logger.info(f"Successful cloning {repo_url} (branch: {branch}) to {temp_dir}")
            return IndexJobResponse(
                meta=MetaResponse(
                    request_id=request.meta.request_id,
                    start_datetime=datetime.now(), # будет перезаписано
                    end_datetime=datetime.now(), # будет перезаписано
                    status="done"
                ),
                repo_url=request.repo_url,
                job_status=IndexJobStatus(
                    status="loaded",
                    chunks_processed=0,
                    repo_path=temp_dir
                )
            )
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.error(f"Failed to clone repository: {e} for request_id={request.meta.request_id}.")
            return IndexJobResponse(
                meta=MetaResponse(
                    request_id=request.meta.request_id,
                    start_datetime=datetime.now(), # будет перезаписано
                    end_datetime=datetime.now(), # будет перезаписано
                    status="error"
                ),
                repo_url=request.repo_url,
                job_status=IndexJobStatus(
                    status="failed",
                    chunks_processed=0,
                    repo_path=temp_dir,
                    description_error=str(e)
                )
            )

    async def save_vectors(
        self,
        vectors: List[Dict[str, Any]],
        index_job_response: IndexJobResponse
    ) -> IndexJobResponse:
        """
        Сохраняет вектора и метаданные в векторную БД.
        Принимает список словарей {id, vector, payload}.
        """
        if not vectors:
            return self._error_response(index_job_response, "No vectors to save into QDrant.")

        collection_name = self.collection_name
        self.logger.info(
            f"Start saving {len(vectors)} vectors into QDrant collection '{collection_name}'"
            f"for request_id={index_job_response.meta.request_id}."
        )

        try:
            collections_response = self.vector_db_client.get_collections()
            existing_collections = []
            if "result" in collections_response and "collections" in collections_response["result"]:
                existing_collections = [
                    col["name"] for col in collections_response["result"]["collections"]
                ]

            if collection_name not in existing_collections:
                self.logger.info(f"Collection '{collection_name}' does not exist. Creating...")

                create_response = self.vector_db_client.create_collection(collection_name)
                if create_response.get("status") != "ok":
                    return self._error_response(
                        index_job_response, f"Failed to create collection: {create_response}."
                    )

                self.logger.info(f"Setting up payload indexes for '{collection_name}'...")
                self.vector_db_client._setup_collection_indexes(collection_name)

                self.logger.info(
                    f"Collection '{collection_name}' created successfully"
                    f"for request_id={index_job_response.meta.request_id}."
                )

            upsert_response = self.vector_db_client.add_vectors(collection_name, vectors)

            if upsert_response.get("status") == "ok":
                return self._success_response(
                    index_job_response,
                    f"Successfully saved {len(vectors)} vectors"
                )
            else:
                return self._error_response(
                    index_job_response,
                    f"Database returned non-ok status: {upsert_response}"
                )

        except Exception as e:
            return self._error_response(index_job_response, f"Error while saving vectors to QDrant: {e}")

    def createdir(self, directory: str) -> None:
        """Метод созданий вспомогательной директории."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def cleanup(self, path: str) -> None:
        """Метод для ручной очистки, если требуется"""
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    def _error_response(self, index_job_response: IndexJobResponse, error_message: str) -> IndexJobResponse:
        self.logger.error(f"{error_message} for request_id={index_job_response.meta.request_id}.")
        index_job_response.meta.status = "error"
        index_job_response.job_status.status = "saved_to_qdrant"
        index_job_response.job_status.description_error = error_message
        return index_job_response

    def _success_response(self, index_job_response: IndexJobResponse, success_message: str) -> IndexJobResponse:
        self.logger.info(f"{success_message} for request_id={index_job_response.meta.request_id}.")
        index_job_response.meta.status = "done"
        index_job_response.job_status.status = "saved_to_qdrant"
        return index_job_response
