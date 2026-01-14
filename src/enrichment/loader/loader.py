import os
import shutil
import tempfile
from datetime import datetime
from omegaconf import DictConfig
from src.core.db import VectorDBClient
from src.core.schemas import (
    IndexRequest,
    IndexJobResponse,
    IndexJobStatus,
    MetaResponse,
)
from typing import List, Dict, Any
from src.utils.logger import get_logger
from src.utils.github import resolve_full_github_url, download_github_archive


class LoaderConnecter:
    """
    Отвечает за взаимодействие с внешними источниками данных (Git)
    и хранилищами (Vector DB).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.download_path = cfg.paths.temp_repo_storage
        self.collection_name = cfg.database.collection_name
        self.batch_size = cfg.database.get("batch_size", 500)
        self.vector_db_client = VectorDBClient(cfg)
        self.createdir(self.download_path)

    async def clone_repository(self, request: IndexRequest) -> IndexJobResponse:
        """
        Клонирует репозиторий во временную директорию.
        Поддерживает URL вида:
        - https://github.com/owner/repo
        - https://github.com/owner/repo/tree/branch
        - https://github.com/owner/repo/tree/commit-hash

        Сначала пытается клонировать как branch. Если branch не существует,
        предполагает что это commit hash и использует GitHub Archive API (без истории).

        Возвращает путь к склонированной папке.
        """
        repo_url = str(request.repo_url)

        # Парсим URL для извлечения owner, repo, base_url и branch/commit из пути
        owner, reponame, base_url, commit_hash = resolve_full_github_url(repo_url)

        temp_dir = tempfile.mkdtemp(
            prefix=f"repo_{request.meta.request_id}_", dir=self.download_path
        )

        try:
            self.logger.info(
                "Assuming '%s' is a commit hash. Using GitHub Archive API...",
                commit_hash,
            )
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir = tempfile.mkdtemp(
                prefix=f"repo_{request.meta.request_id}_",
                dir=self.download_path,
            )
            download_github_archive(owner, reponame, commit_hash, temp_dir)
            msg = (
                "Successfully downloaded {base_url} at "
                "commit '{url_ref}' via archive API "
                f"for request_id={request.meta.request_id}."
            )
            self.logger.info(msg)

            return IndexJobResponse(
                meta=MetaResponse(
                    request_id=request.meta.request_id,
                    start_datetime=datetime.now(),  # будет перезаписано
                    end_datetime=datetime.now(),  # будет перезаписано
                    status="done",
                ),
                repo_url=f"{base_url}/tree/{commit_hash}",
                job_status=IndexJobStatus(
                    status="loaded", chunks_processed=0, repo_path=temp_dir
                ),
            )
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            msg = (
                "Failed to clone repository: {e} for "
                f"request_id={request.meta.request_id}."
            )
            self.logger.error(msg)
            return IndexJobResponse(
                meta=MetaResponse(
                    request_id=request.meta.request_id,
                    start_datetime=datetime.now(),  # будет перезаписано
                    end_datetime=datetime.now(),  # будет перезаписано
                    status="error",
                ),
                repo_url=request.repo_url,
                job_status=IndexJobStatus(
                    status="failed",
                    chunks_processed=0,
                    repo_path=temp_dir,
                    description_error=str(e),
                ),
            )

    async def save_vectors(
        self, vectors: List[Dict[str, Any]], index_job_response: IndexJobResponse
    ) -> IndexJobResponse:
        """
        Сохраняет вектора и метаданные в векторную БД.
        Принимает список словарей {id, vector, payload}.
        """
        if not vectors:
            return self._error_response(
                index_job_response, "No vectors to save into QDrant."
            )

        collection_name = self.collection_name
        msg = (
            f"Start saving {len(vectors)} vectors into QDrant collection "
            f"'{collection_name}' for request_id={index_job_response.meta.request_id}."
        )
        self.logger.info(msg)

        try:
            collections_response = self.vector_db_client.get_collections()
            existing_collections = []
            if (
                "result" in collections_response
                and "collections" in collections_response["result"]
            ):
                existing_collections = [
                    col["name"] for col in collections_response["result"]["collections"]
                ]

            if collection_name not in existing_collections:
                self.logger.info(
                    f"Collection '{collection_name}' does not exist. Creating..."
                )

                create_response = self.vector_db_client.create_collection(
                    collection_name
                )
                if create_response.get("status") != "ok":
                    return self._error_response(
                        index_job_response,
                        f"Failed to create collection: {create_response}.",
                    )

                self.logger.info(
                    f"Setting up payload indexes for '{collection_name}'..."
                )
                self.vector_db_client._setup_collection_indexes(collection_name)

                self.logger.info(
                    f"Collection '{collection_name}' created successfully"
                    f"for request_id={index_job_response.meta.request_id}."
                )

            # Batch vectors for efficient upload
            total_batches = (len(vectors) + self.batch_size - 1) // self.batch_size

            for batch_idx in range(0, len(vectors), self.batch_size):
                batch_end = min(batch_idx + self.batch_size, len(vectors))
                batch = vectors[batch_idx:batch_end]
                batch_num = (batch_idx // self.batch_size) + 1

                msg = (
                    f"Uploading batch {batch_num}/{total_batches} "
                    f"({len(batch)} vectors) for "
                    f"request_id={index_job_response.meta.request_id}."
                )
                self.logger.debug(msg)

                upsert_response = self.vector_db_client.add_vectors(
                    collection_name, batch
                )

                if upsert_response.get("status") != "ok":
                    msg = (
                        "Database returned non-ok status for batch "
                        f"{batch_num}/{total_batches}: {upsert_response}"
                    )
                    return self._error_response(
                        index_job_response,
                        msg,
                    )

            return self._success_response(
                index_job_response,
                f"Successfully saved {len(vectors)} vectors in {total_batches} batches",
            )

        except Exception as e:
            return self._error_response(
                index_job_response, f"Error while saving vectors to QDrant: {e}"
            )

    def createdir(self, directory: str) -> None:
        """Метод созданий вспомогательной директории."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def cleanup(self, path: str) -> None:
        """Метод для ручной очистки, если требуется"""
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    def _error_response(
        self, index_job_response: IndexJobResponse, error_message: str
    ) -> IndexJobResponse:
        self.logger.error(
            f"{error_message} for request_id={index_job_response.meta.request_id}."
        )
        index_job_response.meta.status = "error"
        index_job_response.job_status.status = "saved_to_qdrant"
        index_job_response.job_status.description_error = error_message
        return index_job_response

    def _success_response(
        self, index_job_response: IndexJobResponse, success_message: str
    ) -> IndexJobResponse:
        self.logger.info(
            f"{success_message} for request_id={index_job_response.meta.request_id}."
        )
        index_job_response.meta.status = "done"
        index_job_response.job_status.status = "saved_to_qdrant"
        return index_job_response

    def is_repo_indexed(self, repo_url: str) -> bool:
        """
        Проверяет, проиндексирован ли репозиторий в векторной базе данных.

        Args:
            repo_url: URL репозитория в формате f"{base_url}/commit/{commit_hash}"

        Returns:
            True если репозиторий уже проиндексирован, False в противном случае
        """
        try:
            # Проверяем существование коллекции
            collections_response = self.vector_db_client.get_collections()
            existing_collections = []
            if (
                "result" in collections_response
                and "collections" in collections_response["result"]
            ):
                existing_collections = [
                    col["name"] for col in collections_response["result"]["collections"]
                ]

            if self.collection_name not in existing_collections:
                msg = (
                    "Collection '{self.collection_name}' does not exist. "
                    "Repo is not indexed."
                )
                self.logger.debug(msg)
                return False

            # Используем scroll для поиска точек с указанным repo_url
            scroll_filter = {
                "must": [{"key": "repo_url", "match": {"value": str(repo_url)}}]
            }

            scroll_response = self.vector_db_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=1,
                with_payload=False,
            )

            # Проверяем, есть ли результаты
            if "result" in scroll_response and "points" in scroll_response["result"]:
                points = scroll_response["result"]["points"]
                is_indexed = len(points) > 0
                return is_indexed
            else:
                self.logger.warning(
                    f"Unexpected response format from QDrant: {scroll_response}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error checking if repo is indexed: {e}")
            return False

    def delete_repo_vectors(self, repo_url: str) -> bool:
        """
        Удаляет все векторы репозитория из векторной базы данных.

        Args:
            repo_url: URL репозитория в формате f"{base_url}/commit/{commit_hash}"

        Returns:
            True если удаление прошло успешно, False в противном случае
        """
        repo_url_str = str(repo_url)
        try:
            # Проверяем существование коллекции
            collections_response = self.vector_db_client.get_collections()
            existing_collections = []
            if (
                "result" in collections_response
                and "collections" in collections_response["result"]
            ):
                existing_collections = [
                    col["name"] for col in collections_response["result"]["collections"]
                ]

            if self.collection_name not in existing_collections:
                msg = (
                    "Collection '{self.collection_name}' does not exist. "
                    "Nothing to delete."
                )
                self.logger.warning(msg)
                return False

            # Сначала проверяем, есть ли что удалять
            scroll_filter = {
                "must": [
                    {
                        "key": "repo_url",
                        "match": {
                            "value": repo_url_str,
                        },
                    }
                ]
            }

            scroll_response = self.vector_db_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=1,
                with_payload=False,
            )

            if "result" in scroll_response and "points" in scroll_response["result"]:
                points = scroll_response["result"]["points"]
                if len(points) == 0:
                    self.logger.info(
                        f"No vectors found for repo {repo_url_str}. Nothing to delete."
                    )
                    return True

            # Удаляем все точки с указанным repo_url
            self.logger.info(f"Deleting vectors for repo {repo_url_str}...")
            delete_response = self.vector_db_client.delete_points(
                collection_name=self.collection_name, delete_filter=scroll_filter
            )

            if delete_response.get("status") == "ok":
                result = delete_response.get("result", {})
                operation_id = result.get("operation_id")
                msg = (
                    f"Successfully deleted vectors for "
                    f"repo {repo_url_str}. Operation ID: {operation_id}"
                )
                self.logger.info(msg)
                return True
            else:
                msg = (
                    "Failed to delete vectors for "
                    f"repo {repo_url_str}: {delete_response}"
                )
                self.logger.error(msg)
                return False

        except Exception as e:
            self.logger.error(f"Error deleting repo vectors: {e}")
            return False
