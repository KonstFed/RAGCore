import os
import re
import shutil
import tempfile
import tarfile
import io
import git
import requests
from datetime import datetime
from omegaconf import DictConfig
from src.core.db import VectorDBClient
from src.core.schemas import (
    IndexRequest,
    IndexJobResponse,
    IndexJobStatus,
    MetaResponse
)
from typing import List, Dict, Any, Tuple, Optional
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

    def _parse_github_url(self, url: str) -> Tuple[str, str, str, Optional[str]]:
        """
        Парсит GitHub URL и извлекает owner, repo, base_url и branch/commit.
        
        Args:
            url: GitHub URL, может содержать /tree/branch или /tree/commit-hash
            
        Returns:
            - owner: владелец репозитория
            - repo: название репозитория
            - base_url: базовый URL репозитория без /tree/...
            - branch_or_commit: извлеченный branch или commit hash, или None
        """
        # Паттерн для URL вида: https://github.com/owner/repo/tree/branch-or-commit
        pattern = r'^https://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+))?/?$'
        match = re.match(pattern, url)
        
        if not match:
            # Если не соответствует паттерну, возвращаем как есть (для не-GitHub URLs)
            return "", "", url, None
        
        owner, repo, tree_ref = match.groups()
        base_url = f"https://github.com/{owner}/{repo}"
        
        return owner, repo, base_url, tree_ref

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
        owner, repo, base_url, url_ref = self._parse_github_url(repo_url)

        temp_dir = tempfile.mkdtemp(prefix=f"repo_{request.meta.request_id}_", dir=self.download_path)

        if url_ref:
            self.logger.info(f"Cloning {base_url} (ref: {url_ref}) to {temp_dir} for request_id={request.meta.request_id}.")
        else:
            self.logger.info(f"Cloning {base_url} (default branch) to {temp_dir} for request_id={request.meta.request_id}.")

        try:
            if url_ref:
                # Всегда сначала пытаемся клонировать как branch
                # Это гарантирует, что мы не перепутаем branch name с commit hash
                try:
                    self.logger.info(f"Attempting to clone '{url_ref}' as branch...")
                    git.Repo.clone_from(
                        url=base_url,
                        to_path=temp_dir,
                        branch=url_ref,
                        depth=1
                    )
                    self.logger.info(f"Successfully cloned {base_url} as branch '{url_ref}'")
                except git.exc.GitCommandError:
                    # Если не получилось как branch, значит это commit hash
                    # Используем GitHub Archive API (без истории)
                    self.logger.info(f"Branch '{url_ref}' does not exist, assuming it's a commit hash. Using GitHub Archive API...")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    temp_dir = tempfile.mkdtemp(prefix=f"repo_{request.meta.request_id}_", dir=self.download_path)
                    self._download_github_archive(owner, repo, url_ref, temp_dir)
                    self.logger.info(f"Successfully downloaded {base_url} at commit '{url_ref}' via archive API")
            else:
                # Клонируем default branch без указания branch
                git.Repo.clone_from(
                    url=base_url,
                    to_path=temp_dir,
                    depth=1
                )
                self.logger.info(f"Successfully cloned {base_url} (default branch)")
            
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

    def _download_github_archive(self, owner: str, repo: str, commit_hash: str, target_dir: str) -> None:
        """
        Скачивает архив репозитория с GitHub для конкретного commit и распаковывает его.
        
        Args:
            owner: Владелец репозитория
            repo: Название репозитория
            commit_hash: Commit hash для скачивания
            target_dir: Директория для распаковки архива
        """
        # GitHub Archive API URL
        archive_url = f"https://github.com/{owner}/{repo}/archive/{commit_hash}.tar.gz"
        
        self.logger.info(f"Downloading archive from {archive_url}...")
        
        # Скачиваем архив
        response = requests.get(archive_url, timeout=300, stream=True)
        response.raise_for_status()
        
        # Распаковываем архив
        self.logger.info(f"Extracting archive to {target_dir}...")
        with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
            tar.extractall(path=target_dir)
        
        # GitHub создает папку вида repo-{short_hash}, находим её
        extracted_dirs = [
            d for d in os.listdir(target_dir) 
            if os.path.isdir(os.path.join(target_dir, d))
        ]
        
        if not extracted_dirs:
            raise ValueError("No directories found in extracted archive")
        
        # Перемещаем содержимое из вложенной папки на уровень выше
        actual_repo_path = os.path.join(target_dir, extracted_dirs[0])
        for item in os.listdir(actual_repo_path):
            shutil.move(
                os.path.join(actual_repo_path, item),
                os.path.join(target_dir, item)
            )
        
        # Удаляем пустую вложенную папку
        os.rmdir(actual_repo_path)
        
        self.logger.info(f"Archive extracted successfully to {target_dir}")

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
