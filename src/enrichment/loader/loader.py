import os
import shutil
import tempfile
import uuid
import git
from omegaconf import DictConfig
from src.core.db import VectorDBClient
from src.core.schemas import IndexRequest, IndexJobResponse
from typing import List, Dict, Any
from src.utils.logger import get_logger


class LoaderConnecter:
    """
    Отвечает за взаимодействие с внешними источниками данных (Git)
    и хранилищами (Vector DB).
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.download_path = cfg.paths.temp_repo_storage
        # TODO реализовать настройку LoaderConnecter из cfg
        self.logger = get_logger(self.__class__.__name__)
        self.vectore_db_client = VectorDBClient(cfg) # TODO реализовать

    async def clone_repository(self, request: IndexRequest) -> str:
        """
        Клонирует репозиторий во временную директорию.
        Возвращает путь к склонированной папке.
        """
        repo_url = str(request.repo_url)
        branch = request.branch

        temp_dir = tempfile.mkdtemp(prefix=f"repo_{request.meta.request_id}_", dir=self.download_path)

        self.logger.info(f"Cloning {repo_url} (branch: {branch}) to {temp_dir}")

        try:
            git.Repo.clone_from(
                url=repo_url,
                to_path=temp_dir,
                branch=branch,
                depth=1
            )
            self.logger.info(f"Successful cloning {repo_url} (branch: {branch}) to {temp_dir}")
            return temp_dir
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.error(f"Failed to clone repository: {e}")
            raise e

    async def save_vectors(self, vectors: List[Dict[str, Any]], request: IndexRequest) -> Dict[str, Any]:
        """
        Сохраняет вектора и метаданные в векторную БД.
        Принимает список словарей {id, vector, payload}.
        """
        # ЭМУЛЯЦИЯ СОХРАНЕНИЯ
        # В реальности здесь код: qdrant_client.upsert(collection_name, points=vectors)
        # TODO реализовать

        self.logger.info("Start saving vectors into VectorStore.")

        if vectors:
            self.logger.debug(f"Sample vector ID: {vectors[0].get('id')}")

        return {
            "status": "vectors_saved",
            "count": len(vectors)
        }

    def cleanup(self, path: str) -> None:
        """Метод для ручной очистки, если требуется"""
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
