from typing import Any, Dict, List, Optional

import requests
from omegaconf import DictConfig

from src.utils.logger import get_logger


class VectorDBClient:
    """Класс клиента векторной базы данных."""

    def __init__(self, cfg: DictConfig) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.db_url = cfg.database.url
        self.dimension = cfg.embeddings.dimension
        self.distance = cfg.embeddings.distance
        self.top_k = cfg.database.top_k

    def get_collections(self) -> Dict:
        """Получает список коллекций из векторной базы данных."""
        response = requests.get(f"{self.db_url}/collections")
        return response.json()

    def create_collection(self, collection_name: str) -> Dict[str, Any]:
        """Создает коллекцию в векторной базе данных."""
        data = {"vectors": {"size": self.dimension, "distance": self.distance}}
        response = requests.put(
            f"{self.db_url}/collections/{collection_name}", json=data
        )
        return response.json()

    def get_collection(self, collection_name: str) -> Dict[str, Any]:
        """Получает информацию о коллекции."""
        response = requests.get(f"{self.db_url}/collections/{collection_name}")
        return response.json()

    def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """Удаляет коллекцию из векторной базы данных."""
        response = requests.delete(f"{self.db_url}/collections/{collection_name}")
        return response.json()

    def add_vectors(
        self, collection_name: str, vectorized_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Добавляет вектора в коллекцию"""

        url = f"{self.db_url}/collections/{collection_name}/points"

        params = {"wait": "true"}
        payload = {"points": vectorized_data}
        headers = {"Content-Type": "application/json"}

        response = requests.put(url, params=params, headers=headers, json=payload)

        return response.json()

    def search(
        self,
        collection_name: str,
        vector: List[float],
        top_k: int,
        query_filter: Optional[Dict] = None,
        with_payload: bool = True,
    ) -> Dict[str, Any]:
        """
        Универсальный поиск с поддержкой фильтров.
        """
        url = f"{self.db_url}/collections/{collection_name}/points/search"

        payload = {
            "vector": vector,
            "limit": top_k,
            "with_payload": with_payload,
            "with_vector": False,
        }

        if query_filter:
            payload["filter"] = query_filter

        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload)
        return response.json()

    def scroll(
        self,
        collection_name: str,
        scroll_filter: Dict,
        limit: int = 1,
        with_payload: bool = True,
    ) -> Dict[str, Any]:
        """
        Метод Scroll для получения конкретных записей (используется для Expansion).
        """
        url = f"{self.db_url}/collections/{collection_name}/points/scroll"

        payload = {
            "filter": scroll_filter,
            "limit": limit,
            "with_payload": with_payload,
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload)
        return response.json()

    def delete_points(
        self,
        collection_name: str,
        delete_filter: Dict,
    ) -> Dict[str, Any]:
        """
        Удаляет точки из коллекции по фильтру.

        Args:
            collection_name: Название коллекции
            delete_filter: Фильтр для выбора точек для удаления (формат QDrant filter)

        Returns:
            Ответ от QDrant API
        """
        url = f"{self.db_url}/collections/{collection_name}/points/delete"

        payload = {
            "filter": delete_filter,
        }

        params = {"wait": "true"}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, params=params, headers=headers, json=payload)
        return response.json()

    def _setup_collection_indexes(self, collection_name: str) -> None:
        """
        Создает индексы полей (Payload Indexes) в QDrant.
        Необходимы для быстрой фильтрации и гибридного поиска (BM25).
        """
        keyword_fields = ["chunk_id", "filepath", "language", "repo_url", "request_id"]
        integer_fields = ["start_line_no", "end_line_no", "chunk_size"]

        for field in keyword_fields:
            self._create_payload_index(collection_name, field, "keyword")

        for field in integer_fields:
            self._create_payload_index(collection_name, field, "integer")

        self._create_payload_index(collection_name, "content", "text")

    def _create_payload_index(
        self, collection_name: str, field_name: str, field_type: str
    ) -> None:
        """
        Отправляет запрос в QDrant на создание индекса для конкретного поля.
        """
        url = f"{self.db_url}/collections/{collection_name}/index"
        headers = {"Content-Type": "application/json"}
        payload = {"field_name": field_name, "field_schema": field_type}
        if field_type == "text":
            payload["field_schema"] = {
                "type": "text",
                "tokenizer": "word",
                "min_token_len": 2,
                "max_token_len": 100,
                "lowercase": True,
            }

        try:
            response = requests.put(url, headers=headers, json=payload)
            if response.status_code != 200:
                msg = (
                    f"Failed to create index for field '{field_name}' in "
                    f"'{collection_name}': {response.text}"
                )
                self.logger.warning(msg)
        except Exception as e:
            msg = f"Error creating index for '{field_name}': {e}"
            self.logger.error(f"Error creating index for '{field_name}': {e}")
