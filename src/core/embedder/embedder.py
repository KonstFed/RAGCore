import os
from omegaconf import DictConfig
from src.core.schemas import Chunk
from typing import List, Dict, Any
import random
from src.core.schemas import Chunk, IndexRequest, IndexConfig
from src.utils.logger import get_logger


class EmbeddingModel:
    """
    Класс для векторизации текста.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # TODO реализовать настройку EmbeddingModel из cfg
        self.logger = get_logger(self.__class__.__name__)
        self.dimension = 1024

    async def vectorize(self, chunks: List[Chunk], request: IndexRequest, config: IndexConfig) -> List[Dict[str, Any]]:
        """
        Принимает список чанков, возвращает структуру готовую для вставки в Векторную БД.
        Формат возврата: List[{id: uuid, vector: list, payload: dict}]
        """
        vectors_data = []

        texts = [chunk.content for chunk in chunks]

        self.logger.info("Start vectorize chunks.")
        # ЭМУЛЯЦИЯ ПОЛУЧЕНИЯ ЭМБЕДИНГОВ
        # В реальности: embeddings = self.model.encode(texts)
        embeddings = self._mock_embeddings(len(texts))

        for chunk, vector in zip(chunks, embeddings):
            payload = chunk.metadata.model_dump(mode='json')

            payload['repo_url'] = str(request.repo_url)
            payload['request_id'] = str(request.meta.request_id)
            payload['content'] = chunk.content

            vector_record = {
                "id": str(chunk.metadata.chunk_id),
                "vector": vector,
                "payload": payload
            }
            vectors_data.append(vector_record)

        self.logger.info(f"Successful done vectorize chunks for request_id={request.meta.request_id}.")

        return vectors_data

    def _mock_embeddings(self, count: int) -> List[List[float]]: # TODO убрать, сейчас используется для теста
        """Генерирует случайные вектора для теста."""
        return [
            [random.random() for _ in range(self.dimension)] 
            for _ in range(count)
        ]
