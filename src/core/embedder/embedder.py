from omegaconf import DictConfig
import requests
import json
from src.core.schemas import Chunk
from typing import List, Dict, Any
from src.core.schemas import Chunk, IndexRequest, IndexConfig
from src.utils.logger import get_logger


class EmbeddingModel:
    """
    Класс для векторизации текста.
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.url = cfg.embeddings.url
        self.api_key = cfg.embeddings.api_key
        self.model_name = cfg.embeddings.model_name

    async def vectorize(self, chunks: List[Chunk], request: IndexRequest, config: IndexConfig) -> List[Dict[str, Any]]:
        """
        Принимает список чанков, возвращает структуру готовую для вставки в Векторную БД.
        Формат возврата: List[{id: uuid, vector: list, payload: dict}]
        """
        vectors_data = []

        texts = [chunk.content for chunk in chunks]

        self.logger.info("Start vectorize chunks.")
        embeddings = self.embed(texts)

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


    def embed(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model_name,
            "task": "nl2code.query",
            "truncate": False,
            "input": texts
        }
        response = requests.post(self.url, headers=headers, data=json.dumps(data))
        return [r.get('embedding') for r in response.json()['data']]
