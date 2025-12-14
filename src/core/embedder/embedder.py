from omegaconf import DictConfig
import requests
import json
from src.core.schemas import Chunk
from typing import List, Dict, Any, Tuple
from src.core.schemas import Chunk, IndexRequest, IndexConfig, IndexJobResponse
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

    async def vectorize(
        self,
        chunks: List[Chunk],
        config: IndexConfig,
        index_response: IndexJobResponse
    ) -> Tuple[IndexJobResponse, List[Dict[str, Any]]]:
        """
        Принимает список чанков, возвращает структуру готовую для вставки в Векторную БД.
        Формат возврата: List[{id: uuid, vector: list, payload: dict}]
        """
        vectors_data = []

        texts = [chunk.content for chunk in chunks]

        self.logger.info(f"Start vectorize chunks for request_id={index_response.meta.request_id}.")

        try:
            embeddings = self.embed(texts)

            for chunk, vector in zip(chunks, embeddings):
                payload = chunk.metadata.model_dump(mode='json')

                payload['repo_url'] = str(index_response.repo_url)
                payload['request_id'] = str(index_response.meta.request_id)
                payload['content'] = chunk.content

                vector_record = {
                    "id": str(chunk.metadata.chunk_id),
                    "vector": vector,
                    "payload": payload
                }
                vectors_data.append(vector_record)

            index_response.job_status.status = "vectorized"
            index_response.meta.status = "done"
            index_response.job_status.chunks_processed = len(vectors_data)
            self.logger.info(f"Successful done vectorize chunks for request_id={index_response.meta.request_id}.")
        except Exception as e:
            self.logger.error(f"Error vectorize chunks for request_id={index_response.meta.request_id} with body: {e}")
            index_response.job_status.status = "vectorized"
            index_response.meta.status = "error"
            return index_response, []

        return index_response, vectors_data

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
        # response = requests.post(self.url, headers=headers, data=json.dumps(data))
        # return [r.get('embedding') for r in response.json()['data']]
        import numpy as np
        return [
            np.random.random(1536) for _ in range(len(texts))
        ]