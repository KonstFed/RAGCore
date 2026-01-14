from omegaconf import DictConfig
from pathlib import Path
import requests
import json
from src.core.schemas import Chunk
from typing import List, Dict, Any, Tuple
from src.core.schemas import IndexJobResponse
from src.utils.logger import get_logger


class EmbeddingModel:
    """
    Класс для векторизации текста.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.provider = getattr(cfg.embeddings, "default_provider", "jina")
        self.url = cfg.embeddings.url
        self.api_key = cfg.embeddings.api_key
        self.model_name = cfg.embeddings.model_name
        self.batch_size = cfg.embeddings.batch_size
        self.dump_dir = cfg.paths.temp_chunks_storage

    async def vectorize(
        self, chunks: List[Chunk], index_response: IndexJobResponse
    ) -> Tuple[IndexJobResponse, List[Dict[str, Any]]]:
        """
        Принимает список чанков,
        возвращает структуру готовую для вставки в Векторную БД.
        Формат возврата: List[{id: uuid, vector: list, payload: dict}]
        """
        vectors_data = []

        texts = [chunk.content for chunk in chunks]

        self.logger.info(
            f"Start vectorize chunks for request_id={index_response.meta.request_id}."
        )

        try:
            embeddings = self.embed_chunks(texts)

            for chunk, vector in zip(chunks, embeddings):
                payload = chunk.metadata.model_dump(mode="json")

                payload["repo_url"] = str(index_response.repo_url)
                payload["request_id"] = str(index_response.meta.request_id)
                payload["content"] = chunk.content

                vector_record = {
                    "id": str(chunk.metadata.chunk_id),
                    "vector": vector,
                    "payload": payload,
                }
                vectors_data.append(vector_record)

            self._save_chunks_locally(vectors_data, index_response.meta.request_id)

            index_response.job_status.status = "vectorized"
            index_response.meta.status = "done"
            index_response.job_status.chunks_processed = len(vectors_data)
            msg = (
                f"Successful done vectorize chunks for "
                f"request_id={index_response.meta.request_id}."
            )
            self.logger.info(msg)
        except Exception as e:
            msg = (
                f"Error vectorize chunks for "
                f"request_id={index_response.meta.request_id} with body: {e}"
            )
            self.logger.error(msg)
            index_response.job_status.status = "vectorized"
            index_response.meta.status = "error"
            return index_response, []

        return index_response, vectors_data

    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """Векторизует чанки из репозитория с батчевой обработкой."""
        all_embeddings = []

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        for batch_num, i in enumerate(range(0, len(texts), self.batch_size)):
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            msg = (
                f"Processing batch {batch_num + 1}/{total_batches} with "
                f"batch_size {self.batch_size}."
            )
            self.logger.debug(msg)
            batch_texts = texts[i : i + self.batch_size]

            if self.provider == "openrouter":
                data = {
                    "model": self.model_name,
                    "input": batch_texts,
                }
            else:
                data = {
                    "model": self.model_name,
                    "task": "nl2code.passage",
                    "truncate": True,
                    "input": batch_texts[9:11],
                }
            try:
                response = requests.post(
                    self.url, headers=headers, data=json.dumps(data)
                )
                response.raise_for_status()

                response_data = response.json()

                if "data" in response_data:
                    batch_embeddings = [
                        r.get("embedding") for r in response_data["data"]
                    ]
                    all_embeddings.extend(batch_embeddings)
                else:
                    msg = (
                        f"Unexpected response format for batch starting at index "
                        f"{i}: {response_data}"
                    )
                    self.logger.error(msg)

            except requests.exceptions.RequestException as e:
                msg = f"Request failed for batch starting at index {i}: {e}"
                self.logger.error(msg)

        return all_embeddings

    def embed_query(self, texts: List[str]) -> List[List[float]]:
        """Векторизует пользовательский запрос."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.provider == "openrouter":
            data = {
                "model": self.model_name,
                "input": texts,
            }
        else:
            data = {
                "model": self.model_name,
                "task": "nl2code.query",
                "truncate": True,
                "input": texts,
            }
        response = requests.post(self.url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            msg = (
                f"Failed to get embedding for query: "
                f"status={response.status_code}, response={response.text}"
            )
            self.logger.error(msg)
            return [[]]
        self.logger.info("Successfuly embedded user question")
        return [r.get("embedding") for r in response.json()["data"]]

    def _save_chunks_locally(
        self, chunks: List[Dict[str, Any]], request_id: str
    ) -> None:
        """
        Сериализует список чанков в JSON и сохраняет на диск.
        Возвращает путь к созданному файлу.
        """
        try:
            output_dir = Path(self.dump_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{request_id}.json"
            file_path = output_dir / filename

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(
                f"Failed to save chunks locally for request_id={request_id}: {e}"
            )
