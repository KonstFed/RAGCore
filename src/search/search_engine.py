import os
from datetime import datetime
from src.core.service import BaseService
from src.core.schemas import QueryRequest, QueryResponse, Chunk
from src.search.preprocessor import Preprocessor
from src.search.postprocessor import Postprocessor
from src.search.rewriter import QueryRewriter
from src.search.retriever import Retriever
from src.search.reranker import Reranker
from src.search.qa import QAGenerator


class SearchEngine(BaseService):
    """
    Класс отвечает за поиск и генерацию ответа (`/query`).
    Пайплайн: Preprocess -> QueryRewrite -> Retrieve -> Rerank -> ContextExpansion -> QA (LLM) -> Postprocess.
    """
    def __init__(self, config_path: str = "configs/deployment_config.yaml"):
        super().__init__(config_path)

        self.preprocessor = self._init_preprocessor()
        self.query_rewriter = self._init_query_rewriter()
        self.retriever = self._init_retriever()
        self.reranker = self._init_reranker()
        self.qa = self._init_qa()
        self.postprocessor = self._init_postprocessor()

        self.logger.info("SearchEngine service initialized.")

    def _init_preprocessor(self) -> Preprocessor:
        return Preprocessor

    def _init_query_rewriter(self) -> QueryRewriter:
        return QueryRewriter

    def _init_retriever(self) -> Retriever:
        return Retriever

    def _init_reranker(self) -> Reranker:
        return Reranker

    def _init_qa(self) -> QAGenerator:
        return QAGenerator

    def _init_postprocessor(self) -> Postprocessor:
        return Postprocessor

    async def predict(self, request: QueryRequest) -> QueryResponse:
        """
        Пайплайн обработки пользовательского запроса.
        """
        start_detatime = datetime.now()

        try:
            response = self.preprocessor.pipeline(request)

            response = self.query_rewriter.pipeline(response)

            response = self.retriever.retrieval(response)

            response = self.reranker.pipeline(response)

            response = self.retriever.expansion(response)

            response = await self.qa.pipeline(response)

            response = self.postprocessor.pipeline(response)

            response.update({ # TODO подумать, как лучше сделать из postprocessor
                "meta": {
                    "request_id": request.meta.request_id,
                    "start_detatime": start_detatime,
                    "start_detatime": datetime.now(),
                    "status": "done"
                }
            })

            self.logger.info(f"Job {request.meta.request_id} completed successfully.")
        except Exception as e:
            self.logger.exception(f"Critical error in job {request.meta.request_id}")

        # TODO костыль
        response = {
            "meta": {
                "request_id": request.meta.request_id,
                "start_detatime": start_detatime,
                "end_datetime": datetime.now(),
                "status": "done"
            },
            "messages": request.query.messages,
            "answer": "Какой-то ответ сгенерированный от LLM, который прошёл построцессиг.",
            "sources": [
                {
                    "content": "Чанк 1",
                    "metadata": {
                        "chunk_id": "cdb68191-fbb0-4d35-ba8a-3d9ec19a4f29",
                        "filepath": "src/utils/helper.py",
                        "file_name": "helper.py",
                        "chunk_size": 390,
                        "line_count": 20,
                        "start_line_no": 0,
                        "end_line_no": 20,
                        "node_count": 0,
                        "language": "python"
                    }
                },
                {
                    "content": "Чанк 2",
                    "metadata": {
                        "chunk_id": "cdb68191-fbb0-4d35-ba8a-3d9ec19a4f29",
                        "filepath": "src/utils/helper.py",
                        "file_name": "helper.py",
                        "chunk_size": 100,
                        "line_count": 10,
                        "start_line_no": 20,
                        "end_line_no": 30,
                        "node_count": 0,
                        "language": "python"
                    }
                }
            ],
            "llm_usage": {
                "prompt_tokens": 10000,
                "completion_tokens": 102030
            }
        }

        return QueryResponse(**response)
