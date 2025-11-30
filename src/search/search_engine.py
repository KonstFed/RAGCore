import os
from datetime import datetime
from src.core.service import BaseService
from src.core.schemas import QueryRequest, QueryResponse, Chunk, SearchConfig
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
        return Preprocessor()

    def _init_query_rewriter(self) -> QueryRewriter:
        return QueryRewriter()

    def _init_retriever(self) -> Retriever:
        return Retriever()

    def _init_reranker(self) -> Reranker:
        return Reranker()

    def _init_qa(self) -> QAGenerator:
        return QAGenerator()

    def _init_postprocessor(self) -> Postprocessor:
        return Postprocessor()

    async def predict(self, request: QueryRequest, config: SearchConfig) -> QueryResponse:
        """
        Пайплайн обработки пользовательского запроса.
        """
        start_detatime = datetime.now()

        try:
            response = self.preprocessor.pipeline(request, config)

            response = await self.query_rewriter.pipeline(response, config)

            response = self.retriever.retrieval(response, config)

            response = await self.reranker.pipeline(response, config)

            response = self.retriever.expansion(response, config)

            response = await self.qa.pipeline(response, config)

            response = self.postprocessor.pipeline(response, config)

            response.update({ # TODO подумать, как лучше сделать формирование финального ответа
                "meta": {
                    "request_id": request.meta.request_id,
                    "start_detatime": start_detatime,
                    "end_datetime": datetime.now(),
                    "status": "done"
                }
            })

            self.logger.info(f"Job {request.meta.request_id} completed successfully.")
        except Exception as e:
            self.logger.exception(f"Critical error in job {request.meta.request_id}")

        return QueryResponse(**response)
