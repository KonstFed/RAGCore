from datetime import datetime
from omegaconf import DictConfig
from src.core.service import BaseService
from src.core.schemas import QueryRequest, QueryResponse, SearchConfig
from src.search.preprocessor import Preprocessor
from src.search.postprocessor import Postprocessor
from src.search.rewriter import QueryRewriter
from src.search.retriever import Retriever
from src.search.reranker import Reranker
from src.search.qa import QAGenerator


class SearchEngine(BaseService):
    """
    Класс отвечает за поиск и генерацию ответа (`/query`).
    Пайплайн: Preprocess -> QueryRewrite -> Retrieve ->
    -> Rerank -> ContextExpansion -> QA (LLM) -> Postprocess.
    """

    def __init__(self, config_path: str = "configs/deployment_config.yaml"):
        super().__init__(config_path)

        self.preprocessor = self._init_preprocessor(self.config)
        self.query_rewriter = self._init_query_rewriter(self.config)
        self.retriever = self._init_retriever(self.config)
        self.reranker = self._init_reranker(self.config)
        self.qa = self._init_qa(self.config)
        self.postprocessor = self._init_postprocessor(self.config)

        self.logger.info("SearchEngine service initialized.")

    def _init_preprocessor(self, config: DictConfig) -> Preprocessor:
        return Preprocessor(config)

    def _init_query_rewriter(self, config: DictConfig) -> QueryRewriter:
        return QueryRewriter(config)

    def _init_retriever(self, config: DictConfig) -> Retriever:
        return Retriever(config)

    def _init_reranker(self, config: DictConfig) -> Reranker:
        return Reranker(config)

    def _init_qa(self, config: DictConfig) -> QAGenerator:
        return QAGenerator(config)

    def _init_postprocessor(self, config: DictConfig) -> Postprocessor:
        return Postprocessor(config)

    async def predict(
        self, request: QueryRequest, config: SearchConfig
    ) -> QueryResponse:
        """
        Пайплайн обработки пользовательского запроса.
        """
        start_datetime = datetime.now()
        current_data = request

        try:
            current_data = self.preprocessor.pipeline(current_data, config)
            if isinstance(current_data, QueryResponse):
                return self._finalize_response(current_data, request, start_datetime)

            current_data = await self.query_rewriter.pipeline(current_data, config)

            current_data = self.retriever.retrieval(current_data, config)

            current_data = await self.reranker.pipeline(current_data, config)
            if isinstance(current_data, QueryResponse):
                return self._finalize_response(current_data, request, start_datetime)

            current_data = self.retriever.expansion(current_data, config)

            response = await self.qa.pipeline(current_data, config)

            response = self.postprocessor.pipeline(response, config)

            return self._finalize_response(response, request, start_datetime)

        except Exception:
            self.logger.exception(f"Critical error in job {request.meta.request_id}")

    def _finalize_response(
        self, response: QueryResponse, request: QueryRequest, start_time: datetime
    ) -> QueryResponse:
        """
        Вспомогательный метод для обновления метаданных перед возвратом ответа.
        Гарантирует, что request_id совпадает и проставляет время выполнения.
        """
        end_time = datetime.now()

        response.meta.request_id = request.meta.request_id
        response.meta.start_datetime = start_time
        response.meta.end_datetime = end_time
        response.meta.status = (
            "done" if not response.meta.status else response.meta.status
        )

        self.logger.info(
            f"Job {request.meta.request_id} completed. "
            f"Status: {response.status}. "
            f"Duration: {(end_time - start_time).total_seconds():.2f}s"
        )
        return response
