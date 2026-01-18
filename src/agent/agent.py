import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig

from src.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    ANALYSIS_PROMPT_TEMPLATE,
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_USER_TEMPLATE
)
from src.core.schemas import (
    AgentAction,
    AgentConfig,
    AgentState,
    IterationResult,
    QueryRequest,
    QueryResponse,
    SearchAdjustments,
    SearchConfig,
    FilterAdjustments,
    RetrieverConfig,
    RerankerConfig,
    LLMConfig,
    MetaResponse,
    Chunk,
    QaConfig,
)
from src.core.service import BaseService
from src.search.search_engine import SearchEngine
from src.core.llm import LLMClient
from src.utils.logger import get_logger


class Agent(BaseService):
    """
    Агент для углубленного итеративного поиска по репозиторию.

    Выполняет многоитерационный поиск с автоматическим анализом результатов,
    переформулированием запросов, настройкой фильтров и параметров поиска
    для достижения оптимальных результатов.
    """

    def __init__(self, config_path: str = "configs/deployment_config.yaml"):
        super().__init__(config_path)
        self.logger = get_logger(self.__class__.__name__)

        self.searcher = self._init_searcher(config_path)
        self._llm_client: Optional[LLMClient] = None
        self._llm_config: Optional[LLMConfig] = None

        self.logger.info("Agent service initialized.")

    def _init_searcher(self, config_path: str) -> SearchEngine:
        return SearchEngine(config_path)

    def _get_llm_client(self, llm_config: LLMConfig) -> LLMClient:
        """Получить или создать LLM клиент с кэшированием."""
        if self._llm_client is None or self._llm_config != llm_config:
            self._llm_client = LLMClient(llm_config)
            self._llm_config = llm_config
        return self._llm_client

    async def _call_llm(
        self,
        llm_config: LLMConfig,
        system_prompt: str,
        user_prompt: str,
        parse_json: bool = True
    ) -> Dict[str, Any]:
        """
        Вызвать LLM и получить ответ.

        Args:
            llm_config: Конфигурация LLM.
            system_prompt: Системный промпт.
            user_prompt: Пользовательский промпт.
            parse_json: Парсить ответ как JSON.

        Returns:
            Словарь с ответом LLM или распарсенный JSON.
        """
        client = self._get_llm_client(llm_config)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = await client.chat_completion(
            messages=messages,
            temperature=llm_config.parameters.temperature if llm_config.parameters else 0.1,
            max_tokens=llm_config.parameters.max_tokens if llm_config.parameters else 2048,
        )

        content = response.choices[0].message.content

        if parse_json:
            content = self._extract_json(content)
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse LLM JSON response: {e}")
                return {"error": str(e), "raw_content": content}

        return {"content": content}

    def _extract_json(self, text: str) -> str:
        """Извлечь JSON из текста, убрав markdown обертку."""
        text = text.strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    def _get_default_search_config(self) -> SearchConfig:
        """Получить конфигурацию поиска по умолчанию."""
        return SearchConfig(
            retriever=RetrieverConfig(
                size=10,
                threshold=0.3,
                bm25_weight=0.3
            ),
            reranker=RerankerConfig(
                enabled=True,
                top_k=5,
                threshold=0.4
            ),
            qa=QaConfig(enabled=False)
        )

    def _apply_search_adjustments(
        self,
        config: SearchConfig,
        adjustments: Optional[SearchAdjustments]
    ) -> SearchConfig:
        """Применить корректировки к конфигурации ретривера и реранкера."""
        if adjustments is None:
            return config

        config_dict = config.model_dump()

        if config_dict.get("retriever") is None:
            config_dict["retriever"] = {}

        retriever = config_dict["retriever"]
        if adjustments.retriever_size is not None:
            retriever["size"] = adjustments.retriever_size
        if adjustments.retriever_threshold is not None:
            retriever["threshold"] = adjustments.retriever_threshold
        if adjustments.bm25_weight is not None:
            retriever["bm25_weight"] = adjustments.bm25_weight

        if config_dict.get("reranker") is None:
            config_dict["reranker"] = {"enabled": True}

        reranker = config_dict["reranker"]
        if adjustments.reranker_top_k is not None:
            reranker["top_k"] = adjustments.reranker_top_k
        if adjustments.reranker_threshold is not None:
            reranker["threshold"] = adjustments.reranker_threshold

        return SearchConfig(**config_dict)

    def _apply_filter_adjustments(
        self,
        config: SearchConfig,
        adjustments: Optional[FilterAdjustments]
    ) -> SearchConfig:
        """Применить корректировки фильтров."""
        if adjustments is None:
            return config

        config_dict = config.model_dump()
        conditions = []

        if adjustments.languages:
            conditions.append({
                "name": "language",
                "operator": "in",
                "value": adjustments.languages
            })

        if adjustments.include_filepaths:
            for pattern in adjustments.include_filepaths:
                conditions.append({
                    "name": "filepath",
                    "operator": "wildcard",
                    "value": pattern
                })

        if adjustments.exclude_filepaths:
            for pattern in adjustments.exclude_filepaths:
                conditions.append({
                    "operator": "and",
                    "values": [{
                        "name": "filepath",
                        "operator": "wildcard",
                        "value": pattern
                    }]
                })

        if conditions:
            if len(conditions) == 1:
                filter_node = conditions[0]
            else:
                filter_node = {
                    "operator": "and",
                    "values": conditions
                }

            config_dict["filtering"] = {
                "enabled": True,
                "filter": filter_node
            }

        return SearchConfig(**config_dict)

    def _calculate_chunk_stats(
        self,
        chunks: List[Chunk],
        relevance_threshold: float
    ) -> Dict[str, Any]:
        """Вычислить статистику по чанкам."""
        if not chunks:
            return {
                "count": 0,
                "relevant_count": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
            }

        scores = []
        for chunk in chunks:
            score = chunk.reranker_relevance_score or chunk.retrieval_relevance_score or 0.0
            scores.append(score)

        relevant_count = sum(1 for s in scores if s >= relevance_threshold)

        return {
            "count": len(chunks),
            "relevant_count": relevant_count,
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores)
        }

    def _build_chunks_summary(
        self,
        chunks: List[Chunk],
        max_chunks: int = 5,
        max_content_length: int = 300
    ) -> str:
        """Построить текстовое описание найденных чанков для LLM."""
        if not chunks:
            return "Чанки не найдены."

        summaries = []
        for i, chunk in enumerate(chunks[:max_chunks]):
            score = chunk.reranker_relevance_score or chunk.retrieval_relevance_score or 0.0

            content = chunk.content.strip()
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."

            content = content.replace("\n", "\n   ")

            summaries.append(
                f"### Чанк {i + 1}\n"
                f"- **Файл:** `{chunk.metadata.filepath}`\n"
                f"- **Строки:** {chunk.metadata.start_line_no}-{chunk.metadata.end_line_no}\n"
                f"- **Язык:** {chunk.metadata.language or 'unknown'}\n"
                f"- **Score:** {score:.3f}\n"
                f"```\n{content}\n```"
            )

        result = "\n\n".join(summaries)

        if len(chunks) > max_chunks:
            result += f"\n\n... и ещё {len(chunks) - max_chunks} чанков"

        return result

    def _build_history_summary(self, iterations: List[IterationResult]) -> str:
        """Построить краткую историю итераций."""
        if not iterations:
            return "Это первая итерация."

        lines = []
        for it in iterations[-3:]:
            lines.append(
                f"- Итерация {it.iteration}: "
                f"найдено {it.chunks_found} чанков, "
                f"релевантных {it.relevant_chunks_count}, "
                f"avg_score={it.avg_relevance_score:.3f}, "
                f"действие: {it.action.action_type}"
            )

        return "\n".join(lines)

    async def _analyze_with_llm(
        self,
        state: AgentState,
        chunks: List[Chunk],
        config: AgentConfig,
    ) -> AgentAction:
        """Проанализировать результаты с помощью LLM."""
        stats = self._calculate_chunk_stats(chunks, config.relevance_score_threshold)
        remaining_time = config.max_time_seconds - (time.time() - state.start_time)

        retriever = state.search_config.retriever or RetrieverConfig()
        reranker = state.search_config.reranker or RerankerConfig()
        filtering = state.search_config.filtering

        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            original_query=state.original_query,
            current_query=state.current_query,
            iteration=len(state.iterations) + 1,
            max_iterations=config.max_iterations,
            remaining_time=remaining_time,
            confidence_threshold=config.confidence_threshold,
            min_relevant_chunks=config.min_relevant_chunks,
            retriever_size=retriever.size,
            retriever_threshold=retriever.threshold,
            bm25_weight=retriever.bm25_weight,
            reranker_enabled=reranker.enabled,
            reranker_top_k=reranker.top_k,
            filters=filtering.model_dump() if filtering and filtering.enabled else "нет",
            num_chunks=stats["count"],
            relevant_chunks=stats["relevant_count"],
            relevance_threshold=config.relevance_score_threshold,
            avg_score=stats["avg_score"],
            max_score=stats["max_score"],
            chunks_summary=self._build_chunks_summary(chunks),
            history_summary=self._build_history_summary(state.iterations)
        )

        try:
            result = await self._call_llm(
                llm_config=config.llm_config,
                system_prompt=AGENT_SYSTEM_PROMPT,
                user_prompt=prompt,
                parse_json=True
            )

            if "error" in result:
                self.logger.warning(f"LLM returned error, using heuristics: {result.get('error')}")
                return self._heuristic_analysis(state, chunks, config)

            search_adj = None
            if result.get("search_adjustments"):
                search_adj = SearchAdjustments(**{
                    k: v for k, v in result["search_adjustments"].items()
                    if v is not None
                })

            filter_adj = None
            if result.get("filter_adjustments"):
                filter_adj = FilterAdjustments(**{
                    k: v for k, v in result["filter_adjustments"].items()
                    if v is not None
                })

            return AgentAction(
                action_type=result.get("action_type", "stop_limit"),
                confidence=result.get("confidence", stats["avg_score"]),
                reasoning=result.get("reasoning", "LLM analysis"),
                refined_query=result.get("refined_query"),
                search_adjustments=search_adj,
                filter_adjustments=filter_adj,
                focus_areas=result.get("focus_areas")
            )

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return self._heuristic_analysis(state, chunks, config)

    def _heuristic_analysis(
        self,
        state: AgentState,
        chunks: List[Chunk],
        config: AgentConfig,
    ) -> AgentAction:
        """Эвристический анализ без LLM."""
        stats = self._calculate_chunk_stats(chunks, config.relevance_score_threshold)
        remaining_time = config.max_time_seconds - (time.time() - state.start_time)
        iteration = len(state.iterations) + 1

        if iteration >= config.max_iterations or remaining_time < 5:
            return AgentAction(
                action_type="stop_limit",
                confidence=stats["avg_score"],
                reasoning=f"Достигнут лимит: итерация {iteration}/{config.max_iterations}, "
                          f"осталось {remaining_time:.1f} сек."
            )

        if (stats["relevant_count"] >= config.min_relevant_chunks and
            stats["avg_score"] >= config.confidence_threshold):
            return AgentAction(
                action_type="stop_success",
                confidence=stats["avg_score"],
                reasoning=f"Найдено {stats['relevant_count']} релевантных чанков "
                          f"со средним score {stats['avg_score']:.3f}"
            )

        if stats["count"] == 0:
            return AgentAction(
                action_type="expand_search",
                confidence=0.0,
                reasoning="Чанки не найдены, расширяем параметры поиска.",
                search_adjustments=SearchAdjustments(
                    retriever_size=20,
                    retriever_threshold=0.1,
                    bm25_weight=0.5,
                )
            )

        if stats["relevant_count"] < config.min_relevant_chunks:
            if stats["count"] >= 5 and stats["avg_score"] < 0.3:
                return AgentAction(
                    action_type="refine_query",
                    confidence=stats["avg_score"],
                    reasoning=f"Низкая релевантность (avg={stats['avg_score']:.3f}), "
                              "требуется уточнение запроса."
                )

            return AgentAction(
                action_type="expand_search",
                confidence=stats["avg_score"],
                reasoning=f"Найдено мало релевантных чанков "
                          f"({stats['relevant_count']}/{config.min_relevant_chunks})",
                search_adjustments=SearchAdjustments(
                    retriever_size=min(30, (state.search_config.retriever.size or 10) + 10),
                    retriever_threshold=max(0.1, (state.search_config.retriever.threshold or 0.3) - 0.1)
                )
            )

        if stats["avg_score"] >= 0.5:
            return AgentAction(
                action_type="stop_success",
                confidence=stats["avg_score"],
                reasoning="Достигнут приемлемый уровень релевантности."
            )

        return AgentAction(
            action_type="stop_limit",
            confidence=stats["avg_score"],
            reasoning="Не удалось значительно улучшить результаты."
        )

    async def _analyze_results(
        self,
        state: AgentState,
        chunks: List[Chunk],
        config: AgentConfig,
    ) -> AgentAction:
        """Проанализировать результаты поиска и определить следующее действие."""
        if config.llm_config is not None:
            return await self._analyze_with_llm(state, chunks, config)
        else:
            return self._heuristic_analysis(state, chunks, config)

    def _add_unique_chunks(self, state: AgentState, new_chunks: List[Chunk]) -> List[Chunk]:
        """Добавить уникальные чанки в состояние."""
        added = []
        for chunk in new_chunks:
            chunk_id = str(chunk.metadata.chunk_id)
            if chunk_id not in state.seen_chunk_ids:
                state.seen_chunk_ids.add(chunk_id)
                state.all_chunks.append(chunk)
                added.append(chunk)
        return added

    def _get_best_chunks(self, state: AgentState, top_k: int = 10) -> List[Chunk]:
        """Получить лучшие чанки из всех итераций."""
        sorted_chunks = sorted(
            state.all_chunks,
            key=lambda c: c.reranker_relevance_score or c.retrieval_relevance_score or 0,
            reverse=True
        )
        return sorted_chunks[:top_k]

    def _format_sources_for_answer(self, chunks: List[Chunk]) -> str:
        """Форматировать источники для генерации финального ответа."""
        if not chunks:
            return "Источники не найдены."

        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(
                f"### Источник {i + 1}\n"
                f"**Файл:** `{chunk.metadata.filepath}` "
                f"(строки {chunk.metadata.start_line_no}-{chunk.metadata.end_line_no})\n"
                f"**Язык:** {chunk.metadata.language or 'не определен'}\n\n"
                f"```{chunk.metadata.language or ''}\n{chunk.content}\n```"
            )

        return "\n\n---\n\n".join(parts)

    async def _generate_final_answer(
        self,
        query: str,
        chunks: List[Chunk],
        config: AgentConfig
    ) -> str:
        """Сгенерировать финальный ответ на основе найденных чанков."""
        if not config.llm_config:
            return self._format_sources_for_answer(chunks)

        if not chunks:
            return "К сожалению, не удалось найти релевантные фрагменты кода для ответа на ваш вопрос."

        sources_text = self._format_sources_for_answer(chunks)

        prompt = FINAL_ANSWER_USER_TEMPLATE.format(
            query=query,
            sources=sources_text
        )

        try:
            result = await self._call_llm(
                llm_config=config.llm_config,
                system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
                user_prompt=prompt,
                parse_json=False
            )
            return result.get("content", sources_text)
        except Exception as e:
            self.logger.error(f"Failed to generate final answer: {e}")
            return sources_text

    async def predict(
        self,
        request: QueryRequest,
        config: AgentConfig
    ) -> QueryResponse:
        """
        Выполнить углубленный итеративный поиск.

        Args:
            request: Запрос пользователя с историей сообщений.
            config: Конфигурация агента.

        Returns:
            QueryResponse с найденными источниками и сгенерированным ответом.
        """
        start_time = time.time()
        start_datetime = datetime.now()

        self.logger.info(
            f"Starting agent search for request {request.meta.request_id}, "
            f"max_iterations={config.max_iterations}, "
            f"max_time={config.max_time_seconds}s"
        )

        current_query = ""
        if request.query.messages:
            for msg in reversed(request.query.messages):
                if msg.role == "user":
                    current_query = msg.content
                    break

        if not current_query:
            self.logger.warning("No user query found in request")
            return self._build_error_response(
                request, start_datetime, "No query provided"
            )

        state = AgentState(
            current_query=current_query,
            original_query=current_query,
            search_config=config.initial_search_config or self._get_default_search_config(),
            start_time=start_time,
        )

        final_chunks: List[Chunk] = []
        last_response: Optional[QueryResponse] = None

        for iteration in range(1, config.max_iterations + 1):
            iteration_start = time.time()

            elapsed = time.time() - start_time
            if elapsed >= config.max_time_seconds:
                self.logger.info(f"Time limit reached at iteration {iteration}")
                break

            self.logger.info(
                f"Iteration {iteration}/{config.max_iterations}: "
                f"query='{state.current_query[:80]}...'"
            )

            search_request = request.model_copy(deep=True)
            if state.current_query != state.original_query:
                for msg in reversed(search_request.query.messages):
                    if msg.role == "user":
                        msg.content = state.current_query
                        break

            try:
                response = await self.searcher.predict(
                    search_request,
                    state.search_config
                )
                chunks = response.sources or []
                last_response = response

                new_chunks = self._add_unique_chunks(state, chunks)
                self.logger.debug(f"Found {len(chunks)} chunks, {len(new_chunks)} new")

            except Exception as e:
                self.logger.error(f"Search failed at iteration {iteration}: {e}")
                chunks = []

            action = await self._analyze_results(state, chunks, config)

            stats = self._calculate_chunk_stats(chunks, config.relevance_score_threshold)
            iteration_result = IterationResult(
                iteration=iteration,
                query_used=state.current_query,
                chunks_found=len(chunks),
                relevant_chunks_count=stats["relevant_count"],
                avg_relevance_score=stats["avg_score"],
                max_relevance_score=stats["max_score"],
                action=action,
                duration_seconds=time.time() - iteration_start,
                search_config_snapshot=state.search_config.model_dump(),
            )
            state.iterations.append(iteration_result)

            self.logger.info(
                f"Iteration {iteration} completed: "
                f"chunks={len(chunks)}, relevant={stats['relevant_count']}, "
                f"avg_score={stats['avg_score']:.3f}, "
                f"action={action.action_type}, confidence={action.confidence:.3f}"
            )

            if action.action_type in ("stop_success", "stop_limit"):
                break

            if action.action_type == "refine_query" and action.refined_query:
                if config.enable_query_refinement:
                    state.current_query = action.refined_query
                    self.logger.info(f"Query refined to: '{action.refined_query[:80]}...'")

            if action.action_type in ("expand_search", "narrow_search", "combined_action"):
                if config.enable_retriever_adjustment and action.search_adjustments:
                    state.search_config = self._apply_search_adjustments(
                        state.search_config,
                        action.search_adjustments
                    )

            if action.action_type in ("adjust_filters", "combined_action"):
                if config.enable_filter_adjustment and action.filter_adjustments:
                    state.search_config = self._apply_filter_adjustments(
                        state.search_config,
                        action.filter_adjustments
                    )

        end_datetime = datetime.now()
        final_chunks = self._get_best_chunks(state, top_k=10)

        if not final_chunks:
            status = "no_relevance_sources"
        elif config.generate_final_answer and config.llm_config:
            status = "llm_rag"
        else:
            status = "no_llm"

        if config.generate_final_answer and final_chunks:
            answer = await self._generate_final_answer(
                state.original_query,
                final_chunks,
                config
            )
        elif final_chunks:
            answer = f"Найдено {len(final_chunks)} релевантных фрагментов кода."
        else:
            answer = "Не удалось найти релевантные фрагменты кода для ответа на ваш вопрос."

        llm_usage = None
        if last_response and hasattr(last_response, 'llm_usage'):
            llm_usage = last_response.llm_usage

        total_duration = time.time() - start_time
        self.logger.info(
            f"Agent search completed: "
            f"{len(state.iterations)} iterations, "
            f"{len(state.all_chunks)} total chunks, "
            f"{len(final_chunks)} final chunks, "
            f"duration={total_duration:.2f}s"
        )

        return QueryResponse(
            meta=MetaResponse(
                request_id=request.meta.request_id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                status="done" if final_chunks else "no_relevance_sources",
            ),
            status=status,
            messages=request.query.messages,
            answer=answer,
            sources=final_chunks,
            llm_usage=llm_usage,
        )

    def _build_error_response(
        self,
        request: QueryRequest,
        start_datetime: datetime,
        error_message: str
    ) -> QueryResponse:
        """Построить ответ об ошибке."""
        return QueryResponse(
            meta=MetaResponse(
                request_id=request.meta.request_id,
                start_datetime=start_datetime,
                end_datetime=datetime.now(),
                status="error",
            ),
            status="no_relevance_sources",
            messages=request.query.messages,
            answer=f"Ошибка: {error_message}",
            sources=[],
            llm_usage=None
        )
