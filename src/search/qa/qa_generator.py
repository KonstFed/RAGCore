from copy import deepcopy
from datetime import datetime
from omegaconf import DictConfig
from src.core.llm import LLMClient
from src.core.schemas import QueryRequest, QueryResponse, SearchConfig
from src.search.qa.resources.prompts import DEFAULT_SYSTEM_PROMPT
from src.search.qa.resources.templates import (
    DEFAULT_USER_PROMPT_TEMPLATE,
    DEFAULT_CONTEXT_TEMPLATE,
)
from src.utils.logger import get_logger


class QAGenerator:
    """
    Класс генерации ответа (QA).
    Формирует промпт из чанков и истории сообщений, обращается к LLM.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.fallback_message = cfg.qa.fallback_message
        self.llm_client = LLMClient(cfg)
        # дефолтная конфигурация LLM (из service config)
        self.default_llm_config = getattr(cfg, "llm", None)

    async def pipeline(
        self, request: QueryRequest, config: SearchConfig
    ) -> QueryResponse:
        self.logger.info(f"Run qa pipeline for request_id={request.meta.request_id}.")

        config = config.qa if config else None
        if not config or not config.enabled:
            response_dict = {
                "meta": {
                    "request_id": request.meta.request_id,
                    "start_datetime": datetime.now(),  # будет перезаписано
                    "end_datetime": datetime.now(),  # будет перезаписано
                    "status": "done",
                },
                "status": "no_llm",
                "messages": request.query.messages,
                "answer": self.fallback_message,
                "sources": request.query.sources,
                "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
            msg = (
                "Finished qa pipeline because not config or enabled=false"
                f"for request_id={request.meta.request_id}."
            )
            self.logger.warning(msg)
            return QueryResponse(**response_dict)

        sources = request.query.sources or []
        messages = request.query.messages

        context_str = ""
        if config and config.templates:
            template = config.templates.context_template or DEFAULT_CONTEXT_TEMPLATE

        for chunk in sources:
            try:
                formatted_chunk = template.format(
                    content=chunk.content, metadata=chunk.metadata
                )
                context_str += formatted_chunk + "\n---\n"
            except Exception:
                context_str += f"{chunk.content}\n---\n"

        system_prompt = DEFAULT_SYSTEM_PROMPT
        if config and config.llm_config and config.llm_config.system_prompt:
            system_prompt = config.llm_config.system_prompt or DEFAULT_SYSTEM_PROMPT

        llm_messages = [{"role": "system", "content": system_prompt}]

        history = deepcopy(messages)
        last_user_msg = history[-1]

        if config and config.templates:
            user_template = (
                config.templates.user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
            )
            combined_content = user_template.format(
                messages=last_user_msg.content, contexts=context_str
            )
            last_user_msg.content = combined_content
        else:
            last_user_msg.content = (
                f"Context:\n{context_str}\n\nQuestion: {last_user_msg.content}"
            )

        for msg in history:
            llm_messages.append({"role": msg.role, "content": msg.content})

        # Debug: логируем, что уйдет в LLM (урезаем, чтобы не засорять логи)
        try:
            sys_preview = system_prompt[:400]
            user_preview = last_user_msg.content[:400]
            self.logger.debug(
                f"QA prompt preview for request_id={request.meta.request_id}: "
                f"system[{len(system_prompt)}]={sys_preview!r}, "
                f"user[{len(last_user_msg.content)}]={user_preview!r}, "
                f"sources={len(sources)}"
            )
        except Exception:
            pass

        # вызываем LLM
        try:
            llm_cfg = None
            if config and config.llm_config:
                llm_cfg = config.llm_config
            elif self.default_llm_config:
                llm_cfg = self.default_llm_config

            response_text, llm_usage = self.llm_client.agenerate(llm_messages, llm_cfg)
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            response_text = self.fallback_message
            llm_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        response_dict = {
            "meta": {
                "request_id": request.meta.request_id,
                "start_datetime": datetime.now(),  # будет перезаписано
                "end_datetime": datetime.now(),  # будет перезаписано
                "status": "done",
            },
            "status": "llm_rag",
            "messages": request.query.messages,
            "answer": response_text,
            "sources": sources,
            "llm_usage": llm_usage,
        }
        self.logger.info(
            f"Successful finished qa pipeline for request_id={request.meta.request_id}."
        )
        return QueryResponse(**response_dict)
