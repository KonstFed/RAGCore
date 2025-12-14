import os
from copy import deepcopy
from datetime import datetime
from omegaconf import DictConfig
from src.core.llm import LLMClient
from src.core.schemas import QueryRequest, QueryResponse, SearchConfig
from src.search.qa.resources.prompts import DEFAULT_SYSTEM_PROMPT
from src.search.qa.resources.templates import DEFAULT_USER_PROMPT_TEMPLATE, DEFAULT_CONTEXT_TEMPLATE
from typing import Any, Dict, List, Tuple
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

    async def pipeline(self, request: QueryRequest, config: SearchConfig) -> QueryResponse:
        self.logger.info(f"Run qa pipeline for request_id={request.meta.request_id}.")

        config = config.qa if config else None
        if not config or config.enabled == False:
            response_dict = {
                "meta": {
                    "request_id": request.meta.request_id,
                    "start_datetime": datetime.now(), # будет перезаписано
                    "end_datetime": datetime.now(), # будет перезаписано
                    "status": "done"
                },
                "status": "no_llm",
                "messages": request.query.messages,
                "answer": self.fallback_message,
                "sources": request.query.sources,
                "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }
            self.logger.warning(f"Finished qa pipeline because not config or enabled=false for request_id={request.meta.request_id}.")
            return QueryResponse(**response_dict)

        sources = request.query.sources or []
        messages = request.query.messages

        context_str = ""
        if config and config.templates:
            template = config.templates.context_template or DEFAULT_CONTEXT_TEMPLATE

        for chunk in sources:
            try:
                formatted_chunk = template.format(
                    content=chunk.content,
                    metadata=chunk.metadata
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
            user_template = config.templates.user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
            combined_content = user_template.format(
                messages=last_user_msg.content,
                contexts=context_str
            )
            last_user_msg.content = combined_content
        else:
            last_user_msg.content = f"Context:\n{context_str}\n\nQuestion: {last_user_msg.content}"

        for msg in history:
            llm_messages.append({"role": msg.role, "content": msg.content})

        # TODO реализовать вызов LLM
        # response_text = await llm_client.agenerate(llm_messages, config.llm_config)
        # включая заполнение параметра llm_usage
        response_text = f"Это сгенерированный ответ на вопрос по {len(sources)} файлам."
        llm_usage = {"prompt_tokens": 0, "completion_tokens": 0}

        response_dict = {
            "meta": {
                "request_id": request.meta.request_id,
                "start_datetime": datetime.now(), # будет перезаписано
                "end_datetime": datetime.now(), # будет перезаписано
                "status": "done"
            },
            "status": "llm_rag",
            "messages": request.query.messages,
            "answer": response_text,
            "sources": sources,
            "llm_usage": llm_usage
        }
        self.logger.info(f"Successful finished qa pipeline for request_id={request.meta.request_id}.")
        return QueryResponse(**response_dict)
