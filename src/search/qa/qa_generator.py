import os
from copy import deepcopy
from src.core.llm import LLMClient
from src.search.qa.resources.prompts import DEFAULT_SYSTEM_PROMPT
from src.search.qa.resources.templates import DEFAULT_USER_PROMPT_TEMPLATE, DEFAULT_CONTEXT_TEMPLATE
from typing import Any, Dict, List, Tuple


class QAGenerator:
    """
    Класс генерации ответа (QA).
    Формирует промпт из чанков и истории сообщений, обращается к LLM.
    """
    def __init__(self):
        pass # TODO реализовать коннект к LLMClient

    async def pipeline(self, request: QueryRequest) -> Dict[str, Any]:
        config = request.search_config.qa if request.search_config else None

        sources = request.query.sources or []
        messages = request.query.messages

        context_str = ""
        if config and config.templates:
            template = config.templates.context_template or DEFAULT_CONTEXT_TEMPLATE

        for chunk in sources:
            # Подстановка полей метаданных в шаблон
            # Внимание: простая реализация f-string подстановки через format
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
            system_prompt = config.llm_config.system_prompt

        llm_messages = [{"role": "system", "content": system_prompt}]

        # Добавляем историю.
        # Обычно контекст добавляется в последнее сообщение пользователя или как отдельное системное.
        # Реализуем добавление контекста в последнее сообщение согласно user_prompt_template

        history = deepcopy(messages)
        last_user_msg = history[-1]

        if config and config.templates:
            user_template = config.templates.user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
            # Преобразуем историю в строку или вставляем контекст
            # В простейшем случае, мы модифицируем последнее сообщение:
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

        response_dict = {
            "answer": response_text,
            "sources": sources,
            "meta": request.meta.dict() if request.meta else {}
        }

        if "request_id" not in response_dict["meta"] and request.meta:
            response_dict["meta"]["request_id"] = request.meta.request_id

        return response_dict
