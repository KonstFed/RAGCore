import os
from copy import deepcopy
from src.core.llm import LLMClient
from src.core.schemas import QueryRequest, SearchConfig
from src.search.qa.resources.prompts import DEFAULT_SYSTEM_PROMPT
from src.search.qa.resources.templates import DEFAULT_USER_PROMPT_TEMPLATE, DEFAULT_CONTEXT_TEMPLATE
from typing import Any, Dict, List, Tuple
from src.utils.logger import get_logger


class QAGenerator:
    """
    Класс генерации ответа (QA).
    Формирует промпт из чанков и истории сообщений, обращается к LLM.
    """
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        pass # TODO реализовать коннект к LLMClient

    async def pipeline(self, request: QueryRequest, config: SearchConfig) -> Dict[str, Any]:
        self.logger.info("Run qa pipeline.")
        config = config.qa if config else None

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
            system_prompt = config.llm_config.system_prompt or DEFAULT_SYSTEM_PROMPT

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
            "sources": [ # TODO временный костыль для `sources`, чтобы тестировать смотреть UI
                {
                    "content": "import numpy as np\nimport pandas as pd\nimport json\n\nclass Car:\n\tdef __init__(self, wheels_count: int, brand: str):\n\t\tself.wheels = wheels_count\n",
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
                    "content": "class Car:\n\tdef __init__(self, wheels_count: int, brand: str):\n\t\tself.brand = brand\n\t\tself.sound = 'bibip'\n\t\t\n\n\tdef bibip(self):\n\t\tprint(self.sound)\n",
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
                },
            ],
            "meta": request.meta.dict() if request.meta else {},
            "messages": history,
            "llm_usage": {
                "prompt_tokens": 100, "completion_tokens": 180
            }
        }

        if "request_id" not in response_dict["meta"] and request.meta:
            response_dict["meta"]["request_id"] = request.meta.request_id

        self.logger.info("Successful finished qa pipeline.")
        return response_dict
