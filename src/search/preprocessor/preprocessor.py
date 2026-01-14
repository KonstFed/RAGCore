import re
from datetime import datetime
from typing import Union
from omegaconf import DictConfig
from src.core.schemas import (
    QueryRequest,
    QueryResponse,
    SearchConfig,
    ContentBlockingSettings,
    TextSanitizationSettings,
)
from src.utils.logger import get_logger


class Preprocessor:
    """
    Класс предварительной обработки запроса.

    Отвечает за нормализацию текста, удаление чувствительных данных
    и блокировку запрещенного контента.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.fallback_message = cfg.preprocessor.fallback_message

    def pipeline(
        self, request: QueryRequest, config: SearchConfig
    ) -> Union[QueryRequest, QueryResponse]:
        if not config or not config.query_preprocessor:
            return request
        self.logger.info(
            f"Run preprocessor pipeline for request_id={request.meta.request_id}."
        )

        config = config.query_preprocessor

        last_message = request.query.messages[-1]
        content = last_message.content

        # 1. Blacklist check
        if config.blacklist and config.blacklist.enabled:
            if self._check_blacklist(content, config.blacklist):
                response_dict = {
                    "meta": {
                        "request_id": request.meta.request_id,
                        "start_datetime": datetime.now(),  # будет перезаписано
                        "end_datetime": datetime.now(),  # будет перезаписано
                        "status": "done",
                    },
                    "status": "preprocessor_filtering",
                    "messages": request.query.messages,
                    "answer": config.blacklist.fallback_message
                    or self.fallback_message,
                    "sources": [],
                    "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }
                self.logger.warning("Blacklist triggered. Returning filtered response.")
                return QueryResponse(**response_dict)

        # 2. Whitespace normalization
        if config.normalize_whitespace:
            content = " ".join(content.split())

        # 3. Max length crop
        if config.max_length and len(content) > config.max_length:
            content = content[: config.max_length]

        # 4. Custom substitutions
        if config.custom_substitutions:
            for rule in config.custom_substitutions:
                content = re.sub(rule.pattern, rule.replacement, content)

        # 5. Sanitization (PII removal)
        if config.sanitization and config.sanitization.enabled:
            content = self._sanitize(content, config.sanitization)

        last_message.content = content
        request.query.messages[-1] = last_message
        msg = (
            "Successful finished preprocessor pipeline "
            f"for request_id={request.meta.request_id}."
        )
        self.logger.info(msg)
        return request

    def _check_blacklist(self, text: str, settings: ContentBlockingSettings) -> bool:
        if not settings.trigger_patterns:
            return False
        for pattern in settings.trigger_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _sanitize(self, text: str, settings: TextSanitizationSettings) -> str:
        if settings.regex_patterns:
            for pattern in settings.regex_patterns:
                text = re.sub(pattern, settings.replacement_token, text)

        if settings.stop_words:
            for word in settings.stop_words:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                text = pattern.sub(settings.replacement_token, text)
        return text
