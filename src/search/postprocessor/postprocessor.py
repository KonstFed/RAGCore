import re
from src.core.schemas import (
    SearchConfig,
    QueryResponse,
    ContentBlockingSettings,
    TextSanitizationSettings,
)
from src.utils.logger import get_logger
from omegaconf import DictConfig


class Postprocessor:
    """
    Класс постобработки ответа.
    Форматирование Markdown, добавление цитат, очистка.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.fallback_message = cfg.postprocessor.fallback_message

    def pipeline(self, response: QueryResponse, config: SearchConfig) -> QueryResponse:
        self.logger.info(
            f"Run postprocessor pipeline for request_id={response.meta.request_id}."
        )
        if not config or not config.query_postprocessor:
            return response

        config = config.query_postprocessor

        answer = response.answer
        sources = response.sources

        # 1. Blacklist filtering
        if config.blacklist and config.blacklist.enabled:
            if self._check_blacklist(answer, config.blacklist):
                response.meta.status = "done"
                response.status = "postprocessor_filtering"
                response.answer = self.fallback_message
                self.logger.warning("Blacklist triggered. Returning filtered response.")
                return response

        # 2. Sanitization (Повторная очистка)
        if config.sanitization and config.sanitization.enabled:
            answer = self._sanitize(answer, config.sanitization)

        # 3. Форматирование Markdown
        if config.format_markdown:
            # Проверка закрытых тегов кода ```
            if answer.count("```") % 2 != 0:
                answer += "\n```"

        # 4. Add citations (Добавление ссылок на источники в конце генеративного ответа)
        if config.add_citations:
            if sources and isinstance(sources, list):
                citations = "\n\n**Sources:**\n"
                unique_files = set()
                for chunk in sources:
                    # chunk может быть объектом Chunk или словарем,
                    # если произошла сериализация
                    if hasattr(chunk, "metadata"):
                        meta = chunk.metadata
                        name = meta.file_name
                        path = meta.filepath
                    elif isinstance(chunk, dict):
                        meta = chunk.get("metadata", {})
                        name = meta.get("file_name", "unknown")
                        path = meta.get("filepath", "unknown")
                    else:
                        continue

                    if path not in unique_files:
                        citations += f"- [{name}]({path})\n"
                        unique_files.add(path)

                answer += citations

        response.answer = answer

        msg = (
            "Successful finished postprocessor pipeline "
            f"for request_id={response.meta.request_id}."
        )
        self.logger.info(msg)
        return response

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
