from typing import Any, Dict, List, Tuple
from src.core.schema import (
    QueryRequest,
    ContentBlockingSettings,
    TextSanitizationSettings
)


class Preprocessor:
    """
    Класс предварительной обработки запроса.
    Отвечает за нормализацию текста, удаление чувствительных данных и блокировку запрещенного контента.
    """
    def __init__(self):
        pass # TODO реализовать инит - подтягивание словарей, default регулярок и т.д.

    def pipeline(self, request: QueryRequest) -> QueryRequest:
        if not request.search_config or not request.search_config.query_preprocessor:
            return request

        config = request.search_config.query_preprocessor

        last_message = request.query.messages[-1]
        content = last_message.content

        # 1. Blacklist check
        if config.blacklist and config.blacklist.enabled:
            if self._check_blacklist(content, config.blacklist):
                # Если сработал blacklist, заменяем контент на fallback и помечаем запрос
                # В реальной системе тут можно кидать исключение, но по схеме мы меняем контент
                last_message.content = config.blacklist.fallback_message
                return request

        # 2. Whitespace normalization
        if config.normalize_whitespace:
            content = " ".join(content.split())

        # 3. Max length crop
        if config.max_length and len(content) > config.max_length:
            content = content[:config.max_length]

        # 4. Custom substitutions
        if config.custom_substitutions:
            for rule in config.custom_substitutions:
                content = re.sub(rule.pattern, rule.replacement, content)

        # 5. Sanitization (PII removal)
        if config.sanitization and config.sanitization.enabled:
            content = self._sanitize(content, config.sanitization)

        last_message.content = content
        request.query.messages[-1] = last_message

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
