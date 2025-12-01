from src.core.schemas import SearchConfig
from typing import Any, Dict, List, Tuple
from src.utils.logger import get_logger


class Postprocessor:
    """
    Класс постобработки ответа.
    Форматирование Markdown, добавление цитат, очистка.
    """
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        pass # TODO реализовать подтягивание дефолтных RegExp и словарей

    def pipeline(self, response: Dict[str, Any], config: SearchConfig) -> Dict[str, Any]:
        self.logger.info("Run postprocessor pipeline.")
        if not config or not config.query_postprocessor:
            return response

        config = config.query_postprocessor

        answer = response.get("answer", "")
        sources = response.get("sources", [])

        # 1. Blacklist filtering
        if config.blacklist and config.blacklist.enabled:
            pass # TODO реализовать

        # 2. Sanitization (Повторная очистка)
        if config.sanitization and config.sanitization.enabled:
            pass # TODO реаизовать

        if config.format_markdown:
            # 3. Форматирование Markdown
            # Проверка закрытых тегов кода ```
            if answer.count("```") % 2 != 0:
                answer += "\n```"

        if config.add_citations:
            # 4. Add citations (Добавление ссылок на источники в конце генеративного ответа)
            if sources and isinstance(sources, list):
                citations = "\n\n**Sources:**\n"
                unique_files = set()
                for chunk in sources:
                    # chunk может быть объектом Chunk или словарем, если произошла сериализация
                    if hasattr(chunk, 'metadata'):
                        meta = chunk.metadata
                        name = meta.file_name
                        path = meta.filepath
                    elif isinstance(chunk, dict):
                        meta = chunk.get('metadata', {})
                        name = meta.get('file_name', 'unknown')
                        path = meta.get('filepath', 'unknown')
                    else:
                        continue
                    
                    if path not in unique_files:
                        citations += f"- [{name}]({path})\n"
                        unique_files.add(path)

                answer += citations

        response["answer"] = answer

        self.logger.info("Successful finished postprocessor pipeline.")
        return response
