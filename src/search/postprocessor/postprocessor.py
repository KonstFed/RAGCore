from typing import Any, Dict, List, Tuple


class Postprocessor:
    """
    Класс постобработки ответа.
    Форматирование Markdown, добавление цитат, очистка.
    """
    def __init__(self):
        pass # TODO реализовать подтягивание дефолтных RegExp и словарей

    def pipeline(self, response: Dict[str, Any]) -> Dict[str, Any]:
        answer = response.get("answer", "")
        sources = response.get("sources", [])

        # TODO столкнулся с проблемой, что в Posprocessor надо передавать search_config
        # Есть три пути:
        #   1. вынести search_config вообще их QueryRequest в отдельную сущность
        #   2. справиться через Dict, qa -> postprocessor, закинув туда доп. параметром search_config,
        #      а потом в postprocessor аккуратно парсить это
        #   3. В QueryResponse сделать Optional meta (end_datetime / start_datetime / status),
        #      конвертнуть из qa -> QueryResponse -> postprocessor --> end

        # 1. Sanitization (Повторная очистка)
        # Если бы был конфиг: if config.sanitization...
        if "[REDACTED]" not in answer:
             # Пример простой логики
             pass

        # 2. Форматирование Markdown
        # Проверка закрытых тегов кода ```
        if answer.count("```") % 2 != 0:
            answer += "\n```"

        # 3. Add citations (Добавление ссылок на источники)
        # (Предположим, это включено)
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
        return response