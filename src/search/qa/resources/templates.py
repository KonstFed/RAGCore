DEFAULT_USER_PROMPT_TEMPLATE = (
    """Диалог с пользователем: {messages}\nНайденные источники:\n{contexts}"""
)


DEFAULT_CONTEXT_TEMPLATE = """Filepath: {metadata.filepath}, start line number: {metadata.start_line_no}, end line number: {metadata.end_line_no}\n\n{content}"""  # noqa: E501
