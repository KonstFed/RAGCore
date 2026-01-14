import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, HttpUrl, UUID4


class AstChunkerConfig(BaseModel):
    """Конфигурация библиотеки astchunk."""

    max_chunk_size: int = Field(
        1000, description="Максимальное количество токенов/символов в чанке."
    )
    chunk_overlap: int = Field(50, description="Перекрытие между чанками.")
    chunk_expansion: bool = Field(
        True,
        description=(
            "Включает ли astchunk расширение контекста до границ функциональных блоков."
        ),
    )
    metadata_template: str = Field("default")


class TextSplitterConfig(BaseModel):
    """Конфигурация LangChain text splitter"""

    chunk_size: int = Field(1000, description="Максимальный размер чанка в символах.")
    chunk_overlap: int = Field(200, description="Перекрытие между чанками в символах.")
    separators: Optional[List[str]] = Field(
        None,
        description=(
            "Список разделителей для разбиения текста. "
            "По умолчанию используются стандартные разделители LangChain."
        ),
    )


class EmbeddingConfig(BaseModel):
    model_name: Literal[
        "e5-large",
        "qwen3-embedding-0.6b",
        "qwen3-embedding-4b",
        "jina-code-embeddings-1.5b",
    ] = Field("qwen3-embedding-0.6b")
    dimensions: int = Field(1024)
    max_tokens: int = Field(8192)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class MetaRequest(BaseModel):
    """Метаданные запроса."""

    request_id: UUID4 = Field(
        ..., description="Уникальный id запроса (должен совпадать с ответом)."
    )


class MetaResponse(BaseModel):
    """Метаданные ответа."""

    request_id: UUID4 = Field(
        ..., description="Уникальный id ответа (должен совпадать с запросом)."
    )
    start_datetime: datetime = Field(..., description="Время начала обработки.")
    end_datetime: datetime = Field(..., description="Время окончания обработки.")
    status: Literal["error", "done", "timeout"]


class IndexJobStatus(BaseModel):
    status: Optional[
        Literal["failed", "loaded", "parsed", "vectorized", "saved_to_qdrant"]
    ] = None
    chunks_processed: Optional[int] = None
    repo_path: str = None
    description_error: Optional[str] = None


class IndexJobResponse(BaseModel):
    repo_url: HttpUrl = Field(..., example="https://github.com/owner/repo")
    meta: MetaResponse
    job_status: IndexJobStatus


class DeleteResponse(BaseModel):
    """Ответ на запрос удаления индекса."""

    repo_url: HttpUrl = Field(..., example="https://github.com/owner/repo")
    success: bool = Field(..., description="Успешно ли выполнено удаление")
    meta: MetaResponse
    message: Optional[str] = Field(
        None, description="Дополнительное сообщение о результате операции"
    )


class IndexConfig(BaseModel):
    ast_chunker_config: Optional[AstChunkerConfig] = None
    ast_chunker_languages: List[Literal["python", "java", "typescript", "csharp"]] = (
        Field(
            description=(
                "Список языков для AST chunking. "
                "Если указать пустой список, то AST chunking не будет использоваться."
            ),
            default=["python", "java", "typescript", "csharp"],
        )
    )
    text_splitter_config: TextSplitterConfig
    embedding_config: Optional[EmbeddingConfig] = None
    exclude_patterns: Optional[
        List[Literal["tests/", "*.lock", "__pycache__", ".venv", "build"]]
    ] = Field(None, description="Список паттернов .gitignore для исключения файлов.")


class IndexRequest(BaseModel):
    repo_url: HttpUrl = Field(..., example="https://github.com/owner/repo")
    meta: MetaRequest
    config: Optional[IndexConfig] = None


class ChunkMetadata(BaseModel):
    """Метаданные чанка кода."""

    chunk_id: UUID4 = Field(
        default_factory=uuid.uuid4, description="Уникальный id чанка."
    )
    filepath: str = Field(
        ...,
        description="Полный путь к файлу внутри репозитория.",
        example="src/utils/helpers.py",
    )

    chunk_size: Optional[int] = Field(None, description="Размер чанка в символах.")
    line_count: Optional[int] = Field(None, description="Количество строк в чанке.")
    start_line_no: int = Field(
        ..., description="Номер первой строки кода в оригинальном файле."
    )
    end_line_no: int = Field(..., description="Номер последней строки кода.")
    node_count: Optional[int] = Field(
        None, description="Количество AST узлов в чанке (специфично для astchunk)."
    )
    language: Optional[
        Literal["python", "go", "java", "cpp", "javascript", "csharp", "typescript"]
    ] = Field(None, description="Язык программирования.")

    @property
    def file_name(self) -> str:
        return Path(self.filepath).name


class Chunk(BaseModel):
    """Единица кода после обработки."""

    content: str = Field(..., description="Исходный код чанка (сырой текст).")
    metadata: ChunkMetadata
    retrieval_relevance_score: Optional[float] = Field(
        None, description="Оценка релевантности (similarity score) при поиске."
    )
    reranker_relevance_score: Optional[float] = Field(
        None, description="Оценка релевантности (reranker score) при переранжировании."
    )


class LLMGenerationParams(BaseModel):
    """Гиперпараметры для настройки генерации LLM."""

    temperature: float = Field(
        0.1,
        ge=0.0,
        le=2.0,
        description="Степень 'креативности'. 0 - детерминированный ответ.",
    )
    max_tokens: Optional[int] = Field(
        None,
        description=(
            "Максимальное количество токенов в ответе. "
            "Если null, используется лимит модели."
        ),
        example=4096,
    )
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling.")
    frequency_penalty: float = Field(
        0.0, ge=-2.0, le=2.0, description="Штраф за частоту повторения токенов."
    )
    presence_penalty: float = Field(
        0.0, ge=-2.0, le=2.0, description="Штраф за повторное появление токенов."
    )


class LLMConfig(BaseModel):
    """Конфигурация LLM."""

    provider: Literal[
        "openai",
        "azure",
        "mistral",
        "anthropic",
        "google",
        "ollama",
        "vllm",
        "custom",
        "openrouter",
    ] = Field("openrouter", description="Провайдер API")
    model_name: Literal[
        "GigaChat-2-Max",
        "Qwen3-4B-Instruct-2507",
        "mistral-large-latest",
        "openai/gpt-oss-120b",
        "openrouter/anthropic/claude-3.5-sonnet",
    ] = Field(
        "openai/gpt-oss-120b",
        description="Название модели (или deployment name для выбранного провайдера).",
    )
    base_url: Optional[HttpUrl] = Field(
        None,
        description="Базовый URL API. Обязателен для Ollama, vLLM или прокси.",
        example="http://localhost:11434/v1",
    )
    system_prompt: Optional[str] = Field(
        "Ты - ассистент для ответов на вопросы по репозиторию с кодом. "
        "Тебе дан диалог с пользователем и релевантные контексты "
        "по некоторому репозиторию.",
        description="Базовый системный промпт.",
    )
    parameters: Optional[LLMGenerationParams] = None
    extra_options: Optional[Dict[str, Any]] = Field(
        None,
        description="Дополнительные параметры, специфичные для конкретного провайдера.",
        example={"timeout": 60, "retry_count": 3},
    )


class TextSanitizationSettings(BaseModel):
    """Настройки для поиска и замены чувствительных данных."""

    enabled: bool = Field(True, description="Включить или выключить модуль очистки.")
    regex_patterns: Optional[List[str]] = Field(
        None,
        description="Список регулярных выражений. Найденные совпадения будут заменены.",
        example=["^(sk-[a-zA-Z0-9]{48})$"],
    )
    stop_words: Optional[List[str]] = Field(
        None,
        description="Список слов или фраз (case-insensitive), которые нужно скрыть.",
    )
    replacement_token: str = Field(
        "[REDACTED]",
        description="Строка-заглушка, на которую заменяются найденные паттерны.",
    )


class ContentBlockingSettings(BaseModel):
    """Настройки полной замены ответа/запроса при обнаружении нежелательных патернов."""

    enabled: bool = False
    trigger_patterns: Optional[List[str]] = Field(
        None,
        description="Список RegExp. Если найден ХОТЯ БЫ ОДИН, весь текст заменяется.",
    )
    fallback_message: str = Field(
        "Content blocked due to policy violation.",
        description="Текст, который полностью заменит оригинальное сообщение.",
    )


class RegexSubstitutionRule(BaseModel):
    pattern: str = Field(..., description="Регулярное выражение, которое нужно найти.")
    replacement: str = Field(
        ..., description="Строка, на которую нужно заменить найденный паттерн."
    )


class QueryPreprocessorConfig(BaseModel):
    max_length: int = Field(5000)
    normalize_whitespace: bool = Field(True)
    sanitization: Optional[TextSanitizationSettings] = None
    custom_substitutions: Optional[List[RegexSubstitutionRule]] = Field(
        None, description="Список правил для замены части текста по RegExp."
    )
    blacklist: Optional[ContentBlockingSettings] = None


class QueryPostprocessorConfig(BaseModel):
    format_markdown: bool = Field(
        True, description="Убедиться, что ответ является валидным Markdown."
    )
    add_citations: bool = Field(
        False, description="Добавлять ли ссылки на файлы репозитория в конце ответа."
    )
    sanitization: Optional[TextSanitizationSettings] = None
    custom_substitutions: Optional[List[RegexSubstitutionRule]] = Field(
        None, description="Исправление артефактов генерации."
    )
    blacklist: Optional[ContentBlockingSettings] = Field(
        None,
        description="Если модель сгенерировала запрещенный контент, заменить ответ.",
    )


class RewriterTemplates(BaseModel):
    user_prompt_template: str = Field(
        "Диалог с пользователем: {messages}\nНайденные источники:\n{contexts}",
        description="Шаблон запроса с найденными контекстами.",
    )
    context_template: str = Field(
        "Filepath: {metadata.filepath}, start line number: "
        "{metadata.start_line_no}, end line number: "
        "{metadata.end_line_no}\n\n{content}",
        description="Шаблон для подставления полей чанков.",
    )


class QueryRewriterConfig(BaseModel):
    """Настройки модуля генерации финального ответа с помощью LLM."""

    enabled: bool = False
    llm_config: Optional[LLMConfig] = None
    templates: Optional[RewriterTemplates] = None
    max_user_messages: int = Field(
        3, description="Максимальное количество сообщений для учета при переписывании."
    )


class QaConfig(BaseModel):
    """Настройки модуля генерации финального ответа с помощью LLM (QA)."""

    enabled: bool = False
    llm_config: Optional[LLMConfig] = None
    templates: Optional[RewriterTemplates] = None


class FilterCondition(BaseModel):
    """Конечное условие сравнения поля с значением."""

    name: Literal[
        "filepath", "file_name", "language", "chunk_size", "node_count", "start_line_no"
    ]
    value: Union[str, int, float, bool, List[Any]] = Field(
        ..., description="Значение для сравнения."
    )
    operator: Literal[
        "eq", "neq", "gt", "gte", "lt", "lte", "in", "wildcard", "contains"
    ]


class FilterGroup(BaseModel):
    """Логическая группа, объединяющая несколько условий или подгрупп."""

    operator: Literal["and", "or"]
    values: List["FilterNode"]


FilterNode = Union["FilterGroup", "FilterCondition"]


class FilteringConfig(BaseModel):
    """Конфигурация фильтрации поиска."""

    enabled: bool = False
    filter: Optional[FilterNode] = None


class RetrieverConfig(BaseModel):
    """Настройка модуля извлечения релевантных источников."""

    embedding_config: Optional["EmbeddingConfig"] = None
    size: int = Field(10, description="Количество извлекаемых источников.")
    threshold: float = Field(0.0)
    bm25_weight: float = Field(
        0.0, ge=0.0, le=1.0, description="Вес ключевых слов для полнотекстового поиска."
    )


class RerankerConfig(BaseModel):
    """Настройка модуля переранжирования найденных источников."""

    enabled: bool = False
    model_name: Literal["qwen3-reranker-0.6b"] = "qwen3-reranker-0.6b"
    top_k: int = Field(3, description="Количество чанков после переранжирования.")
    threshold: float = Field(0.5)


class ContextExpansionConfig(BaseModel):
    """Настройки расширения найденного контекста."""

    enabled: bool = False
    before_chunk: int = Field(0, description="Количество чанков 'до'.")
    after_chunk: int = Field(0, description="Количество чанков 'после'.")


class SearchConfig(BaseModel):
    query_preprocessor: Optional[QueryPreprocessorConfig] = None
    query_rewriter: Optional[QueryRewriterConfig] = None
    filtering: Optional[FilteringConfig] = None
    retriever: Optional[RetrieverConfig] = None
    reranker: Optional[RerankerConfig] = None
    context_expansion: Optional[ContextExpansionConfig] = None
    qa: Optional[QaConfig] = None
    query_postprocessor: Optional[QueryPostprocessorConfig] = None


class QueryObject(BaseModel):
    messages: List[Message]
    sources: Optional[List[Chunk]] = None


class QueryRequest(BaseModel):
    repo_url: HttpUrl = Field(..., example="https://github.com/owner/repo")
    meta: MetaRequest
    query: QueryObject
    stream: bool = False


class LLMUsageObject(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class QueryResponse(BaseModel):
    meta: MetaResponse
    status: Literal[
        "preprocessor_filtering",
        "postprocessor_filtering",
        "no_relevance_sources",
        "llm_rag",
        "no_llm",
    ]
    messages: List[Message]
    answer: str = Field(..., description="Сгенерированный ответ LLM.")
    sources: List[Chunk] = Field(
        None, description="Список чанков, использованных для генерации ответа."
    )
    llm_usage: LLMUsageObject
