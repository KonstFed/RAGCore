from uuid import uuid4

import gradio as gr

from src.assistant import Assistant


assistant = Assistant(service_cfg_path="configs/deployment_config.yaml")


def _build_delete_request(repo_url: str) -> dict:
    return {
        "meta": {"request_id": str(uuid4())},
        "repo_url": repo_url,
    }


def _build_index_request(repo_url: str) -> tuple[dict, dict]:
    request = {
        "meta": {"request_id": str(uuid4())},
        "repo_url": repo_url,
        "branch": "main",
    }

    config = {
        "ast_chunker_config": {
            "max_chunk_size": 1000,
            "chunk_overlap": 50,
            "extensions": [
                ".py",
                ".ipynb",
                ".cpp",
                ".h",
                ".java",
                ".ts",
                ".tsx",
                ".cs",
            ],
            "chunk_expansion": True,
            "metadata_template": "default",
        },
        "text_splitter_config": {"chunk_size": 500, "chunk_overlap": 50},
        "exclude_patterns": ["*.lock", "__pycache__", ".venv", "build"],
    }

    return request, config


def _build_search_config() -> dict:
    return {
        "query_preprocessor": {
            "enabled": True,
            "normalize_whitespace": True,
            "sanitization": {
                "enabled": True,
                "regex_patterns": ["jailbreak", "hallucinations"],
                "replacement_token": ""
            }
        },
        "query_rewriter": {"enabled": False},
        "retriever": {"enabled": True},
        "filtering": {"enabled": True},
        "reranker": {"enabled": False},
        "context_expansion": {"enabled": True},
        "qa": {"enabled": True},
        "query_postprocessor": {
            "enabled": True,
            "format_markdown": True,
            "sanitization": {
                "enabled": True,
                "regex_patterns": ["can't", "wtf"],
                "replacement_token": ""
            }
        }
    }


def _build_agent_config(
    max_iterations: int = 5,
    max_time_seconds: float = 120.0,
    confidence_threshold: float = 0.7,
    min_relevant_chunks: int = 3,
    relevance_score_threshold: float = 0.5,
    enable_query_refinement: bool = True,
    enable_filter_adjustment: bool = True,
    enable_retriever_adjustment: bool = True,
    generate_final_answer: bool = True,
    use_llm: bool = True,
    llm_model: str = "openai/gpt-oss-120b"
) -> dict:
    """Построить конфигурацию агента для deep research."""
    config = {
        "max_iterations": max_iterations,
        "max_time_seconds": max_time_seconds,
        "confidence_threshold": confidence_threshold,
        "min_relevant_chunks": min_relevant_chunks,
        "relevance_score_threshold": relevance_score_threshold,
        "enable_query_refinement": enable_query_refinement,
        "enable_filter_adjustment": enable_filter_adjustment,
        "enable_retriever_adjustment": enable_retriever_adjustment,
        "generate_final_answer": generate_final_answer
    }

    if use_llm:
        config["llm_config"] = {
            "provider": "openrouter",
            "model_name": llm_model,
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 4096
            }
        }

    # Initial Search Engine configuration
    config["initial_search_config"] = {
        "retriever": {
            "size": 10,
            "threshold": 0.3,
            "bm25_weight": 0.3
        },
        "reranker": {
            "enabled": True,
            "top_k": 5,
            "threshold": 0.4
        },
        "qa": {"enabled": False}
    }

    return config


async def index_repo(repo_url: str) -> str:
    if not repo_url:
        return "❌ **Ошибка:** Введите GitHub URL."

    request, config = _build_index_request(repo_url)

    try:
        response = await assistant.index(request, config)

        # Calculate duration
        duration = (
            response.meta.end_datetime - response.meta.start_datetime
        ).total_seconds()

        # Build verbose response
        result = []
        result.append("## 📊 Результат индексации\n")
        result.append(f"**Request ID:** `{response.meta.request_id}`\n")
        result.append(f"**Repository URL:** {response.repo_url}\n")
        result.append(f"**Время выполнения:** {duration:.2f} секунд\n")
        result.append(f"**Статус:** {response.meta.status}\n")

        # Check if repo was already indexed
        is_already_indexed = (
            response.job_status.description_error
            and "already indexed" in response.job_status.description_error.lower()
        )

        if is_already_indexed:
            result.append("\n⚠️ **Репозиторий уже проиндексирован**\n")
            result.append(
                "Индексация была пропущена, так как репозиторий "
                "уже существует в базе данных.\n"
            )
        else:
            # Show job status details
            if response.job_status.status:
                status_emoji = {
                    "failed": "❌",
                    "loaded": "📥",
                    "parsed": "🔍",
                    "vectorized": "🧮",
                    "saved_to_qdrant": "✅",
                }
                emoji = status_emoji.get(response.job_status.status, "ℹ️")
                result.append(
                    f"\n**Статус задачи:** {emoji} {response.job_status.status}\n"
                )

            # Show chunks processed
            if response.job_status.chunks_processed is not None:
                result.append(
                    f"**Обработано чанков:** {response.job_status.chunks_processed}\n"
                )

            # Show errors if any
            if response.meta.status == "error":
                result.append("\n### ❌ Ошибка при индексации\n")
                if response.job_status.description_error:
                    result.append(
                        f"**Описание ошибки:**\n```\n"
                        f"{response.job_status.description_error}\n```\n"
                    )
                else:
                    result.append("Произошла ошибка во время индексации.\n")
            elif response.job_status.status == "saved_to_qdrant":
                result.append("\n### ✅ Индексация завершена успешно\n")
                result.append(
                    "Репозиторий успешно проиндексирован и сохранен "
                    "в векторную базу данных.\n"
                )

        return "".join(result)

    except Exception as e:
        return f"❌ **Критическая ошибка:** {type(e).__name__}: {str(e)}"


async def delete_index(repo_url: str) -> str:
    if not repo_url:
        return "❌ **Ошибка:** Введите GitHub URL."

    request = _build_delete_request(repo_url)

    try:
        response = await assistant.delete_index(request)

        # Calculate duration
        duration = (
            response.meta.end_datetime - response.meta.start_datetime
        ).total_seconds()

        # Build verbose response
        result = []
        result.append("## 🗑️ Результат удаления индекса\n")
        result.append(f"**Request ID:** `{response.meta.request_id}`\n")
        result.append(f"**Repository URL:** {response.repo_url}\n")
        result.append(f"**Время выполнения:** {duration:.2f} секунд\n")
        result.append(f"**Статус:** {response.meta.status}\n")

        if response.success:
            result.append("\n### ✅ Удаление завершено успешно\n")
            result.append(
                "Индекс репозитория успешно удален из векторной базы данных.\n"
            )
            if response.message:
                result.append(f"\n**Сообщение:** {response.message}\n")
        else:
            result.append("\n### ❌ Ошибка при удалении\n")
            if response.message:
                result.append(f"**Описание ошибки:**\n```\n{response.message}\n```\n")
            else:
                result.append("Произошла ошибка во время удаления индекса.\n")

        return "".join(result)

    except Exception as e:
        return f"❌ **Критическая ошибка:** {type(e).__name__}: {str(e)}"


def _collect_sources(response) -> list[dict]:
    sources = []
    if getattr(response, "sources", None):
        for source in response.sources:
            sources.append(
                {
                    "filepath": source.metadata.filepath,
                    "language": source.metadata.language or "",
                    "content": source.content,
                }
            )
    return sources


def _render_sources(sources: list[dict], show_sources: bool) -> str:
    if not sources:
        return "Источники:\n- не найдено\n"

    sources_md = "Источники:\n"
    for source in sources:
        sources_md += f"- {source['filepath']}\n"
        if show_sources:
            sources_md += f"\n```{source['language']}\n"
            sources_md += f"{source['content']}\n"
            sources_md += "```\n"
    return sources_md


def _render_sources_detailed(sources: list[dict], show_content: bool) -> str:
    """Рендерить источники с дополнительной информацией для агента."""
    if not sources:
        return "### 📚 Источники\n\n*Источники не найдены*\n"

    sources_md = "### 📚 Найденные источники\n\n"
    for i, source in enumerate(sources, 1):
        score = source.get("reranker_score") or source.get("retrieval_score") or 0
        sources_md += f"**{i}. `{source['filepath']}`**\n"
        sources_md += f"   - Строки: {source.get('start_line', '?')}-{source.get('end_line', '?')}\n"
        sources_md += f"   - Язык: {source.get('language') or 'не определен'}\n"
        sources_md += f"   - Релевантность: {score:.3f}\n"

        if show_content:
            sources_md += f"\n```{source.get('language', '')}\n"
            sources_md += f"{source['content']}\n"
            sources_md += "```\n"
        sources_md += "\n"

    return sources_md


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _normalize_history(history: list[dict] | None) -> list[dict]:
    """Ensure history is list of {'role': str, 'content': str}."""
    if not history:
        return []
    out = []
    for m in history:
        if isinstance(m, dict) and "role" in m:
            out.append(
                {
                    "role": str(m.get("role", "")),
                    "content": _content_to_text(m.get("content")),
                }
            )
    return out


def _last_pairs(history: list[dict], pairs: int = 3) -> list[dict]:
    """Keep last N user+assistant pairs = 2*N messages."""
    max_msgs = 2 * pairs
    return history if len(history) <= max_msgs else history[-max_msgs:]


async def chat(
    repo_url: str,
    message: str,
    show_sources: bool,
    history_state: list[dict],
    chatbot_history: list[dict],
):
    history_state = _normalize_history(history_state)
    chatbot_history = _normalize_history(chatbot_history)

    if not repo_url:
        return (
            "Введите URL репозитория.",
            "Источники:\n- не найдено\n",
            [],
            history_state,
            chatbot_history,
        )

    if not message:
        return (
            "Введите вопрос.",
            "Источники:\n- не найдено\n",
            [],
            history_state,
            chatbot_history,
        )

    # Backend context: last 3 Q/A pairs (6 msgs) + new question
    context_messages = _last_pairs(history_state, pairs=3)
    request_messages = context_messages + [{"role": "user", "content": message}]

    request = {
        "meta": {"request_id": str(uuid4())},
        "query": {"messages": request_messages},
        "repo_url": repo_url,
    }
    config = _build_search_config()

    try:
        response = await assistant.query(request, config)
        answer_text = (getattr(response, "answer", "") or "").strip()
    except Exception as e:
        return (
            f"Ошибка: {type(e).__name__}: {e}",
            "Источники:\n- не найдено\n",
            [],
            history_state,
            chatbot_history,
        )

    final_answer = answer_text or "Ответ пуст."

    sources = _collect_sources(response)
    sources_md = _render_sources(sources, show_sources)

    chatbot_history = chatbot_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": final_answer},
    ]

    new_history_state = request_messages + [
        {"role": "assistant", "content": final_answer}
    ]
    new_history_state = _last_pairs(new_history_state, pairs=3)

    return sources_md, sources, new_history_state, chatbot_history


def update_sources(show_sources: bool, sources: list[dict]):
    return _render_sources(sources or [], show_sources)


async def agent_research(
    repo_url: str,
    question: str,
    max_iterations: int,
    max_time_seconds: float,
    confidence_threshold: float,
    min_relevant_chunks: int,
    relevance_score_threshold: float,
    enable_query_refinement: bool,
    enable_filter_adjustment: bool,
    enable_retriever_adjustment: bool,
    generate_final_answer: bool,
    use_llm: bool,
    llm_model: str,
    show_sources_content: bool
):
    """Выполнить углубленный агентский поиск по репозиторию."""

    if not repo_url:
        return (
            "❌ **Ошибка:** Введите GitHub URL репозитория.",
            "",
            []
        )

    if not question:
        return (
            "❌ **Ошибка:** Введите вопрос для исследования.",
            "",
            []
        )

    # Формируем запрос
    request = {
        "meta": {"request_id": str(uuid4())},
        "query": {"messages": [{"role": "user", "content": question}]},
        "repo_url": repo_url,
    }

    # Формируем конфигурацию агента
    config = _build_agent_config(
        max_iterations=max_iterations,
        max_time_seconds=max_time_seconds,
        confidence_threshold=confidence_threshold,
        min_relevant_chunks=min_relevant_chunks,
        relevance_score_threshold=relevance_score_threshold,
        enable_query_refinement=enable_query_refinement,
        enable_filter_adjustment=enable_filter_adjustment,
        enable_retriever_adjustment=enable_retriever_adjustment,
        generate_final_answer=generate_final_answer,
        use_llm=use_llm,
        llm_model=llm_model
    )

    try:
        response = await assistant.deep_research(request, config)

        duration = (
            response.meta.end_datetime - response.meta.start_datetime
        ).total_seconds()

        result_parts = []
        result_parts.append("## 🔬 Результат углубленного исследования\n\n")
        result_parts.append(f"**Request ID:** `{response.meta.request_id}`\n\n")
        result_parts.append(f"**Статус:** {response.meta.status}\n\n")
        result_parts.append(f"**Время выполнения:** {duration:.2f} сек.\n\n")
        result_parts.append(f"**Режим:** {response.status}\n\n")
        result_parts.append("---\n\n")
        result_parts.append("### 💡 Ответ\n\n")
        result_parts.append(response.answer or "*Ответ не сгенерирован*")
        result_parts.append("\n")

        result_md = "".join(result_parts)

        sources = _collect_sources(response)
        sources_md = _render_sources_detailed(sources, show_sources_content)

        return result_md, sources_md, sources

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return (
            f"❌ **Критическая ошибка:** {type(e).__name__}: {str(e)}\n\n"
            f"```\n{error_trace}\n```",
            "",
            []
        )


def update_agent_sources(show_content: bool, sources: list[dict]):
    """Обновить отображение источников агента."""
    return _render_sources_detailed(sources or [], show_content)


with gr.Blocks(title="RAGCode") as demo:
    gr.Markdown("# RAGCode")

    with gr.Tabs():
        with gr.Tab("🗂️ Индексировать репозиторий"):
            repo_url_input = gr.Textbox(label="GitHub URL")
            with gr.Row():
                index_button = gr.Button("Индексировать", variant="primary")
                delete_button = gr.Button("Удалить индекс", variant="stop")
            index_status = gr.Markdown()
            index_button.click(index_repo, inputs=repo_url_input, outputs=index_status)
            delete_button.click(
                delete_index, inputs=repo_url_input, outputs=index_status
            )

        with gr.Tab("💬 Чат по коду"):
            chat_repo_url = gr.Textbox(label="URL репозитория")

            chatbot = gr.Chatbot(label="История", height=420)

            sources = gr.Markdown("Источники:\n- не найдено\n")
            message_input = gr.Textbox(label="Ваш вопрос")
            show_sources = gr.Checkbox(
                label="Показывать содержимое источников", value=False
            )

            sources_state = gr.State([])
            history_state = gr.State([])

            send_button = gr.Button("Спросить")

            send_button.click(
                chat,
                inputs=[
                    chat_repo_url,
                    message_input,
                    show_sources,
                    history_state,
                    chatbot,
                ],
                outputs=[sources, sources_state, history_state, chatbot],
            )

            show_sources.change(
                update_sources,
                inputs=[show_sources, sources_state],
                outputs=[sources],
            )

        with gr.Tab("🔬 Агентский поиск"):
            gr.Markdown("""
            ## Углубленный агентский поиск
            
            Агент выполняет итеративный поиск по репозиторию с автоматическим 
            анализом результатов, переформулированием запросов и настройкой параметров 
            для достижения оптимальных результатов.
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    agent_repo_url = gr.Textbox(
                        label="GitHub URL репозитория",
                        placeholder="https://github.com/owner/repo",
                    )
                    agent_question = gr.Textbox(
                        label="Вопрос для исследования",
                        placeholder="Как реализована аутентификация в этом проекте?",
                        lines=3,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Параметры агента")

                    with gr.Accordion("Основные настройки", open=True):
                        agent_max_iterations = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Макс. итераций",
                            info="Максимальное количество итераций поиска",
                        )
                        agent_max_time = gr.Slider(
                            minimum=10,
                            maximum=300,
                            value=120,
                            step=10,
                            label="Макс. время (сек)",
                            info="Таймаут выполнения агента",
                        )
                        agent_confidence = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.05,
                            label="Порог уверенности",
                            info="Минимальная уверенность для остановки",
                        )

                    with gr.Accordion("Настройки релевантности", open=False):
                        agent_min_chunks = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=3,
                            step=1,
                            label="Мин. релевантных чанков",
                        )
                        agent_relevance_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Порог релевантности чанка",
                        )

                    with gr.Accordion("Разрешения агента", open=False):
                        agent_enable_query_refinement = gr.Checkbox(
                            label="Переформулирование запроса",
                            value=True,
                            info="Разрешить агенту изменять запрос",
                        )
                        agent_enable_filter_adjustment = gr.Checkbox(
                            label="Настройка фильтров",
                            value=True,
                            info="Разрешить агенту менять фильтры",
                        )
                        agent_enable_retriever_adjustment = gr.Checkbox(
                            label="Настройка ретривера",
                            value=True,
                            info="Разрешить агенту менять параметры поиска",
                        )
                        agent_generate_answer = gr.Checkbox(
                            label="Генерировать финальный ответ",
                            value=True,
                            info="Использовать LLM для генерации ответа",
                        )

                    with gr.Accordion("Настройки LLM", open=False):
                        agent_use_llm = gr.Checkbox(
                            label="Использовать LLM для анализа",
                            value=True,
                            info="Если выключено, используются эвристики",
                        )
                        agent_llm_model = gr.Dropdown(
                            choices=[
                                "openai/gpt-oss-120b",
                                "openrouter/anthropic/claude-3.5-sonnet",
                                "mistral-large-latest",
                                "GigaChat-2-Max",
                            ],
                            value="openai/gpt-oss-120b",
                            label="Модель LLM",
                        )

            agent_run_button = gr.Button(
                "🚀 Запустить исследование",
                variant="primary",
                size="lg",
            )

            gr.Markdown("---")

            with gr.Row():
                with gr.Column(scale=2):
                    agent_result = gr.Markdown(
                        value="*Результаты исследования появятся здесь...*",
                        label="Результат",
                    )

                with gr.Column(scale=1):
                    agent_show_sources_content = gr.Checkbox(
                        label="Показывать содержимое источников",
                        value=False,
                    )
                    agent_sources = gr.Markdown(
                        value="### 📚 Источники\n\n*Источники появятся после выполнения поиска*",
                        label="Источники",
                    )

            agent_sources_state = gr.State([])

            agent_run_button.click(
                agent_research,
                inputs=[
                    agent_repo_url,
                    agent_question,
                    agent_max_iterations,
                    agent_max_time,
                    agent_confidence,
                    agent_min_chunks,
                    agent_relevance_threshold,
                    agent_enable_query_refinement,
                    agent_enable_filter_adjustment,
                    agent_enable_retriever_adjustment,
                    agent_generate_answer,
                    agent_use_llm,
                    agent_llm_model,
                    agent_show_sources_content,
                ],
                outputs=[
                    agent_result,
                    agent_sources,
                    agent_sources_state,
                ],
            )

            agent_show_sources_content.change(
                update_agent_sources,
                inputs=[agent_show_sources_content, agent_sources_state],
                outputs=[agent_sources],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8501)
