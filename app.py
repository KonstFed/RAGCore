from uuid import uuid4

import gradio as gr

from src.assistant import Assistant


assistant = Assistant(service_cfg_path="configs/deployment_config.yaml")


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
                "replacement_token": "",
            },
        },
        "query_rewriter": {"enabled": False},
        "retriever": {"enabled": True},
        "filtering": {"enabled": True},
        "reranker": {"enabled": True},
        "context_expansion": {"enabled": True},
        "qa": {"enabled": True},
        "query_postprocessor": {
            "enabled": True,
            "format_markdown": True,
            "sanitization": {
                "enabled": True,
                "regex_patterns": ["can't", "wtf"],
                "replacement_token": "",
            },
        },
    }


async def index_repo(repo_url: str) -> str:
    if not repo_url:
        return "Введите GitHub URL."

    request, config = _build_index_request(repo_url)
    response = await assistant.index(request, config)
    return (
        f"Репозиторий с request_id={response.meta.request_id} "
        f"в статусе '{response.job_status.status}'"
    )


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
            out.append({"role": str(m.get("role", "")), "content": _content_to_text(m.get("content"))})
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
    chatbot_history: list[dict],s
):
    history_state = _normalize_history(history_state)
    chatbot_history = _normalize_history(chatbot_history)

    if not repo_url:
        return "Введите URL репозитория.", "Источники:\n- не найдено\n", [], history_state, chatbot_history

    if not message:
        return "Введите вопрос.", "Источники:\n- не найдено\n", [], history_state, chatbot_history

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
        return f"Ошибка: {type(e).__name__}: {e}", "Источники:\n- не найдено\n", [], history_state, chatbot_history

    final_answer = answer_text or "Ответ пуст."

    sources = _collect_sources(response)
    sources_md = _render_sources(sources, show_sources)

    chatbot_history = chatbot_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": final_answer},
    ]

    new_history_state = request_messages + [{"role": "assistant", "content": final_answer}]
    new_history_state = _last_pairs(new_history_state, pairs=3)

    return sources_md, sources, new_history_state, chatbot_history


def update_sources(show_sources: bool, sources: list[dict]):
    return _render_sources(sources or [], show_sources)


with gr.Blocks(title="RAGCode") as demo:
    gr.Markdown("# RAGCode")

    with gr.Tabs():
        with gr.Tab("Индексировать репозиторий"):
            repo_url_input = gr.Textbox(label="GitHub URL")
            index_button = gr.Button("Индексировать")
            index_status = gr.Markdown()
            index_button.click(index_repo, inputs=repo_url_input, outputs=index_status)

        with gr.Tab("Чат по коду"):
            chat_repo_url = gr.Textbox(label="URL репозитория")

            chatbot = gr.Chatbot(label="История", height=420)

            sources = gr.Markdown("Источники:\n- не найдено\n")
            message_input = gr.Textbox(label="Ваш вопрос")
            show_sources = gr.Checkbox(label="Показывать содержимое источников", value=False)

            sources_state = gr.State([])
            history_state = gr.State([])

            send_button = gr.Button("Спросить")

            send_button.click(
                chat,
                inputs=[chat_repo_url, message_input, show_sources, history_state, chatbot],
                outputs=[sources, sources_state, history_state, chatbot],
            )

            show_sources.change(
                update_sources,
                inputs=[show_sources, sources_state],
                outputs=[sources],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8501)
