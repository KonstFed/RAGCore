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
                # removed "" because it matches everywhere
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


async def chat(repo_url: str, message: str, show_sources: bool):
    if not repo_url:
        return "Введите URL репозитория.", "Источники:\n- не найдено\n", []

    if not message:
        return "Введите вопрос.", "Источники:\n- не найдено\n", []

    request = {
        "meta": {"request_id": str(uuid4())},
        "query": {"messages": [{"role": "user", "content": message}]},
        "repo_url": repo_url,
    }
    config = _build_search_config()

    try:
        response = await assistant.query(request, config)
        answer_text = (getattr(response, "answer", "") or "").strip()
    except Exception as e:
        return f"Ошибка: {type(e).__name__}: {e}", "Источники:\n- не найдено\n", []

    sources = _collect_sources(response)
    sources_md = _render_sources(sources, show_sources)

    return answer_text or "Ответ пуст.", sources_md, sources


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
            answer = gr.Markdown("Ответ появится здесь.")
            sources = gr.Markdown("Источники:\n- не найдено\n")
            message_input = gr.Textbox(label="Ваш вопрос")
            show_sources = gr.Checkbox(label="Показывать содержимое источников", value=False)
            sources_state = gr.State([])
            send_button = gr.Button("Спросить")

            send_button.click(
                chat,
                inputs=[chat_repo_url, message_input, show_sources],
                outputs=[answer, sources, sources_state],
            )

            show_sources.change(
                update_sources,
                inputs=[show_sources, sources_state],
                outputs=[sources],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8501)
