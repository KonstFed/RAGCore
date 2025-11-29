import streamlit as st
import asyncio
from src.assistant import Assistant
from uuid import uuid4


@st.cache_resource
def get_assistant():
    return Assistant(
        service_cfg_path="configs/deployment_config.yaml",
        swagger_path="api/api.yaml"
    )

assistant = get_assistant()

# TODO здесь нужно прикрутить UI: Gradio / Streamlit etc. или сделать доступность сервиса через aiohttp

st.set_page_config(page_title="RAGCode")

tab_index, tab_chat = st.tabs(["Индексировать репозиторий", "Чат по коду"])

with tab_index:
    repo_url = st.text_input("GitHub URL")

    if st.button("Индексировать") and repo_url:

        request = {
            "meta": {
                "request_id": str(uuid4())
            },
            "repo_url": repo_url,
            "branch": "main",
            "config": {
                "chunker_config": {
                    "max_chunk_size": 1000,
                    "chunk_overlap": 50,
                    "extensions": [".py", ".ipynb", ".cpp", ".h", ".java", ".ts", ".tsx", ".cs"],
                    "chunk_expansion": True,
                    "metadata_template": "default"
                },
                "embedding_config": {
                    "model_name": "qwen3-embedding-0.6b",
                    "dimensions": 1024,
                    "max_tokens": 8192
                },
                "exclude_patterns": ["*.lock", "__pycache__", ".venv", "build"]
            }
        }

        with st.spinner('Думаю...'):
            try:
                response = asyncio.run(assistant.index(request))
                st.success(f"Репозиторий с request_id={response.job_id} в статусе '{response.status}'")
            except Exception as e:
                st.error(f"Произошла ошибка: {e}")


with tab_chat:
    repo_url = st.text_input("URL репозитория")
    question = st.text_area("Ваш вопрос")

    if st.button("Спросить") and question and repo_url:

        request = {
            "meta": {
                "request_id": str(uuid4())
            },
            "query": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Я задаю какой-то вопрос"
                    }
                ]
            },
            "repo_url": repo_url
        }

        with st.spinner('Думаю...'):
            try:
                response = asyncio.run(assistant.query(request))

                st.markdown(response.answer)

                with st.expander("Источники"):
                    for source in response.sources:
                        st.code(source.content, language=source.metadata.language or "text")
                        st.caption(f"File: {source.metadata.filepath}")

            except Exception as e:
                st.error(f"Произошла ошибка: {e}")
