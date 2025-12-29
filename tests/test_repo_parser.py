import uuid
import pytest
from pathlib import Path

from src.assistant import Assistant
from src.core.schemas import (
    IndexRequest,
    IndexConfig,
    MetaRequest,
)

@pytest.fixture
def config_path() -> str:
    return "tests/data/test_deployment_config.yaml"

@pytest.fixture
def index_request() -> IndexRequest:
    return IndexRequest(
        meta=MetaRequest(request_id=str(uuid.uuid4())),
        repo_url="https://github.com/yilinjz/astchunk", # placeholder
    )

@pytest.fixture
def index_config() -> IndexConfig:
    config = { # IndexConfig
        "ast_chunker_config": {
            "max_chunk_size": 100,
            "chunk_overlap": 20,
            "extensions": [".py", ".ipynb", ".cpp", ".h", ".java", ".ts", ".tsx", ".cs"],
            "chunk_expansion": True,
            "metadata_template": "default"
        },
        "text_splitter_config": {
            "chunk_size": 500,
            "chunk_overlap": 50,
        },
        "embedding_config": {
            "model_name": "qwen3-embedding-0.6b",
            "dimensions": 1024,
            "max_tokens": 8192
        },
        "exclude_patterns": ["*.lock", "__pycache__", ".venv", "build"],
        "ast_chunker_languages": ["python", "java", "typescript", "csharp"]

    }
    return IndexConfig(**config)


def test_repo_parser(config_path: str, index_request: IndexRequest, index_config: IndexConfig) -> None:
    assistant = Assistant(config_path)

    chunks = assistant.enrichment.parser.pipeline(
        repo_path=Path("tests/data/test_repo"),
        request=index_request,
        config=index_config,
    )

    # simple test for checking if it runs at all
    assert len(chunks) == 22


    n_python_chunks = len([c for c in chunks if c.metadata.file_name == "a.py"])
    n_md_chunks = len([c for c in chunks if c.metadata.file_name == "some_text.md"])

    assert n_python_chunks == 7
    assert n_md_chunks == 15
