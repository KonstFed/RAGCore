# TODO: could be great to write with parallelization
# because rn it takes long time to run :)
import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import List, Dict

import pandas as pd

from src.assistant import Assistant


def parse_repo_metadata(repo_meta_path: Path) -> List[str]:
    """
    Парсит метаданные репозиториев из текста возращает url с commit hash.

    Returns:
        List[str]
    """

    if not repo_meta_path.exists():
        raise FileNotFoundError(f"Repo file not found: {repo_meta_path}")

    result = []

    with repo_meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            repo_url, commit_hash = line.split()
            repo_url_with_commit = f"{repo_url}/tree/{commit_hash}"
            result.append(repo_url_with_commit)

    return result


async def index_repositories(
    assistant: Assistant, repos: List[str], index_config: dict
):
    """
    Индексирует список репозиториев с указанными commit hashes.

    Args:
        assistant: Assistant instance
        repos: List of (repo_url, commit_hash) tuples
        index_config: Configuration for indexing
    """
    indexed_repos = set()

    for idx, repo_url_with_commit in enumerate(repos):
        # Формируем URL с commit hash в формате /tree/commit-hash

        print(f"\n[{idx + 1}/{len(repos)}] Processing {repo_url_with_commit}...")

        # Проверяем, не индексировали ли мы уже этот репозиторий
        if repo_url_with_commit in indexed_repos:
            print(f"  Repository {repo_url_with_commit} already indexed, skipping...")
            continue

        try:
            index_request = {
                "repo_url": repo_url_with_commit,
                "meta": {"request_id": str(uuid.uuid4())},
            }

            print(f"  Indexing repository: {repo_url_with_commit}")
            response = await assistant.index(index_request, index_config)

            if response.meta.status == "done":
                print(f"  Successfully indexed {repo_url_with_commit}")
                indexed_repos.add(repo_url_with_commit)
            else:
                msg = (
                    f"  Failed to index {repo_url_with_commit}: "
                    f"{response.job_status.description_error}"
                )
                print(msg)

        except Exception as e:
            print(
                f"  Exception occurred during indexing of {repo_url_with_commit}: {e}"
            )


async def eval_single_repo(
    assistant: Assistant,
    repo_url: str,
    questions: pd.DataFrame,
    output_file: Path,
    query_config: dict,
):
    """
    Оценивает ответы ассистента на вопросы для одного репозитория.

    Args:
        assistant: Assistant instance
        repo_url: URL of the repository
        questions: DataFrame with questions
        output_file: Path to output JSONL file
        query_config: Configuration for querying
    """
    results = []
    for idx, row in questions.iterrows():
        question_text = row.question
        print(f"  [{idx + 1}/{len(questions)}] Question: {question_text[:80]}...")

        try:
            # Формируем запрос к ассистенту
            query_request = {
                "repo_url": repo_url,
                "meta": {"request_id": str(uuid.uuid4())},
                "query": {"messages": [{"role": "user", "content": question_text}]},
            }

            # Выполняем запрос
            response = await assistant.query(query_request, query_config)

            # Извлекаем ответ
            answer = response.answer if response.answer else "No answer generated"

            # Формируем результат (только question и answer)
            result = {"question": question_text, "answer": answer}

            results.append(result)
            print("    ✓ Answer generated")

        except Exception as e:
            # В случае ошибки записываем полную информацию об ошибке
            error_details = str(e)
            error_msg = f"Error: {error_details}"
            print(
                f"    ✗ Error: {error_msg[:200]}..."
            )  # Показываем первые 200 символов

            result = {"question": question_text, "answer": error_msg}
            results.append(result)

    # Записываем результаты в JSONL файл
    with output_file.open("a", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"  ✓ Saved {len(results)} results to {output_file}")
    return results


async def eval(repo_meta_path: Path, question_path: Path, output_path: Path = None):
    """
    Основная функция оценки.

    Args:
        repo_meta_path: Path to file with repository metadata
        question_path: Path to directory with question files
        output_path: Path to output JSONL file
        (default: results.jsonl in question_path directory)
    """
    assistant = Assistant(service_cfg_path="configs/deployment_config.yaml")

    index_config = {  # IndexConfig
        "ast_chunker_config": {
            "max_chunk_size": 1500,
            "chunk_overlap": 100,
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
        "text_splitter_config": {
            "chunk_size": 2500,
            "chunk_overlap": 300,
        },
        "exclude_patterns": ["*.lock", "__pycache__", ".venv", "build"],
    }

    query_config = {  # SearchConfig
        "query_preprocessor": {
            "enabled": True,
            "normalize_whitespace": True,
            "sanitization": {
                "enabled": True,
                "regex_patterns": ["jailbreak", "hallucinations"],
                "replacement_token": "",
            },
        },
        # временно отключаем rewriter для запроса
        "query_rewriter": {"enabled": False},
        "retriever": {"enabled": True},
        "filtering": {"enabled": True},
        # отключаем / включаем reranker (если внешний API нестабилен)
        "reranker": {"enabled": False},
        "context_expansion": {"enabled": True},
        "qa": {"enabled": True},
        "query_postprocessor": {
            "enabled": True,
            "format_markdown": True,
            "sanitization": {
                "enabled": True,
                "regex_patterns": ["can't", "wtf", ""],
                "replacement_token": "",
            },
        },
    }

    # Читаем метаданные репозиториев
    print("Reading repository metadata...")
    repos = parse_repo_metadata(repo_meta_path)
    print(f"Found {len(repos)} repositories to index")

    repo_name_to_url: Dict[str, str] = {}
    for repo_url_with_commit in repos:
        # Извлекаем имя репозитория из URL
        # Например: https://github.com/django/django/tree/14fc2e9 -> django
        parts = repo_url_with_commit.split("/")
        repo_name = parts[4]  # Индекс 4 - имя репозитория (после owner)
        repo_name_to_url[repo_name] = repo_url_with_commit

    # Индексируем все репозитории
    await index_repositories(assistant, repos, index_config=index_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Очищаем выходной файл если он существует
    if output_path.exists():
        output_path.unlink()

    # Обрабатываем вопросы
    if question_path.is_dir():
        for question_file in question_path.iterdir():
            if question_file.suffix != ".jsonl":
                continue
            repo_name = question_file.stem
            print(f"\nProcessing questions for {repo_name}...")

            # Получаем URL репозитория
            repo_url = repo_name_to_url.get(repo_name)
            if not repo_url:
                print(
                    f"  ⚠ Warning: Could not find repo URL for {repo_name}, skipping..."
                )
                continue

            questions = pd.read_json(question_file, lines=True)
            await eval_single_repo(
                assistant, repo_url, questions, output_path, query_config
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SWE-QA benchmark")
    parser.add_argument(
        "--repo_meta",
        type=Path,
        required=True,
        help="Path to file with repository metadata",
    )
    parser.add_argument(
        "--question_path",
        type=Path,
        required=True,
        help="Path to directory with question files",
    )
    parser.add_argument(
        "--output", type=Path, required=False, help="Path to output JSONL file"
    )
    args = parser.parse_args()

    asyncio.run(eval(args.repo_meta, args.question_path, args.output))
