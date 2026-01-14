"""Handle github repos url"""

import os
import shutil
import re
from typing import Optional
import tarfile
import io
import requests


def _parse_github_url(url: str) -> tuple[str, str, str, Optional[str]]:
    """
    Парсит GitHub URL и извлекает owner, repo, base_url и branch/commit.

    Args:
        url: GitHub URL, может содержать /tree/branch или /tree/commit-hash

    Returns:
        - owner: владелец репозитория
        - repo: название репозитория
        - base_url: базовый URL репозитория без /tree/...
        - branch_or_commit: извлеченный branch или commit hash, или None
    """
    # Паттерн для URL вида: https://github.com/owner/repo/tree/branch-or-commit
    # или https://github.com/owner/repo/commit/commit-hash
    pattern = r"^https://github\.com/([^/]+)/([^/]+)(?:/(?:tree|commit)/([^/]+))?/?$"
    match = re.match(pattern, url)

    if not match:
        raise ValueError(f"Invalid GitHub URL: {url}")

    owner, repo, tree_ref = match.groups()
    base_url = f"https://github.com/{owner}/{repo}"

    return owner, repo, base_url, tree_ref


def resolve_full_github_url(url: str) -> tuple[str, str, str, str]:
    """
    Resolves a GitHub URL to its (owner, repo, base_url, commit_hash).

    - If a branch or commit is provided, resolves to that commit.
    - If not, resolves to the latest commit of the default branch.

    Args:
        url: GitHub repository URL (optionally with /tree/<branch> or /commit/<sha>).

    Returns:
        (owner, repo, base_url, commit_hash)

    Raises:
        ValueError: Invalid or unresolvable URL
        requests.HTTPError: On API error
    """
    owner, reponame, base_url, tree_ref = _parse_github_url(url)
    api_url = f"https://api.github.com/repos/{owner}/{reponame}"
    if tree_ref:
        response = requests.get(api_url + f"/commits/{tree_ref}", timeout=5)
        response.raise_for_status()
        commit_hash = response.json()["sha"]
    else:
        try:
            response = requests.get(api_url, timeout=5)
            response.raise_for_status()
            default_branch = response.json().get("default_branch", "main")
        except Exception:
            default_branch = "main"
        response = requests.get(api_url + f"/commits/{default_branch}", timeout=5)
        response.raise_for_status()
        commit_hash = response.json()["sha"]

    return owner, reponame, base_url, commit_hash


def download_github_archive(
    owner: str, repo: str, commit_hash: str, target_dir: str
) -> None:
    """
    Скачивает архив репозитория с GitHub для конкретного commit и распаковывает его.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        commit_hash: Commit hash для скачивания
        target_dir: Директория для распаковки архива
    """
    # GitHub Archive API URL
    archive_url = f"https://github.com/{owner}/{repo}/archive/{commit_hash}.tar.gz"
    # Скачиваем архив
    response = requests.get(archive_url, timeout=300, stream=True)
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
        tar.extractall(path=target_dir)

    # GitHub создает папку вида repo-{short_hash}, находим её
    extracted_dirs = [
        d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))
    ]

    if not extracted_dirs:
        raise ValueError("No directories found in extracted archive")

    # Перемещаем содержимое из вложенной папки на уровень выше
    actual_repo_path = os.path.join(target_dir, extracted_dirs[0])
    for item in os.listdir(actual_repo_path):
        shutil.move(
            os.path.join(actual_repo_path, item), os.path.join(target_dir, item)
        )

    # Удаляем пустую вложенную папку
    os.rmdir(actual_repo_path)
