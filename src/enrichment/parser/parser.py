import os
import fnmatch
import json
from pathlib import Path
from typing import List, Optional, Tuple
from omegaconf import DictConfig
from astchunk import ASTChunkBuilder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.schemas import (
    Chunk,
    ChunkMetadata,
    IndexConfig,
    IndexJobResponse,
)
from src.utils.logger import get_logger


class RepoParser:
    """
    Отвечает за обход файловой системы и чанкинг кода.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.default_exclude = cfg.parser.default_exclude
        self.extension_map = cfg.parser.extension_map
        self.dump_dir = cfg.paths.temp_chunks_storage

    def pipeline(
        self, config: IndexConfig, index_job_response: IndexJobResponse
    ) -> Tuple[IndexJobResponse, List[Chunk]]:
        """
        Запускает процесс парсинга репозитория.
        """
        chunks = []

        exclude_patterns = set(self.default_exclude)
        if config.exclude_patterns:
            exclude_patterns.update(config.exclude_patterns)

        # init ast chunker
        ast_chunker_map = {}
        if config.ast_chunker_config:
            for language in config.ast_chunker_languages:
                ast_chunker_map[language] = ASTChunkBuilder(
                    language=language,
                    **config.ast_chunker_config.model_dump(),
                )

        # init text splitter for non-AST languages
        splitter_cfg = config.text_splitter_config.model_dump()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=splitter_cfg["chunk_size"],
            chunk_overlap=splitter_cfg["chunk_overlap"],
            separators=splitter_cfg.get("separators"),
        )

        repo_path = index_job_response.job_status.repo_path
        msg = (
            "Start parsing repository {repo_path} "
            f"for request_id={index_job_response.meta.request_id}."
        )
        self.logger.info(msg)

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not self._is_excluded(d, exclude_patterns)]

            for file in files:
                if self._is_excluded(file, exclude_patterns):
                    continue

                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, repo_path)

                file_chunks = self._process_file(
                    full_path, relative_path, file, ast_chunker_map, text_splitter
                )
                chunks.extend(file_chunks)

        msg = (
            "Successful done parsing repository {repo_path} "
            f"for request_id={index_job_response.meta.request_id}."
        )
        self.logger.info(msg)
        index_job_response.job_status.status = "parsed"

        _chunks_path = self._save_chunks_locally(
            chunks, str(index_job_response.meta.request_id)
        )
        msg = (
            "Successful dump chunks into local storage: {chunks_path} "
            f"for request_id={index_job_response.meta.request_id}"
        )
        self.logger.info(msg)

        return index_job_response, chunks

    def _save_chunks_locally(self, chunks: List[Chunk], request_id: str) -> str:
        """
        Сериализует список чанков в JSON и сохраняет на диск.
        Возвращает путь к созданному файлу.
        """
        try:
            output_dir = Path(self.dump_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{request_id}.json"
            file_path = output_dir / filename

            data_to_save = [chunk.model_dump(mode="json") for chunk in chunks]

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)

            return str(file_path.absolute())

        except Exception as e:
            self.logger.error(f"Failed to save chunks locally for {request_id}: {e}")
            return ""

    def _is_excluded(self, name: str, patterns: set) -> bool:
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    def _process_file(
        self,
        full_path: str,
        relative_path: str,
        filename: str,
        ast_chunker_map: dict[str, ASTChunkBuilder],
        text_splitter: Optional[RecursiveCharacterTextSplitter],
    ) -> List[Chunk]:
        """Читает файл и разбивает на чанки."""
        _, ext = os.path.splitext(filename)
        language = self.extension_map.get(ext)

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return []

        if language in ast_chunker_map:
            # use AST chunker if language match
            return self._chunk_ast(content, relative_path, ast_chunker_map[language])
        else:
            # default use lanchain text splitter
            return self._chunk_langchain(
                content, relative_path, language, text_splitter
            )

    def _chunk_ast(
        self, content: str, filepath: str, ast_chunker: ASTChunkBuilder
    ) -> List[Chunk]:
        """
        Простой пример AST чанкинга для Python: разбиваем по функциям и классам.
        """
        raw_chunks = ast_chunker.chunkify(content)
        chunks = []
        for chunk in raw_chunks:
            chunk = Chunk.model_validate(chunk)
            chunk.metadata.language = ast_chunker.language
            chunk.metadata.filepath = filepath
            chunks.append(chunk)
        return chunks

    def _chunk_langchain(
        self,
        content: str,
        filepath: str,
        language: str,
        text_splitter: RecursiveCharacterTextSplitter,
    ) -> List[Chunk]:
        """Использует LangChain text splitter"""
        text_chunks = text_splitter.split_text(content)
        chunks = []
        current_pos = 0

        for chunk_text in text_chunks:
            # Находим позицию чанка в оригинальном контенте
            chunk_start_pos = content.find(chunk_text, current_pos)
            if chunk_start_pos == -1:
                # Fallback: считаем строки в самом чанке
                chunk_lines = chunk_text.splitlines()
                start_line = current_pos // 100 + 1  # Приблизительная оценка
                end_line = start_line + len(chunk_lines) - 1
            else:
                # Точный подсчет строк
                start_line = content[:chunk_start_pos].count("\n") + 1
                end_line = content[: chunk_start_pos + len(chunk_text)].count("\n")
                current_pos = chunk_start_pos + len(chunk_text)

            chunk_lines = chunk_text.splitlines()
            meta = ChunkMetadata(
                filepath=filepath,
                chunk_size=len(chunk_text),
                line_count=len(chunk_lines),
                start_line_no=start_line,
                end_line_no=end_line,
                language=language,
            )
            chunks.append(Chunk(content=chunk_text, metadata=meta))

        return chunks
