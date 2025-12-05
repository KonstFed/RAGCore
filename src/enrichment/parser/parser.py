import os
import fnmatch
import json
import uuid
from pathlib import Path
from typing import List

from astchunk import ASTChunkBuilder

from src.core.schemas import IndexRequest, Chunk, ChunkMetadata, IndexConfig, MetaRequest
from src.utils.logger import get_logger


class RepoParser:
    """
    Отвечает за обход файловой системы и чанкинг кода.
    """

    AST_CHUNKER_LANGUAGES = ["python", "java", "typescript", "csharp"]

    def __init__(self, cfg):
        self.default_exclude = cfg.parser.default_exclude
        self.extension_map = cfg.parser.extension_map
        self.dump_dir = cfg.paths.temp_chunks_storage

        self.logger = get_logger(self.__class__.__name__)

    def pipeline(
        self, repo_path: str, request: IndexRequest, config: IndexConfig
    ) -> List[Chunk]:
        """
        Запускает процесс парсинга репозитория.
        """
        chunks = []
        # TODO: раньше тут был почему-то IndexConfig() и `config` просто игнорился
        # если я неправильно исправил подправьте и протестируйте
        config = request.config or config
        exclude_patterns = set(self.default_exclude)
        if config.exclude_patterns:
            exclude_patterns.update(config.exclude_patterns)

        # init ast chunker
        ast_chunker_map = {}
        for language in self.AST_CHUNKER_LANGUAGES:
            ast_chunker_map[language] = ASTChunkBuilder(
                language=language, **config.chunker_config.model_dump(),
            )

        self.logger.info(f"Start parsing repository {repo_path}.")

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not self._is_excluded(d, exclude_patterns)]

            for file in files:
                if self._is_excluded(file, exclude_patterns):
                    continue

                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, repo_path)

                file_chunks = self._process_file(
                    full_path, relative_path, file, ast_chunker_map
                )
                chunks.extend(file_chunks)

        self.logger.info(f"Successful done parsing repository {repo_path}.")

        chunks_path = self._save_chunks_locally(chunks, str(request.meta.request_id))

        self.logger.info(f"Successful dump chunks into local storage: {chunks_path}")

        return chunks

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
    ) -> List[Chunk]:
        """Читает файл и разбивает на чанки."""
        _, ext = os.path.splitext(filename)
        language = self.extension_map.get(ext)

        if not language:
            return []

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return []

        if language in self.AST_CHUNKER_LANGUAGES:
            return self._chunk_ast(
                content, relative_path, filename, ast_chunker_map[language]
            )
        else:
            return self._chunk_simple_lines(content, relative_path, filename, language)

    def _chunk_ast(
        self, content: str, filepath: str, filename: str, ast_chunker: ASTChunkBuilder
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

    def _chunk_simple_lines(
        self, content: str, filepath: str, filename: str, language: str
    ) -> List[Chunk]:
        """Наивная нарезка по 100 строк для других языков."""
        lines = content.splitlines()
        total_lines = len(lines)
        chunk_size = 100
        overlap = 20
        chunks = []

        if total_lines == 0:
            return []

        for i in range(0, total_lines, chunk_size - overlap):
            end = min(i + chunk_size, total_lines)
            chunk_lines = lines[i:end]
            chunk_content = "\n".join(chunk_lines)

            meta = ChunkMetadata(
                filepath=filepath,
                chunk_size=len(chunk_content),
                line_count=len(chunk_lines),
                start_line_no=i + 1,
                end_line_no=end,
                language=language,
            )
            chunks.append(Chunk(content=chunk_content, metadata=meta))

        return chunks

    def _create_single_chunk(
        self, content, filepath, filename, language, start, end
    ) -> List[Chunk]:
        meta = ChunkMetadata(
            chunk_id=uuid.uuid4(),
            filepath=filepath,
            file_name=filename,
            chunk_size=len(content),
            line_count=end - start + 1,
            start_line_no=start,
            end_line_no=end,
            language=language,
        )
        return [Chunk(content=content, metadata=meta)]
