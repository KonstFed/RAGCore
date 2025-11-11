from pathlib import Path
from typing import Iterator

from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from astchunk import ASTChunkBuilder
from pydantic import BaseModel, Field

from repo_chunker.models import ChunkerConfig, Chunk


class RepoIterator:
    """Iterates over repository files and filters out files matching .gitignore patterns."""

    DEFAULT_IGNORE_FOLDER = Path(__file__).parent / "ignore_rules"
    IGNORE_FILE_EXTENSION = ".ignore"

    def __init__(
        self,
        ignore_patterns_folder: Path | str | None = None,
    ):
        """Initialize RepoChunker with filtering options.

        Args:
            ignore_patterns_folder: Folder containing .ignore files (default: ignore_rules)
        """
        self.ignore_patterns_folder = (
            Path(ignore_patterns_folder) if ignore_patterns_folder else self.DEFAULT_IGNORE_FOLDER
        )
        
        self.ignore_spec = self._get_ignore_pattern(self.ignore_patterns_folder)

    def _load_patterns_from_file(self, ignore_file: Path) -> list[str]:
        """Load patterns from a single ignore file.

        Args:
            ignore_file: Path to the ignore file

        Returns:
            List of pattern strings
        """
        patterns = []
        with ignore_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    patterns.append(line)
        return patterns

    def _get_ignore_pattern(self, ignore_folder: Path) -> PathSpec:
        """Load and combine ignore patterns from all .ignore files in the specified folder.

        Args:
            ignore_folder: Path to the folder containing .ignore files.

        Returns:
            A PathSpec object containing patterns from all .ignore files in the folder.
        """
        all_patterns = []
        if ignore_folder.exists() and ignore_folder.is_dir():
            for file in ignore_folder.iterdir():
                if file.is_file() and file.suffix == self.IGNORE_FILE_EXTENSION:
                    patterns = self._load_patterns_from_file(file)
                    all_patterns.extend(patterns)

        return PathSpec.from_lines(GitWildMatchPattern, all_patterns)


    def iterate_files(self, repo_path: Path) -> Iterator[Path]:
        """Iterate over all files in the repository, excluding those matching ignore patterns.

        Args:
            repo_path: Path to the repository root

        Yields:
            Path objects for each file that should be processed

        Raises:
            ValueError: If repo_path doesn't exist or is not a directory
        """
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        if not repo_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {repo_path}")

        repo_path = repo_path.resolve()

        # Load and merge all ignore patterns

        for file_path in repo_path.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue

            file_path = file_path.relative_to(repo_path)

            # Skip if any parent directory is excluded
            if any(
                self.ignore_spec.match_file(parent.as_posix())
                for parent in file_path.parents
                if parent != Path(".")
            ):
                continue

            # Skip if the file itself is excluded
            if self.ignore_spec.match_file(file_path.as_posix()):
                continue

            yield file_path

    def __call__(self, repo_path: Path) -> Iterator[Path]:
        """
        Allows RepoChunker instances to be called directly, yielding non-ignored files in the given repository path.

        Args:
            repo_path (Path): Path to the repository root.

        Yields:
            Path: Files not excluded by ignore rules.
        """
        yield from self.iterate_files(repo_path)




class RouterChunker:
    """Routes files to the appropriate language-specific chunker."""

    def __init__(self, repoiter: RepoIterator, language_configs: dict[str, ChunkerConfig], config: dict=None):
        self.repoiter = repoiter
        self.chunkers = {}
        for language in language_configs:
            language_config = language_configs[language]
            chunker = ASTChunkBuilder(**language_config.model_dump())
            for extension in language_config.extensions:
                self.chunkers[extension] = chunker


    def chunk_file(self, file_path: Path) -> list[Chunk]:
        """Route a file to the appropriate language-specific chunker."""
        chunker = self.chunkers.get(file_path.suffix, None)

        if chunker == None:
            # this is not a code file
            # TODO write chunking for docs
            return []

        with file_path.open("r") as f:
            content = f.read()

        return [Chunk.model_validate(chunk) for chunk in chunker.chunkify(content)]

    def chunk_repo(self, repo_path: Path) -> list[Chunk]:
        """Chunk a repository."""
        chunks = []
        for file_path in self.repoiter(repo_path):
            file_path = repo_path / file_path
            chunks.extend(self.chunk_file(file_path))
        return chunks


class RouterChunkerConfig(BaseModel):
    chunkers: dict[str, ChunkerConfig]

    def create(self) -> RouterChunker:
        repoiter = RepoIterator()
        return RouterChunker(repoiter=repoiter, language_configs=self.chunkers)


if __name__ == "__main__":
    from repo_chunker.config_utils import load_config

    router_config = load_config(
        RouterChunkerConfig,
        Path(__file__).parent / "chunk_config_example.yaml"
    )
    router_chunker = router_config.create()
    chunks = router_chunker.chunk_repo(Path(__file__).parent.parent)
    print("Extracted chunks: ", len(chunks))
    idx = 0
    print("----------------EXAMPLE----------------")
    print(chunks[idx].content)
    print("----------------EXAMPLE END----------------")
