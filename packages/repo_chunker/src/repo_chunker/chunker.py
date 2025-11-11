import chunk
from pathlib import Path
from typing import Iterator

from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern
from astchunk import ASTChunkBuilder


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


class LanguageRouter:
    """Routes files to the appropriate language-specific chunker."""


    language_extensions = {
        ".py": "python",
        ".pyi": "python",
        ".ipynb": "python",
        ".java": "java",
        ".cs": "csharp",
        ".csx": "csharp",
        ".ts": "typescript",
        ".tsx": "typescript",
    }

    DEFAULT_CONFIG = {
        "max_chunk_size": 100,
        "chunk_overlap": 10,
        "chunk_expansion": True,
        "metadata_template": "default",
    }

    def __init__(self, languages: list[str]=["python", "java", "csharp", "typescript"], config: dict=None):
        config = config or self.DEFAULT_CONFIG
        self.chunkers = {language: ASTChunkBuilder(**config, language=language) for language in languages}



    def route(self, file_path: Path) -> Path:
        """Route a file to the appropriate language-specific chunker."""
        language = self.language_extensions.get(file_path.suffix, "unknown")

        if language == "unknown":
            # TODO write chunking for docs and not code
            return None

        chunker = self.chunkers[language]
        with file_path.open("r") as f:
            content = f.read()
        return chunker.chunkify(content)


if __name__ == "__main__":
    from pathlib import Path

    repo_path = Path("/Users/konstantinfedorov/Documents/itmo_study/intro_to_llm/repoqa/RAGCore")
    chunker = RepoIterator()
    router = LanguageRouter()
    print("Files to be processed in repo:")
    for file in chunker(repo_path):
        print("A" * 10, file)
        res = router.route(file)
        if res is not None:
            print(res)


