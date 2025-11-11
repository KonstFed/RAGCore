from pathlib import Path

from astchunk import ASTChunkBuilder
from pydantic import BaseModel

from repochunker.models import Chunk, ChunkerConfig
from repochunker.repoiter import RepoIterator


class RouterChunker:
    """Routes files to the appropriate language-specific chunker."""

    def __init__(
        self,
        repoiter: RepoIterator,
        language_configs: dict[str, ChunkerConfig],
        config: dict = None,
    ):
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

        if chunker is None:
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
        RouterChunkerConfig, Path(__file__).parent / "chunk_config_example.yaml"
    )
    router_chunker = router_config.create()
    chunks = router_chunker.chunk_repo(Path(__file__).parent.parent)
    print("Extracted chunks: ", len(chunks))
    idx = 0
    print("----------------EXAMPLE----------------")
    print(chunks[idx].content)
    print("----------------EXAMPLE END----------------")
