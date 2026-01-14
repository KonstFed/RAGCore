from pydantic import BaseModel, Field


class ChunkerConfig(BaseModel):
    language: str
    max_chunk_size: int
    chunk_overlap: int
    chunk_expansion: bool
    metadata_template: str = "default"

    extensions: list[str] = Field(default_factory=list)


class Metadata(BaseModel):
    filepath: str
    chunk_size: int
    line_count: int
    start_line_no: int
    end_line_no: int
    node_count: int


class Chunk(BaseModel):
    content: str
    metadata: Metadata
