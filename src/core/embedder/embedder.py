import os
from src.core.schemas import Chunk
from typing import Any, Dict, List, Tuple


class EmbeddingModel:
    def __init__(self) -> None:
        pass

    def vectorize(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        pass
