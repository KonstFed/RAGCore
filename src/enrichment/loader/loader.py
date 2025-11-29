from src.enrichment.loader.github import GitHubLoader
from src.core.db import VectorDBClient
from typing import Any, Dict, List, Tuple

class LoaderConnecter:
    def __init__(self):
        self.github = GitHubLoader()
        self.vectore_db_client = VectorDBClient()
