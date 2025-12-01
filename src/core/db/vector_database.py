import os


class VectorDBClient:
    """Класс клаиента векторной базы данных."""
    def __init__(self, cfg):
        self.db_url = cfg.get("DB_URL", "http://localhost:6333")
        self.api_key = cfg.get("DB_API_KEY", None)
