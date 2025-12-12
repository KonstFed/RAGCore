import os
import requests
from typing import List, Dict, Any
import json


class VectorDBClient:
    """Класс клаиента векторной базы данных."""
    def __init__(self, cfg):
        self.db_url = cfg.database.url
        self.dimension = cfg.embeddings.dimension
        self.distance = cfg.embeddings.distance
        self.top_k = cfg.database.top_k

    def get_collections(self) -> Dict:
        """Получает список коллекций из векторной базы данных."""
        response = requests.get(f"{self.db_url}/collections")
        return response.json()

    def create_collection(self, collection_name: str) -> Dict:
        """Создает коллекцию в векторной базе данных."""
        data = {
            "vectors": {
                "size": self.dimension,
                "distance": self.distance
            }
        }
        response = requests.put(f"{self.db_url}/collections/{collection_name}", json=data)
        return response.json()

    def get_collection(self, collection_name: str) -> Dict:
        """Получает информацию о коллекции."""
        response = requests.get(f"{self.db_url}/collections/{collection_name}")
        return response.json()
    
    def delete_collection(self, collection_name: str) -> Dict:
        """Удаляет коллекцию из векторной базы данных."""
        response = requests.delete(f"{self.db_url}/collections/{collection_name}")
        return response.json()
    
    def add_vectors(self, collection_name: str, vectorized_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Добавляет вектора в коллекцию"""

        url = f"{self.db_url}/collections/{collection_name}/points"
        
        params = {
            "wait": "true"
        }
        
        payload = {
            "points": vectorized_data
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.put(
            url,
            params=params,
            headers=headers,
            json=payload
        )
    
        return response.json()
    

    def search_by_vector(self, collection_name: str, vector: List[float]) -> Dict[str, Any]:
        """Выполняет поиск по вектору в коллекции."""
        url = f"{self.db_url}/collections/{collection_name}/points/search"
        payload = {
            "vector": vector,
            "limit": self.top_k,
            "with_payload": True,
            "with_vector": False
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
        )
        
        return response.json()





        



    
