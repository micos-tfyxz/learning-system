from pymongo import MongoClient
from datamodels import VectorRecord
from typing import List

class MongoDBStorage:
    def __init__(self, db_name="textbook_vectors"):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
    
    def store_vectors(self, records: List[VectorRecord]):
        collection = self.db["content_vectors"]
        
        # Modified here: using direct attribute access instead of .get()
        documents = []
        for record in records:
            doc = {
                "content_type": record.content_type,  # Direct attribute access
                "description": record.description,
                "page": record.page,
                "positions": record.positions,
                "original": record.original,
                "vector": record.vector
            }
            documents.append(doc)
        
        # Batch insert
        result = collection.insert_many(documents)
        return len(result.inserted_ids)