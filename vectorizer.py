import openai
import numpy as np
import os
from datamodels import ContentBase, VectorRecord
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential

class VectorizationService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-small"
    
    @retry(stop=stop_after_attempt(3), 
          wait=wait_exponential(multiplier=1, min=4, max=10))
    def batch_vectorize(self, items: List[ContentBase]) -> List[VectorRecord]:
        """增强版批量向量化"""
        descriptions = [item.description for item in items]
        
        response = self.client.embeddings.create(
            input=descriptions,
            model=self.model,
            encoding_format="float"  # 确保返回浮点数格式
        )
        
        return [VectorRecord(
            **item.dict(),
            vector=response.data[idx].embedding
        ) for idx, item in enumerate(items)]