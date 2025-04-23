import numpy as np
from pymongo import MongoClient
from scipy.spatial.distance import cosine
from vectorizer import VectorizationService
from datamodels import ContentBase
from typing import List, Dict
import openai
import os
from datetime import datetime

class ReportGenerator:
    def __init__(self, top_k: int = 3):
        self.vectorizer = VectorizationService()
        self.db_client = MongoClient("mongodb://localhost:27017/")
        self.db = self.db_client["textbook_vectors"]
        self.top_k = top_k
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_report(self, query_text: str, output_path: str = "report.md", append: bool = False):
        relevant_content = self._get_relevant_content(query_text)
        analysis = self._analyze_content(query_text, relevant_content)
        follow_up_questions = self._generate_follow_up_questions(query_text, relevant_content)
        self._create_clean_report(analysis, query_text, output_path, append)
        return follow_up_questions

    def _get_relevant_content(self, query_text: str) -> List[Dict]:
        query_vector = self._vectorize_query(query_text)
        all_docs = self.db["content_vectors"].find({}, {"vector": 1, "content_type": 1, "page": 1, "text": 1})

        results = []
        for doc in all_docs:
            doc_vector = np.array(doc["vector"])
            similarity = 1 - cosine(query_vector, doc_vector)
            results.append({
                "similarity": similarity,
                "content_type": doc["content_type"],
                "page": doc["page"],
                "text": doc.get("text", "")
            })

        return sorted(results, key=lambda x: x["similarity"], reverse=True)[:self.top_k]

    def _vectorize_query(self, text: str) -> np.ndarray:
        content = ContentBase(
            content_type="query",
            description=text,
            page=0,
            positions=[],
            original=text
        )
        return np.array(self.vectorizer.batch_vectorize([content])[0].vector)

    def _analyze_content(self, query: str, contents: List[Dict]) -> str:
        context = "\n\n".join([
            f"[Page {c['page']} - {c['content_type']}]\n{c['text']}" for c in contents
        ])

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": (
                    "You're an educational assistant helping university students understand academic material. "
                    "Generate a well-structured MARKDOWN document based on the following query and references.\n"
                    "Make sure the markdown contains:\n"
                    "- A short summary\n"
                    "- A detailed explanation\n"
                    "Use clear, accessible language for a student audience."
                )
            }, {
                "role": "user",
                "content": f"Query: {query}\n\nContext:\n{context}"
            }],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content

    def _generate_follow_up_questions(self, query: str, contents: List[Dict]) -> List[str]:
        context = "\n\n".join([
            f"[Page {c['page']} - {c['content_type']}]\n{c['text']}" for c in contents
        ])

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": (
                    "You're an educational assistant. Based on the following query and references, generate 3-5 related follow-up questions that would help deepen the student's understanding of the topic. "
                    "Return only the questions as a bullet list."
                )
            }, {
                "role": "user",
                "content": f"Query: {query}\n\nContext:\n{context}"
            }],
            temperature=0.5,
            max_tokens=300
        )

        lines = response.choices[0].message.content.strip().split('\n')
        return [line.strip("-• ").strip() for line in lines if line.strip()]

    def _create_clean_report(self, analysis: str, query: str, path: str, append: bool = False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode = "a" if append else "w"

        with open(path, mode, encoding="utf-8") as f:
            if not append:
                f.write(f"# Answer to Your Question\n")
                f.write(f"**Initial Query**: {query}  \n")
                f.write(f"**Started on**: {timestamp}\n\n---\n\n")
            else:
                f.write("\n\n---\n\n")

            f.write(f"Follow-up: {query}\n")
            f.write(f"**Generated on**: {timestamp}\n\n")
            f.write("Explanation\n")
            f.write(analysis + "\n\n")
