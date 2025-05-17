import base64
import numpy as np
from pymongo import MongoClient
from scipy.spatial.distance import cosine
from vectorizer import VectorizationService
from datamodels import ContentBase
from typing import List, Dict
import openai
import os
from datetime import datetime
import re

class PPTGenerator:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.template = """<!-- PPT Slide {page_num} -->
# {title}

![{image_alt}](data:image/png;base64,{image_data})

{explanation}

---
"""

    def generate_explanation(self, description: str, original: str, query: str) -> List[str]:
        prompt = (
            "The user asked the following question:\n"
            f"{query}\n\n"
            "You are given an image described as:\n"
            f"{description}\n\n"
            "And here is some related source content:\n"
            f"{original}...\n\n"
            "Please write up to 3 different short (50-80 words) explanations that use the image to help answer the question. "
            "If the image is simple, just write one or two. "
            "Use the image as supporting evidence — do not explain the image itself, but rather explain the problem using the image to support the answer."
            "Do NOT include filler or weakly-differentiated answers. "
            "Format your response as a numbered list (1., 2., ...)."
        )
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        raw_text = response.choices[0].message.content.strip()
        lines = [line.strip() for line in raw_text.split('\n') if line.strip().startswith(tuple("1234567890"))]
        return lines

class EnhancedQueryService:
    def __init__(self, top_k: int = 10):  
        self.vectorizer = VectorizationService()
        self.db = MongoClient("mongodb://localhost:27017/")["textbook_vectors"]
        self.ppt_gen = PPTGenerator()
        self.top_k = top_k

    def query_to_ppt(self, query_text: str, output_path: str = "presentation.md"):
        query_vector = self._vectorize_query(query_text)
        results = self._hybrid_search(query_text, query_vector)

        md_content = []
        page_counter = 1
        max_pages = 5

        for result in results:
            if page_counter > max_pages:
                break

            slides = self._create_slide(result, start_page=page_counter, query_text=query_text)

            if page_counter + len(slides) - 1 > max_pages:
                slides = slides[:(max_pages - page_counter + 1)] 
            md_content.extend(slides)
            page_counter += len(slides)

        full_content = "\n".join(md_content)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        return output_path

    def _create_slide(self, result: Dict, start_page: int, query_text: str) -> List[str]:
        slides = []
        content_type = result["content_type"]
        base64_data = result["original"]

        if content_type in ("image", "table"):
            explanations = self.ppt_gen.generate_explanation(
                description=result.get("description", ""),
                original=result.get("original_metadata", ""),
                query=query_text
            )
            for i, explanation in enumerate(explanations):
                if i == 0 and start_page == 1:
                    explanation_text = f"**Question:** {query_text}\n\n{explanation}"
                else:
                    explanation_text = explanation

                slides.append(self.ppt_gen.template.format(
                    page_num=start_page + i,
                    title=f"{content_type.capitalize()} (View {i+1})",
                    image_alt=f"Relevant {content_type}",
                    image_data=base64_data,
                    explanation=explanation_text
                ))

        else:
            slides.append(self.ppt_gen.template.format(
                page_num=start_page,
                title=f"{content_type.capitalize()} Explanation",
                image_alt="",
                image_data="",
                explanation=result.get("description", "")
            ))

        return slides

    def _hybrid_search(self, query_text: str, query_vector: np.ndarray) -> List[Dict]:
        vector_results = self._vector_search(query_vector)
        keyword_results = self._keyword_search(query_text)

        seen = set()
        combined = []
        for res in vector_results + keyword_results:
            key = f"{res['content_type']}-{res['page']}"
            if key not in seen:
                seen.add(key)
                combined.append(res)

        return sorted(combined, key=lambda x: x["similarity"], reverse=True)

    def _vector_search(self, query_vector: np.ndarray) -> List[Dict]:
        collection = self.db["content_vectors"]
        results = []
        for doc in collection.find({"content_type": {"$in": ["image", "table"]}}):
            doc_vector = np.array(doc["vector"])
            similarity = 1 - cosine(query_vector, doc_vector)
            results.append({
                "content_type": doc["content_type"],
                "original": doc["original"],
                "description": doc.get("description", ""),
                "original_metadata": doc.get("original_metadata", ""),
                "page": doc.get("page", 0),
                "similarity": similarity
            })
        return sorted(results, key=lambda x: x["similarity"], reverse=True)

    def _keyword_search(self, query_text: str) -> List[Dict]:
        return []

    def _vectorize_query(self, text: str) -> np.ndarray:
        content = ContentBase(
            content_type="query",
            description=text,
            page=0,
            positions=[],
            original=text
        )
        return np.array(self.vectorizer.batch_vectorize([content])[0].vector)
    
    def _extract_slide_explanation_only(self, ppt_path: str, slide_num: str) -> str:
        with open(ppt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        slide_header = f"<!-- PPT Slide {slide_num} -->"
        end_marker = "---"
        inside = False
        explanation_lines = []

        for line in lines:
            if line.strip() == slide_header:
                inside = True
                continue
            if inside:
                if line.strip() == end_marker:
                    break
                if not line.strip().startswith("![") and not line.strip().startswith("<!--"):
                    explanation_lines.append(line.strip())

        return "\n".join(explanation_lines).strip()

    def handle_followup(
            self,
            slide_num: str,
            followup_question: str,
            original_image: str,
            image_description: str,
            query_text: str,
            ppt_path: str = "presentation.md"
        ):


        query_vector = self._vectorize_query(followup_question)

        image_candidates = self._vector_search(query_vector)
        image_candidates = sorted(image_candidates, key=lambda x: x["similarity"], reverse=True)
        best = image_candidates[0]
        best_image_data = best["original"]
        best_image_desc = best["description"]


        parent_explanation_text = self._extract_slide_explanation_only(ppt_path, slide_num)



        related_texts = self._search_text_by_question(followup_question)
        context_text = "\n\n".join([doc['original'] for doc in related_texts])[:1000]


        prompt = (
            f"The user asked a follow-up question about this slide explanation:\n"
            f"{parent_explanation_text}\n\n"
            f"Follow-up question:\n{followup_question}\n\n"
            f"The most relevant image for this follow-up question is described as:\n{best_image_desc}\n\n"
            f"Here is some related context from the document:\n{context_text}\n\n"
            "Please generate up to 2 concise (50–100 words) explanations to help answer this follow-up question. "
            "Format your answers as a numbered list. Use the new image to support your answer if relevant."
        )



        response = self.ppt_gen.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        raw_text = response.choices[0].message.content.strip()
        explanations = [line.strip() for line in raw_text.split('\n') if line.strip().startswith(tuple("1234567890"))][:2]

        with open(ppt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()


        insert_index = None
        slide_header = f"<!-- PPT Slide {slide_num} -->"
        end_marker = "---"

        for idx, line in enumerate(lines):
            if line.strip() == slide_header:
                depth = idx + 1
                while depth < len(lines):
                    if lines[depth].strip() == end_marker:
                        insert_index = depth + 1
                        break
                    depth += 1
                break

        if insert_index is None:
            raise ValueError(f"Slide {slide_num} not found. Cannot insert follow-up.")

        child_slides = []
        for i, explanation in enumerate(explanations, 1):
            nested_id = f"{slide_num}.{i}"
            if i == 1:
                explanation_text = f"**Follow-up Question:** {followup_question}\n\n{explanation}"
            else:
                explanation_text = explanation

            child_md = self.ppt_gen.template.format(
                page_num=nested_id,
                title=f"Follow-up {nested_id}",
                image_alt="Relevant image",
                image_data=best_image_data,
                explanation=explanation_text
            )
            child_slides.append(child_md)


        lines[insert_index:insert_index] = ["\n"] + child_slides + ["\n"]

        with open(ppt_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _search_text_by_question(self, question: str) -> List[Dict]:
        """
        Basic keyword-based text search from MongoDB.
        """
        collection = self.db["content_vectors"]
        keyword = question.lower()
        results = []
        for doc in collection.find({"content_type": "text"}):
            if keyword in doc.get("description", "").lower() or keyword in doc.get("original", "").lower():
                results.append(doc)
        return results[:1] 

# Optional Test
if __name__ == "__main__":
    service = EnhancedQueryService()
    query = "smartphone usage impact on productivity"
    output_file = service.query_to_ppt(query)
    print(f"Presentation generated at: {output_file}")
    with open(output_file, encoding="utf-8") as f:
        print(f.read())
