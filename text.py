import pdfplumber
import re
import json
import openai
from typing import List, Dict, Tuple
from tqdm import tqdm
import hashlib
import os

class PDFProcessor:
    def __init__(self, font_ratio=1.3, pos_threshold=0.2):
        # Title recognition parameters
        self.font_ratio = font_ratio  # Threshold ratio of title font size to body text
        self.pos_threshold = pos_threshold  # Title position threshold (top portion of page)
        self.bold_weight = 0.3  # Weight coefficient for bold feature
        # Cache processed PDF features
        self.font_stats = {}  # Store font feature analysis results
    
    def extract_pdf_features(self, pdf_path: str) -> List[Dict]:
        """Extract PDF text features (handling missing attributes)"""
        features = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                words = page.extract_words(
                    extra_attrs=["fontname", "size"],
                    x_tolerance=3,
                    y_tolerance=3
                )
                features.extend([
                    {
                        "text": word["text"],
                        "font_size": word.get("size", 10),
                        "bold": word.get("bold", False),
                        "fontname": word.get("fontname", "Unknown"),  # Default value
                        "x0": word["x0"],
                        "top": word["top"],
                        "page": page.page_number
                    } for word in words
                ])
        return features

    def analyze_font_statistics(self, features: List[Dict]):
        """Improved font statistics analysis"""
        # Filter invalid font data
        valid_features = [f for f in features if f["fontname"] != "Unknown"]
        
        if not valid_features:
            # Set default values
            self.font_stats = {
                "avg_size": 12,
                "std_size": 1,
                "main_font": (12, False, "Unknown")
            }
            return
        
        font_sizes = [f["font_size"] for f in valid_features]
        avg_size = sum(font_sizes) / len(font_sizes)
        std_size = (sum((x - avg_size)**2 for x in font_sizes)/len(font_sizes))**0.5
        
        # Count main font (ignore Unknown)
        font_counter = {}
        for f in valid_features:
            key = (f["font_size"], f["bold"], f["fontname"])
            font_counter[key] = font_counter.get(key, 0) + 1
        
        main_font = max(font_counter, key=font_counter.get) if font_counter else (12, False, "Unknown")
        
        self.font_stats = {
            "avg_size": avg_size,
            "std_size": std_size,
            "main_font": main_font
        }
    
    def is_heading(self, feature: Dict) -> bool:
        """Improved heading detection logic"""
        # Add font name check
        main_font_name = self.font_stats["main_font"][2]
        current_font_name = feature["fontname"]
        
        score = 0
        
        # Font difference score (when using different font names)
        if current_font_name != main_font_name and current_font_name != "Unknown":
            score += 0.5

        size_ratio = feature["font_size"] / self.font_stats["avg_size"]
        if size_ratio >= self.font_ratio:
            score += 1.5
        
        # Position score
        if feature["top"] < self.font_stats["avg_size"] * 3:
            score += 1.0
            
        # Bold score
        if feature["bold"]:
            score += self.bold_weight
            
        # Numbering format score
        if re.match(r'^(Chapter [IVXLCDM]+|(\d+\.)+\d+|Section\s\d+)', feature["text"]):
            score += 0.5
            
        return score >= 2.5  # Adjusted threshold for accuracy
    
    def structure_content(self, pdf_path: str) -> List[Dict]:
        """Structure PDF content and include text positions"""
        features = self.extract_pdf_features(pdf_path)
        self.analyze_font_statistics(features)

        structured = []
        current_section = {"title": "", "content": [], "positions": []}
        page_height = None

        for idx, feat in enumerate(features):
            # Initialize page height
            if page_height is None:
                with pdfplumber.open(pdf_path) as pdf:
                    page_height = pdf.pages[0].height

            # Heading detection
            if self.is_heading(feat):
                if current_section["title"]:  # Save current section
                    structured.append({
                        "title": current_section["title"],
                        "content": " ".join([x["text"] for x in current_section["content"]]),
                        "positions": current_section["positions"],
                        "page": feat["page"]
                    })
                # Start new section
                current_section = {
                    "title": feat["text"],
                    "content": [],
                    "positions": [],
                    "page": feat["page"]
                }
                # Merge consecutive headings
                next_idx = idx + 1
                while next_idx < len(features) and \
                    abs(feat["top"] - features[next_idx]["top"]) < 5:
                    current_section["title"] += " " + features[next_idx]["text"]
                    next_idx += 1
            else:
                # Paragraph detection (vertical position change)
                if idx > 0 and abs(feat["top"] - features[idx-1]["top"]) > 15:
                    current_section["content"].append({"text": "\n", "x0": None, "top": None})
                current_section["content"].append({
                    "text": feat["text"],
                    "x0": feat["x0"],
                    "top": feat["top"]
                })
                current_section["positions"].append((feat["x0"], feat["top"]))

        # Add last section
        if current_section["title"]:
            structured.append({
                "title": current_section["title"],
                "content": " ".join([x["text"] for x in current_section["content"]]),
                "positions": current_section["positions"],
                "page": feat["page"]
            })

        return structured

class ContentSummarizer:
    def __init__(self, chunk_size=3000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.openai_client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        self.cache = {}
    
    def process_textbook(self, structured_data: List[Dict]) -> List[Dict]:
        results = []
        for section in tqdm(structured_data, desc="Processing sections"):
            summary = self.summarize_section(section["content"], section["title"])
            key_points = self.extract_key_points(summary)
            
            results.append({
                "title": section["title"],
                "page": section["page"],
                "summary": summary,
                "key_points": key_points,
                "content": section["content"],  
                "content_hash": hashlib.md5(section["content"].encode()).hexdigest(),
                "positions": section.get("positions", [])
            })
        return results
    
    def summarize_section(self, content: str, title: str) -> str:
        """Generate section summary (with chunking)"""
        chunks = self._chunk_content(content)
        summaries = []
        
        for chunk in chunks:
            # Check cache
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
            if chunk_hash in self.cache:
                summaries.append(self.cache[chunk_hash])
                continue
                
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{
                    "role": "user",
                    "content": f"Please summarize the following technical content (in English):\nTitle: {title}\nContent: {chunk}"
                }],
                temperature=0.3,
                max_tokens=1000
            )
            summary = response.choices[0].message.content
            self.cache[chunk_hash] = summary
            summaries.append(summary)
        
        # Combine chunk summaries
        combined = "\n".join(summaries)
        final_response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Integrate the following chunk summaries:\n{combined}"
            }],
            temperature=0.2,
            max_tokens=1500
        )
        return final_response.choices[0].message.content
    
    def _chunk_content(self, text: str) -> List[str]:
        """Intelligent chunking (preserve complete sentences)"""
        chunks = []
        sentences = re.split(r'(?<=[.!?\n])', text)
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            sent_length = len(sent)
            if current_length + sent_length > self.chunk_size:
                chunks.append("".join(current_chunk))
                current_chunk = current_chunk[-int(self.overlap/50):]  
                current_length = sum(len(s) for s in current_chunk)
            current_chunk.append(sent)
            current_length += sent_length
        
        if current_chunk:
            chunks.append("".join(current_chunk))
        return chunks
    
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key knowledge points"""
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Extract 3-5 core knowledge points from the following text (in English bullet points):\n{text}"
            }],
            temperature=0.2,
            max_tokens=500
        )
        return [line.strip() for line in response.choices[0].message.content.split("\n") if line.strip()]

if __name__ == "__main__":
    pdf_path = ""
    
    # Step 1: PDF structuring
    print("Parsing PDF file...")
    processor = PDFProcessor(font_ratio=1.3)
    structured_data = processor.structure_content(pdf_path)
    
    # Step 2: Content summarization
    print("\nGenerating summaries...")
    summarizer = ContentSummarizer()
    results = summarizer.process_textbook(structured_data)
    
    # Save results
    with open("textbook_summary.json", "w", encoding="utf-8") as f:
        json.dump({"sections": results}, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing completed! Processed {len(results)} sections.")