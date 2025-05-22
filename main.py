from picture import (
    extract_images_positions,
    extract_figure_captions,
    match_captions_to_images,
    generate_image_description,
    extract_image_data
)
from table import (
    extract_table_captions,
    extract_tables,
    match_captions_to_tables,
    generate_table_description,
    extract_table_original_using_pdf2image
)
from text import PDFProcessor, ContentSummarizer
from vectorizer import VectorizationService
from mongodb_store import MongoDBStorage
import openai
import os
from datamodels import ContentBase

class PDFIntegrator:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def process_all(self):
        print("Processing images...")
        image_positions = extract_images_positions(self.pdf_path)
        captions_by_page = extract_figure_captions(self.pdf_path)
        image_matches = match_captions_to_images(self.pdf_path, image_positions, captions_by_page)
        
        # Process each image match with the updated extraction method
        for img_match in image_matches:
            # Extract image data using the improved method from picture.py
            img_match["image_data"] = extract_image_data(self.pdf_path, img_match)
            # Generate description
            img_match["description"] = generate_image_description(
                img_match, 
                self.pdf_path, 
                self.openai_client
            )
            # Convert matched_image to serializable format if needed
            if isinstance(img_match["matched_image"], dict):
                img_match["matched_image"] = [
                    img_match["matched_image"]["x0"],
                    img_match["matched_image"]["y0"],
                    img_match["matched_image"]["x1"],
                    img_match["matched_image"]["y1"]
                ]
        
        # Process tables
        print("Processing tables...")
        table_captions = extract_table_captions(self.pdf_path)
        tables = extract_tables(self.pdf_path)
        table_matches = match_captions_to_tables(table_captions, tables)
        
        for tab in table_matches:
            tab["original"] = extract_table_original_using_pdf2image(self.pdf_path, tab)
            tab["description"] = generate_table_description(tab, self.pdf_path, self.openai_client)
        
        # Process text 
        print("Processing text...")
        processor = PDFProcessor()
        structured = processor.structure_content(self.pdf_path)
        summarizer = ContentSummarizer()
        text_results = summarizer.process_textbook(structured)
        
        return self._convert_to_unified_format(
            image_matches, 
            table_matches,
            text_results
        )
    
    def _convert_to_unified_format(self, images, tables, texts):
        """Convert outputs from different modules into a unified format"""
        unified = []
        
        # Process images
        for img in images:
            unified.append(ContentBase(
                content_type="image",
                description=img["description"],
                page=img["page"],
                positions=img["matched_image"],  
                original=img["image_data"]       
            ))
        
        # Process tables
        for tab in tables:
            unified.append(ContentBase(
                content_type="table",
                description=tab["description"],
                page=tab["page"],
                positions=list(tab["table"]),
                original=tab["original"]
            ))
        
        # Process text
        for text in texts:
            unified.append(ContentBase(
                content_type="text",
                description=text["summary"],
                page=text["page"],
                positions=text.get("positions", []),
                original=text["content"]
            ))
        
        return unified

if __name__ == "__main__":
    pdf_path = "D:/作业/research/learning system/2409.13997v1.pdf"
    
    # Step 1: Process all content
    integrator = PDFIntegrator(pdf_path)
    unified_data = integrator.process_all()
    
    # Step 2: Vectorization
    vector_service = VectorizationService()
    vectorized = vector_service.batch_vectorize(unified_data)
    
    # Step 3: Store to MongoDB
    storage = MongoDBStorage()
    count = storage.store_vectors(vectorized)
    
    print(f"Successfully stored {count} vectors with content")
