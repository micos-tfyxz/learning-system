import pdfplumber
import re
import json
import os
from tqdm import tqdm
import numpy as np
import openai   
from pdf2image import convert_from_path
from PIL import Image
import io
import base64

def get_line_bbox(line_text, chars):
    """Get the bounding box coordinates of a text line"""
    line_chars = [c for c in chars if c['text'] in line_text]
    if not line_chars:
        return None
    x0 = min(c['x0'] for c in line_chars)
    y0 = min(c['top'] for c in line_chars)
    x1 = max(c['x1'] for c in line_chars)
    y1 = max(c['bottom'] for c in line_chars)
    return (x0, y0, x1, y1)

def is_near(line_bbox, table_bbox, threshold=50):
    """Check if a line is near a table"""
    if not line_bbox or not table_bbox:
        return False
    return (table_bbox[1] - line_bbox[3] < threshold) or (line_bbox[1] - table_bbox[3] < threshold)

def extract_table_captions(pdf_path):
    """Improved table caption extraction: combines layout and context"""
    caption_pattern = re.compile(r'^(Table|表)\s*\d+\s*[:：]?\s*.+$', re.IGNORECASE)
    captions_by_page = {}
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            chars = page.chars
            tables = page.find_tables()
            table_bboxes = [table.bbox for table in tables]
            
            captions = []
            current_caption = []
            for line in text.splitlines():
                line_clean = line.strip()
                if caption_pattern.match(line_clean):
                    line_bbox = get_line_bbox(line, chars)  
                    is_valid = any(is_near(line_bbox, t_bbox) for t_bbox in table_bboxes)
                    if is_valid:
                        current_caption.append(line_clean)
                elif current_caption:
                    captions.append(' '.join(current_caption))
                    current_caption = []
            
            if current_caption:
                captions.append(' '.join(current_caption))
            
            if captions:
                captions_by_page[page_num] = captions
    
    return captions_by_page

def extract_tables(pdf_path):
    """
    Extract all table locations from a PDF,
    returning the bounding boxes (x0, y0, x1, y1) per page.
    """
    tables_by_page = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                table_objs = page.find_tables()
                bboxes = []
                
                for table in table_objs:
                    if hasattr(table, 'bbox'):
                        # Store original bounding box coordinates
                        bbox = table.bbox
                        bboxes.append(bbox)
                        print(f"Page {page_num}, Table bbox: {bbox}")
                
                if bboxes:
                    tables_by_page[page_num] = bboxes
            except Exception as e:
                print(f"Error finding tables on page {page_num}: {str(e)}")
    
    return tables_by_page

def match_captions_to_tables(captions_by_page, tables_by_page):
    """
    Match table captions to tables:
      1. If there is one caption and one table per page, match directly.
      2. If tables outnumber captions, merge tables to match caption count.
      3. If a table spans multiple pages, merge adjacent tables.
      4. Ignore pages with tables but no captions.
    Returns a list of matches: {"page": page_num, "caption": caption, "table": table_bbox}.
    """
    matches = []
    last_page_tables = None
    
    for page_num in sorted(set(captions_by_page.keys()) | set(tables_by_page.keys())):
        captions = captions_by_page.get(page_num, [])
        tables = tables_by_page.get(page_num, [])
        
        if last_page_tables and not captions and tables:
            tables = last_page_tables + tables
            last_page_tables = None
            
        if not captions:
            last_page_tables = tables
            continue

        if len(captions) == len(tables):
            for cap, tab in zip(captions, tables):
                matches.append({"page": page_num, "caption": cap, "table": tab})
        
        elif len(tables) > len(captions):
            num_groups = len(captions)
            group_size = len(tables) // num_groups
            remainder = len(tables) % num_groups
            start = 0
            for i in range(num_groups):
                size = group_size + (1 if i < remainder else 0)
                group = tables[start:start+size]
                x0 = min(b[0] for b in group)
                y0 = min(b[1] for b in group)
                x1 = max(b[2] for b in group)
                y1 = max(b[3] for b in group)
                merged_bbox = (x0, y0, x1, y1)
                matches.append({"page": page_num, "caption": captions[i], "table": merged_bbox})
                start += size
        
        else:
            for cap, tab in zip(captions, tables):
                matches.append({"page": page_num, "caption": cap, "table": tab})
    
    return matches

def generate_table_description(match, pdf_path, openai_client, margin=50):
    """
    Generate a short table description based on its caption and surrounding context.
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[match["page"] - 1]
        table_bbox = match["table"]
        x0 = max(0, table_bbox[0] - margin)
        y0 = max(0, table_bbox[1] - margin)
        x1 = min(page.width, table_bbox[2] + margin)
        y1 = min(page.height, table_bbox[3] + margin)
        context_bbox = (x0, y0, x1, y1)
        context_text = page.within_bbox(context_bbox).extract_text() or ""
        if len(context_text.strip()) < 20:
            context_text = page.extract_text() or ""
    
    prompt = (f"Please generate a concise description (within 50 words) for the table based on the following caption and context. "
              f"Caption: {match['caption']}\nContext: {context_text}\nDescription:")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        description = response.choices[0].message.content.strip()
        if not description:
            description = f"Fallback Description: Caption: {match['caption']}. Context: {context_text}"
    except Exception as e:
        description = f"Fallback Description: Caption: {match['caption']}. Context: {context_text}. Error: {str(e)}"
    return description

def extract_table_original_using_pdf2image(pdf_path, match, margin=50):
    """Extract table image from PDF without y-axis inversion"""
    try:
        images = convert_from_path(pdf_path, first_page=match["page"], last_page=match["page"], dpi=300)
        if not images:
            print("No images generated from PDF")
            return ""
        
        page_img = images[0]
        
        pdf_x0, pdf_y0, pdf_x1, pdf_y1 = match["table"]
        
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[match["page"] - 1]
            pdf_width, pdf_height = page.width, page.height
        
        scale_x = page_img.width / pdf_width
        scale_y = page_img.height / pdf_height
        
        # No flipping — use top-left coordinate logic
        x0 = max(0, int(pdf_x0 * scale_x - margin))
        y0 = max(0, int(pdf_y0 * scale_y - margin))
        x1 = min(page_img.width, int(pdf_x1 * scale_x + margin))
        y1 = min(page_img.height, int(pdf_y1 * scale_y + margin))
        
        # Crop the image
        if x0 < x1 and y0 < y1:
            cropped_img = page_img.crop((x0, y0, x1, y1))
        else:
            print(f"Invalid crop dimensions on page {match['page']}")
            return ""
        
        # Convert to base64
        buffered = io.BytesIO()
        cropped_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error extracting table: {str(e)}")
        return ""



# ==========================
# 6. Main Workflow
# ==========================
if __name__ == "__main__":
    pdf_path =  ""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set the environment variable OPENAI_API_KEY")
    import openai
    openai_client = openai.OpenAI(api_key=openai_api_key)

    print("Extracting table captions...")
    captions_by_page = extract_table_captions(pdf_path)
    
    print("Extracting table locations...")
    tables_by_page = extract_tables(pdf_path)
    
    print("Matching captions to tables...")
    matches = match_captions_to_tables(captions_by_page, tables_by_page)

    print("Generating table descriptions using OpenAI and extracting original tables as images...")
    for match in tqdm(matches, desc="Processing matches"):
        description = generate_table_description(match, pdf_path, openai_client, margin=50)
        match["description"] = description
        match["original"] = extract_table_original_using_pdf2image(pdf_path, match, margin=50)

    output_file = "matched_captions_tables.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"matches": matches}, f, ensure_ascii=False, indent=2)

    print(f"Matching and description generation complete. Results saved to {output_file}")
