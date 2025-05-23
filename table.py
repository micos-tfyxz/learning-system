import pdfplumber
import re
import json
import os
from tqdm import tqdm
import numpy as np
import openai   
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import io
import base64
import fitz 

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

def detect_table_boundaries_dynamic(page_img, initial_bbox, step_size=10, max_expand=100):
    """
    Dynamically detect table boundaries: expand from initial boundary until encountering blank area or page edge
    
    Args:
        page_img: PIL Image object
        initial_bbox: Initial table boundary (x0, y0, x1, y1)
        step_size: Pixel expansion per step
        max_expand: Maximum expansion distance
    
    Returns:
        Optimized bounding box (x0, y0, x1, y1)
    """
    img_array = np.array(page_img.convert('L'))  # Convert to grayscale
    height, width = img_array.shape
    
    x0, y0, x1, y1 = initial_bbox
    
    # Ensure initial boundary is within image bounds
    x0 = max(0, min(x0, width-1))
    y0 = max(0, min(y0, height-1))
    x1 = max(x0+1, min(x1, width))
    y1 = max(y0+1, min(y1, height))
    
    def is_mostly_white(region, threshold=240, white_ratio=0.8):
        """Check if region is mostly white"""
        if region.size == 0:
            return True
        white_pixels = np.sum(region >= threshold)
        return white_pixels / region.size >= white_ratio
    
    def has_table_content(region, threshold=200, content_ratio=0.3):
        """Check if region contains table content (lines, text, etc.)"""
        if region.size == 0:
            return False
        # Detect edges (possible table lines)
        edges = np.abs(np.diff(region.astype(float), axis=0)).sum() + \
                np.abs(np.diff(region.astype(float), axis=1)).sum()
        edge_density = edges / region.size if region.size > 0 else 0
        
        # Detect non-white pixels
        non_white = np.sum(region < threshold)
        non_white_ratio = non_white / region.size
        
        return edge_density > 0.5 or non_white_ratio >= content_ratio
    
    # Expand left
    for expand in range(step_size, max_expand + step_size, step_size):
        new_x0 = max(0, x0 - expand)
        if new_x0 == 0:  # Reached page edge
            break
        # Check newly expanded area
        left_region = img_array[y0:y1, new_x0:x0]
        if is_mostly_white(left_region) and not has_table_content(left_region):
            break
        x0 = new_x0
    
    # Expand right
    for expand in range(step_size, max_expand + step_size, step_size):
        new_x1 = min(width, x1 + expand)
        if new_x1 == width:  # Reached page edge
            break
        # Check newly expanded area
        right_region = img_array[y0:y1, x1:new_x1]
        if is_mostly_white(right_region) and not has_table_content(right_region):
            break
        x1 = new_x1
    
    # Expand up
    for expand in range(step_size, max_expand + step_size, step_size):
        new_y0 = max(0, y0 - expand)
        if new_y0 == 0:  # Reached page edge
            break
        # Check newly expanded area
        top_region = img_array[new_y0:y0, x0:x1]
        if is_mostly_white(top_region) and not has_table_content(top_region):
            break
        y0 = new_y0
    
    # Expand down
    for expand in range(step_size, max_expand + step_size, step_size):
        new_y1 = min(height, y1 + expand)
        if new_y1 == height:  # Reached page edge
            break
        # Check newly expanded area
        bottom_region = img_array[y1:new_y1, x0:x1]
        if is_mostly_white(bottom_region) and not has_table_content(bottom_region):
            break
        y1 = new_y1
    
    return (x0, y0, x1, y1)

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
    Match table captions to tables with improved logic
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
    """Generate a short table description based on its caption and surrounding context."""
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
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        description = response.choices[0].message.content.strip()
        if not description:
            description = f"Fallback Description: Caption: {match['caption']}."
    except Exception as e:
        description = f"Fallback Description: Caption: {match['caption']}. Error: {str(e)}"
    return description

def extract_table_with_dynamic_boundaries(pdf_path, match, base_margin=20, step_size=10, max_expand=100):
    """
    Extract table image using dynamic boundary detection
    
    Args:
        pdf_path: PDF file path
        match: Match object containing page number and table boundary
        base_margin: Base margin
        step_size: Expansion step size
        max_expand: Maximum expansion distance
    """
    try:
        # 1. Convert PDF page to image
        images = convert_from_path(pdf_path, first_page=match["page"], last_page=match["page"], dpi=300)
        if not images:
            print("No images generated from PDF")
            return ""
        
        page_img = images[0]
        
        # 2. Get PDF and image dimension ratios
        pdf_x0, pdf_y0, pdf_x1, pdf_y1 = match["table"]
        
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[match["page"] - 1]
            pdf_width, pdf_height = page.width, page.height
        
        scale_x = page_img.width / pdf_width
        scale_y = page_img.height / pdf_height
        
        # 3. Convert PDF coordinates to image coordinates (add base margin)
        initial_x0 = max(0, int(pdf_x0 * scale_x - base_margin))
        initial_y0 = max(0, int(pdf_y0 * scale_y - base_margin))
        initial_x1 = min(page_img.width, int(pdf_x1 * scale_x + base_margin))
        initial_y1 = min(page_img.height, int(pdf_y1 * scale_y + base_margin))
        
        initial_bbox = (initial_x0, initial_y0, initial_x1, initial_y1)
        
        # 4. Dynamically detect optimal boundary
        optimized_bbox = detect_table_boundaries_dynamic(
            page_img, initial_bbox, step_size=step_size, max_expand=max_expand
        )
        
        x0, y0, x1, y1 = optimized_bbox
        
        print(f"Page {match['page']}: Initial bbox {initial_bbox} -> Optimized bbox {optimized_bbox}")
        
        # 5. Crop image
        if x0 < x1 and y0 < y1:
            cropped_img = page_img.crop((x0, y0, x1, y1))
            
            # Optional: Add debug bounding box
            debug_img = cropped_img.copy()
            draw = ImageDraw.Draw(debug_img)
            w, h = debug_img.size
            draw.rectangle([2, 2, w-3, h-3], outline="red", width=2)
            
        else:
            print(f"Invalid crop dimensions on page {match['page']}: {optimized_bbox}")
            return ""
        
        # 6. Convert to base64
        buffered = io.BytesIO()
        cropped_img.save(buffered, format="PNG")
        
        # Save debug info
        match["boundary_info"] = {
            "initial_bbox": initial_bbox,
            "optimized_bbox": optimized_bbox,
            "expansion": {
                "left": initial_x0 - x0,
                "top": initial_y0 - y0,
                "right": x1 - initial_x1,
                "bottom": y1 - initial_y1
            }
        }
        
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    except Exception as e:
        print(f"Error extracting table with dynamic boundaries: {str(e)}")
        return ""

def save_tables_for_verification(matches, output_dir="extracted_tables"):
    """Save extracted table images as PNG files for manual verification"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, match in enumerate(matches):
        if not match.get("original"):
            continue
            
        try:
            # Decode Base64 image data
            img_data = base64.b64decode(match["original"])
            img = Image.open(io.BytesIO(img_data))
            
            # Generate filename
            caption_snippet = match.get("caption", "table")[:20].replace(" ", "_").replace(":", "")
            filename = f"page_{match['page']}_{i}_{caption_snippet}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save image
            img.save(filepath)
            match["image_filepath"] = filepath
            
            # Print boundary expansion info
            if "boundary_info" in match:
                info = match["boundary_info"]
                print(f"Saved table image: {filepath}")
                print(f"  Boundary expansion - left:{info['expansion']['left']}, top:{info['expansion']['top']}, "
                      f"right:{info['expansion']['right']}, bottom:{info['expansion']['bottom']}")
            
        except Exception as e:
            print(f"Failed to save table image (Match {i}): {str(e)}")

# ==========================
# Main Workflow
# ==========================
if __name__ == "__main__":
    pdf_path = "D:/作业/material.pdf"
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Extract table captions and locations
    print("Extracting table captions...")
    captions_by_page = extract_table_captions(pdf_path)
    
    print("Extracting table locations...")
    tables_by_page = extract_tables(pdf_path)
    
    print("Matching captions to tables...")
    matches = match_captions_to_tables(captions_by_page, tables_by_page)
    
    # Extract table images with dynamic boundaries
    print("Extracting table images with dynamic boundaries...")
    for match in tqdm(matches, desc="Processing tables with dynamic boundaries"):
        # Use dynamic boundary algorithm
        match["original"] = extract_table_with_dynamic_boundaries(
            pdf_path, match, 
            base_margin=20,     # Base margin
            step_size=15,       # Expand 15 pixels per step
            max_expand=120      # Maximum expansion of 120 pixels
        )
        
        # Generate description
        match["description"] = generate_table_description(match, pdf_path, openai_client)
    
    # Save images for verification
    save_tables_for_verification(matches, output_dir="extracted_tables_dynamic")
    
    # Output results
    output_file = "matched_tables_dynamic_boundaries.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"matches": matches}, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")
    print(f"Table images saved in extracted_tables_dynamic/ directory")
    print("Check boundary_info field for boundary expansion details of each table")
