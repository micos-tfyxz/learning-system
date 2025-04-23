import pdfplumber
import re
import json
import os
from tqdm import tqdm
import numpy as np
import openai
import base64
from pdf2image import convert_from_path
from PIL import Image
import io
import fitz  # PyMuPDF

# ==========================
# 1. Image and Caption Extraction and Matching (Improved)
# ==========================

def extract_images_positions(pdf_path):
    """Extract image positions from PDF using pdfplumber."""
    image_positions = {}
    print("\nDebugging image extraction:")
    
    # Open PyMuPDF document once
    doc = None
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Could not open PDF with PyMuPDF: {str(e)}")
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            images = []
            page_height = page.height 
            
            print(f"\nPage {page_num} contains {len(page.images)} image(s):")
            
            # Get PyMuPDF page info for reference if available
            pymupdf_height = None
            if doc and 0 <= (page_num - 1) < doc.page_count:
                try:
                    pymupdf_page = doc[page_num - 1]
                    pymupdf_height = pymupdf_page.rect.height
                    print(f"Page dimensions: pdfplumber={page.width}x{page_height}, PyMuPDF={pymupdf_page.rect.width}x{pymupdf_height}")
                except Exception as e:
                    print(f"Error accessing PyMuPDF page {page_num}: {str(e)}")
            
            for i, img in enumerate(page.images):
                # Convert coordinates consistently - pdfplumber uses bottom-left as origin
                adjusted_y0 = page_height - img['y1']  
                adjusted_y1 = page_height - img['y0']  
                
                adjusted_img = {
                    'x0': img['x0'],
                    'y0': img['y0'],  # Store original y0 (from bottom)
                    'x1': img['x1'],
                    'y1': img['y1'],  # Store original y1 (from bottom)
                    'stream': img.get('stream'),  
                    'name': img.get('name'),
                    'page_width': page.width,     
                    'page_height': page_height
                }
                
                print(f"  Image {i+1} positions:")
                print(f"    Original: [x0={img['x0']:.1f}, y0={img['y0']:.1f}, x1={img['x1']:.1f}, y1={img['y1']:.1f}]")
                print(f"    Adjusted for top-left origin: [x0={img['x0']:.1f}, y0={adjusted_y0:.1f}, x1={img['x1']:.1f}, y1={adjusted_y1:.1f}]")
                
                images.append(adjusted_img)
            image_positions[page_num] = images
    
    # Close PyMuPDF document
    if doc:
        doc.close()
        
    return image_positions

def extract_figure_captions(pdf_path):
    """Extract figure captions using regex patterns."""
    caption_patterns = [
        re.compile(r'Figure\s*\d+[:：]?\s*.*', re.IGNORECASE),
        re.compile(r'Fig\.?\s*\d+[:：]?\s*.*', re.IGNORECASE),
        re.compile(r'Image\s*\d+[:：]?\s*.*', re.IGNORECASE)
    ]
    captions_by_page = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = text.splitlines()
            caps = [line.strip() for line in lines if any(pattern.search(line) for pattern in caption_patterns)]
            if caps:
                captions_by_page[page_num] = caps
                print(f"Page {page_num} captions found: {caps}")
    return captions_by_page

def match_captions_to_images(pdf_path, image_positions, captions_by_page):
    """Match captions to images based on positional relationships."""
    matches = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            images = image_positions.get(page_num, [])
            captions = captions_by_page.get(page_num, [])
            
            print(f"\nMatching page {page_num}: {len(images)} images, {len(captions)} captions")
            
            if not captions or not images:
                continue

            # Try to extract caption positions
            try:
                text_page = page.extract_words()
                caption_positions = []
                
                for caption in captions:
                    # Find first few words from the caption in the extracted text
                    caption_start = ' '.join(caption.split()[:3]).lower()
                    
                    # Find where this caption starts in the page
                    caption_y = None
                    for word in text_page:
                        if caption_start.startswith(word['text'].lower()) or word['text'].lower() in caption_start:
                            # Convert to bottom-origin coordinate system like images
                            caption_y = page.height - word['bottom']
                            break
                    
                    if caption_y is not None:
                        caption_positions.append((caption, caption_y))
                        print(f"  Caption '{caption[:30]}...' at position y={caption_y:.1f}")
                
                # Sort images and captions by y-position
                images_sorted = sorted(images, key=lambda img: img["y0"])
                
                # If we have caption positions, match them to nearest images
                if caption_positions:
                    for caption, cap_y in caption_positions:
                        # Find closest image ABOVE the caption (captions usually appear below images)
                        best_img = None
                        best_dist = float('inf')
                        
                        for img in images_sorted:
                            # Get center of image
                            img_center_y = (img["y0"] + img["y1"]) / 2
                            
                            # Caption is usually below image, so img_center_y should be greater than cap_y
                            if img_center_y > cap_y:
                                dist = img_center_y - cap_y
                                if dist < best_dist:
                                    best_dist = dist
                                    best_img = img
                        
                        # If no image found above, find closest image
                        if best_img is None:
                            for img in images_sorted:
                                dist = abs((img["y0"] + img["y1"]) / 2 - cap_y)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_img = img
                        
                        if best_img:
                            matches.append({"page": page_num, "caption": caption, "matched_image": best_img})
                            print(f"  Matched caption to image with y-pos: {best_img['y0']:.1f}-{best_img['y1']:.1f}")
                else:
                    # Fallback to original logic
                    print("  No caption positions found, using fallback matching")
                    if len(captions) == 1:
                        # Just one caption - use the most central image
                        page_center = (page.width / 2, page.height / 2)
                        dists = [np.hypot(((img["x0"] + img["x1"]) / 2) - page_center[0], 
                                        ((img["y0"] + img["y1"]) / 2) - page_center[1]) for img in images_sorted]
                        best_idx = int(np.argmin(dists)) if dists else 0
                        best_image = images_sorted[best_idx]
                        matches.append({"page": page_num, "caption": captions[0], "matched_image": best_image})
                        print(f"  Matched single caption to central image at y0={best_image['y0']}")
                    else:
                        # Equal counts - match in position order
                        if len(images_sorted) == len(captions):
                            for cap, img in zip(captions, images_sorted):
                                matches.append({"page": page_num, "caption": cap, "matched_image": img})
                                print(f"  Matched caption '{cap[:20]}...' to image at y0={img['y0']}")
                        else:
                            print(f"  Warning: Uneven count - {len(images_sorted)} images vs {len(captions)} captions")
                            # Assign captions to images based on position in document
                            for i, cap in enumerate(captions):
                                if i < len(images_sorted):
                                    matches.append({"page": page_num, "caption": cap, "matched_image": images_sorted[i]})
                                    print(f"  Matched caption '{cap[:20]}...' to image at y0={images_sorted[i]['y0']}")
            
            except Exception as e:
                print(f"Error in caption matching for page {page_num}: {str(e)}")
                # Very simple fallback if everything fails
                for i, cap in enumerate(captions):
                    if i < len(images):
                        matches.append({"page": page_num, "caption": cap, "matched_image": images[i]})
    
    return matches

# ==========================
# 2. Improved Image Extraction Methods (Modified to match table.py logic)
# ==========================

def extract_image_data(pdf_path, match, margin=50):
    """Extract image data using PyMuPDF first, then fallback to pdf2image"""
    print(f"\nExtracting image for page {match['page']}, caption: {match['caption'][:50]}...")
    
    # Method 1: Try PyMuPDF first (more precise)
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(match["page"] - 1)
        
        # Get target coordinates from the match
        target_x0 = match["matched_image"]["x0"]
        target_y0 = match["matched_image"]["y0"]
        target_x1 = match["matched_image"]["x1"]
        target_y1 = match["matched_image"]["y1"]
        
        # Print debugging info
        print(f"Target coordinates (pdfplumber): x0={target_x0:.2f}, y0={target_y0:.2f}, x1={target_x1:.2f}, y1={target_y1:.2f}")
        
        # Important: Convert from pdfplumber coordinate system to PyMuPDF
        # In pdfplumber, y0 is distance from bottom
        # In PyMuPDF, y0 is distance from top
        page_height = page.rect.height
        pymupdf_y0 = page_height - target_y1  # Flip coordinates
        pymupdf_y1 = page_height - target_y0
        
        # Apply margin
        pymupdf_x0 = max(0, target_x0 - margin)
        pymupdf_x1 = min(page.rect.width, target_x1 + margin)
        pymupdf_y0 = max(0, pymupdf_y0 - margin)
        pymupdf_y1 = min(page_height, pymupdf_y1 + margin)
        
        # Create a rectangle to search for images
        search_rect = fitz.Rect(pymupdf_x0, pymupdf_y0, pymupdf_x1, pymupdf_y1)
        print(f"Converted PyMuPDF search rectangle: {search_rect}")
        
        # Try direct area extraction first (more reliable)
        try:
            pix = page.get_pixmap(clip=search_rect, dpi=300)
            img_data = pix.tobytes("png")
            doc.close()
            print("Successfully extracted image area using PyMuPDF")
            return base64.b64encode(img_data).decode("utf-8")
        except Exception as e:
            print(f"Direct area extraction failed: {str(e)}")
            
            # Fallback to image objects if direct extraction fails
            found_image = False
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    # Get the image rectangle
                    for img_info in page.get_image_info():
                        if img_info.get('xref') == xref:
                            img_rect = fitz.Rect(img_info['bbox'])
                            
                            # Check overlap
                            if search_rect.intersects(img_rect):
                                print(f"Found matching image with rect: {img_rect}")
                                base_image = doc.extract_image(xref)
                                if base_image and base_image.get("image"):
                                    found_image = True
                                    doc.close()
                                    print("Successfully extracted image using PyMuPDF")
                                    return base64.b64encode(base_image["image"]).decode("utf-8")
                except Exception as e:
                    print(f"Error processing image xref {xref}: {str(e)}")
                    continue
            
            if not found_image:
                doc.close()
                
    except Exception as e:
        print(f"PyMuPDF extraction failed: {str(e)}")
        if 'doc' in locals() and doc:
            doc.close()
    
    # Method 2: Fallback to pdf2image with fixed coordinate handling
    try:
        print("Falling back to pdf2image method")
        images = convert_from_path(pdf_path, first_page=match["page"], last_page=match["page"], dpi=300)
        if not images:
            return None

        page_img = images[0]
        
        # Get PDF dimensions for scaling
        with pdfplumber.open(pdf_path) as pdf:
            if match["page"] <= len(pdf.pages):
                page = pdf.pages[match["page"] - 1]
                pdf_width = page.width
                pdf_height = page.height
            else:
                pdf_width = page_img.width
                pdf_height = page_img.height
        
        # Calculate scale factor between PDF points and image pixels
        scale_x = page_img.width / pdf_width
        scale_y = page_img.height / pdf_height
        
        # DEBUG: print scaling info
        print(f"PDF dimensions: {pdf_width}x{pdf_height}")
        print(f"Image dimensions: {page_img.width}x{page_img.height}")
        print(f"Scale factors: {scale_x:.2f}x, {scale_y:.2f}y")
        
        # Convert from pdfplumber (bottom-left origin) to image coordinates (top-left origin)
        # with proper scaling
        x0 = max(0, int(match["matched_image"]["x0"] * scale_x - margin))
        # For Y, convert from bottom-origin to top-origin
        y0 = max(0, int((pdf_height - match["matched_image"]["y1"]) * scale_y - margin))
        x1 = min(page_img.width, int(match["matched_image"]["x1"] * scale_x + margin))
        y1 = min(page_img.height, int((pdf_height - match["matched_image"]["y0"]) * scale_y + margin))
        
        print(f"Cropping image at coordinates: ({x0}, {y0}, {x1}, {y1})")

        if x0 >= x1 or y0 >= y1:
            print("Invalid cropping coordinates")
            return None

        cropped_img = page_img.crop((x0, y0, x1, y1))
        
        buffered = io.BytesIO()
        cropped_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"PDF2Image extraction failed: {str(e)}")
        return None

# ==========================
# 3. Generate Image Descriptions
# ==========================

def generate_image_description(match, pdf_path, openai_client, margin=50):
    """Generate an image description using OpenAI."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[match["page"] - 1]
        context_text = page.extract_text() or ""
    
    prompt = (f"Please generate a concise description (within 50 words) for the image based on the following caption and context. "
              f"Caption: {match['caption']}\nContext: {context_text}\nDescription:")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        description = response.choices[0].message.content.strip()
    except Exception as e:
        description = f"Fallback Description: Caption: {match['caption']}. Context: {context_text}. Error: {str(e)}"
    return description

# ==========================
# 4. Main Workflow (Improved)
# ==========================

if __name__ == "__main__":
    # Configuration
    pdf_path = ""
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    output_file = "matched_captions_images_description.json"

    # Step 1: Extract and match
    print("=== Extracting image positions ===")
    image_positions = extract_images_positions(pdf_path)

    print("\n=== Extracting captions ===")
    captions_by_page = extract_figure_captions(pdf_path)

    print("\n=== Matching captions to images ===")
    matches = match_captions_to_images(pdf_path, image_positions, captions_by_page)

    # Step 2: Process matches
    print("\n=== Processing matches ===")
    for match in tqdm(matches, desc="Processing matches"):
        # Extract image data using improved method
        match["image_data"] = extract_image_data(pdf_path, match)
        
        # Generate description
        match["description"] = generate_image_description(match, pdf_path, openai_client)
        
        # Convert matched_image to serializable format
        if isinstance(match["matched_image"], dict):
            match["matched_image"] = [
                match["matched_image"]["x0"],
                match["matched_image"]["y0"],
                match["matched_image"]["x1"],
                match["matched_image"]["y1"]
            ]

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"matches": matches}, f, ensure_ascii=False, indent=2)

    print(f"\nProcessing completed. Results saved to {output_file}")
    print(f"Total matches processed: {len(matches)}")