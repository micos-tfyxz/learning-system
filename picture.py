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
import fitz  

# ==========================
# 1. Image and Caption Extraction and Matching 
# ==========================

def extract_images_positions(pdf_path):
    """Extract image positions from PDF using pdfplumber."""
    image_positions = {}
    
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
    """Match captions to images with font check (bold) and closest distance rule, ensuring each image is used once."""
    matches = []
    used_images = set()

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            images = image_positions.get(page_num, [])
            if not images:
                continue

            text_words = page.extract_words(extra_attrs=["fontname", "size"])
            captions = captions_by_page.get(page_num, [])
            next_captions = captions_by_page.get(page_num + 1, []) if page_num < total_pages else []

            print(f"\nPage {page_num}: {len(images)} images, {len(captions)} captions")

            image_candidates = [
                {"img": img, "used": False} for img in sorted(images, key=lambda i: i["y0"], reverse=True)
            ]

            caption_infos = []

            # Analyze each caption's position and font
            for cap in captions:
                cap_words = cap.split()
                prefix = " ".join(cap_words[:3]).lower()
                found_y = None
                is_bold = False

                for word in text_words:
                    if word["text"].lower() in prefix:
                        found_y = page.height - word["bottom"]
                        # Check if font is bold (heuristic)
                        if "Bold" in word["fontname"] or word["fontname"].endswith("Bd"):
                            is_bold = True
                        break

                if found_y:
                    caption_infos.append({
                        "text": cap,
                        "y": found_y,
                        "is_bold": is_bold,
                        "source": "current"
                    })

            # Also check next page's top captions
            if not caption_infos and next_captions:
                next_text_words = pdf.pages[page_num].extract_words(extra_attrs=["fontname", "size"])
                for cap in next_captions:
                    cap_words = cap.split()
                    prefix = " ".join(cap_words[:3]).lower()
                    found_y = None
                    is_bold = False
                    for word in next_text_words:
                        if word["text"].lower() in prefix:
                            found_y = pdf.pages[page_num].height - word["bottom"]
                            if "Bold" in word["fontname"] or word["fontname"].endswith("Bd"):
                                is_bold = True
                            break
                    if found_y:
                        caption_infos.append({
                            "text": cap,
                            "y": found_y,
                            "is_bold": is_bold,
                            "source": "next"
                        })

            for cap_info in caption_infos:
                caption_text = cap_info["text"]
                caption_y = cap_info["y"]
                is_bold = cap_info["is_bold"]

                best_img = None
                best_dist = float("inf")

                for candidate in image_candidates:
                    if candidate["used"]:
                        continue
                    img = candidate["img"]
                    img_center_y = (img["y0"] + img["y1"]) / 2
                    dist = abs(img_center_y - caption_y)

                    # Prefer images above caption (standard layout)
                    if img_center_y > caption_y:
                        dist *= 0.9  # prioritize upper images slightly

                    if dist < best_dist:
                        best_img = candidate
                        best_dist = dist

                if best_img:
                    best_img["used"] = True
                    matches.append({
                        "page": page_num,
                        "caption": caption_text,
                        "matched_image": best_img["img"],
                        "is_bold_caption": is_bold
                    })
                    print(f"Matched caption '{caption_text[:30]}...' (bold={is_bold}) to image at y0={best_img['img']['y0']:.1f}")

    return matches


# ==========================
# 2. Improved Image Extraction Methods (Modified to match table.py logic)
# ==========================

def extract_image_data(pdf_path, match, margin=50):
    """Extract image data using PyMuPDF first, then fallback to pdf2image, with careful handling to exclude captions"""
    print(f"\nExtracting image for page {match['page']}, caption: {match['caption'][:50]}...")
    
    # Method 1: Try PyMuPDF first (most precise for native PDF images)
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(match["page"] - 1)
        
        # Get target coordinates from the match
        target_x0 = match["matched_image"]["x0"]
        target_y0 = match["matched_image"]["y0"]
        target_y1 = match["matched_image"]["y1"]
        target_x1 = match["matched_image"]["x1"]
        
        # Convert coordinates to PyMuPDF system (y-origin at top)
        page_height = page.rect.height
        pymupdf_y0 = page_height - target_y1
        pymupdf_y1 = page_height - target_y0
        
        # Create a tight rectangle around the image (no margin initially)
        tight_rect = fitz.Rect(target_x0, pymupdf_y0, target_x1, pymupdf_y1)
        
        # Strategy 1: Try to extract the native image object
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            if base_image and base_image.get("image"):
                # Check if this image overlaps with our target area
                img_info = next((i for i in page.get_image_info() if i['xref'] == xref), None)
                if img_info:
                    img_rect = fitz.Rect(img_info['bbox'])
                    if tight_rect.intersects(img_rect):
                        print("Extracted native PDF image object")
                        doc.close()
                        return base64.b64encode(base_image["image"]).decode("utf-8")
        
        # Strategy 2: If native extraction failed, try direct area cropping
        # First attempt: tight crop (no margin)
        try:
            pix = page.get_pixmap(clip=tight_rect, dpi=300)
            img_data = pix.tobytes("png")
            doc.close()
            print("Extracted tight crop of image area")
            return base64.b64encode(img_data).decode("utf-8")
        except Exception as e:
            print(f"Tight crop failed: {str(e)}")
            
            # Second attempt: small margin if tight crop fails
            margin_rect = tight_rect + (-5, -5, 5, 5)  # Small 5-point margin
            try:
                pix = page.get_pixmap(clip=margin_rect, dpi=300)
                img_data = pix.tobytes("png")
                doc.close()
                print("Extracted image with small margin")
                return base64.b64encode(img_data).decode("utf-8")
            except Exception as e:
                print(f"Small margin crop also failed: {str(e)}")
        
        doc.close()
    except Exception as e:
        print(f"PyMuPDF extraction failed: {str(e)}")
        if 'doc' in locals() and doc:
            doc.close()
    
    # Method 2: Fallback to pdf2image only when absolutely necessary
    try:
        print("Falling back to pdf2image with tight cropping")
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
        
        # Calculate scale factors
        scale_x = page_img.width / pdf_width
        scale_y = page_img.height / pdf_height
        
        # Convert coordinates with tight cropping (no margin)
        x0 = int(match["matched_image"]["x0"] * scale_x)
        y0 = int((pdf_height - match["matched_image"]["y1"]) * scale_y)
        x1 = int(match["matched_image"]["x1"] * scale_x)
        y1 = int((pdf_height - match["matched_image"]["y0"]) * scale_y)
        
        print(f"Cropping image at tight coordinates: ({x0}, {y0}, {x1}, {y1})")
        
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

def generate_image_description(match, pdf_path, openai_client):
    """Generate an image description using OpenAI's vision capabilities combined with the caption."""
    # First extract the image data
    image_data = match.get("image_data")
    if not image_data:
        return "No image data available for description"
    
    # Prepare the image content for GPT-4 Vision
    image_url = f"data:image/png;base64,{image_data}"
    
    # Get the caption text
    caption = match.get("caption", "No caption available")
    
    # Prepare the messages for GPT-4 Vision
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an expert at analyzing technical figures from academic papers. "
                        "Please generate a concise yet comprehensive description of this image "
                        "by combining visual analysis with the provided caption. "
                        "Focus on:\n"
                        "1. The main visual elements and their relationships\n"
                        "2. Key patterns or trends shown\n"
                        "3. How the visual elements relate to the caption\n"
                        "4. Any important details that stand out\n\n"
                        f"Caption: {caption}\n\n"
                        "Description:"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high"  
                    }
                }
            ]
        }
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
            temperature=0.2  # Lower temperature for more factual descriptions
        )
        description = response.choices[0].message.content.strip()
        
        # Post-process the description to ensure quality
        description = description.replace("The image shows", "").strip()
        description = description.replace("This image depicts", "").strip()
        if not description.endswith("."):
            description += "."
            
        return description
    except Exception as e:
        print(f"Error generating image description: {str(e)}")
        # Fallback description using just the caption
        return f"Technical figure showing: {caption}"


def save_images_for_verification(matches, output_dir="extracted_images"):
    """Save extracted images as PNG files for visual verification."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, match in enumerate(matches):
        if not match.get("image_data"):
            continue
            
        try:
            # Decode base64 image data
            img_data = base64.b64decode(match["image_data"])
            img = Image.open(io.BytesIO(img_data))
            
            # Generate filename with page number and caption snippet
            caption_snippet = match["caption"][:30].replace(" ", "_").replace(".", "").replace(":", "")
            filename = f"page_{match['page']}_{i}_{caption_snippet}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save image
            img.save(filepath)
            print(f"Saved verification image: {filepath}")
            
            # Also add the filepath to the match dictionary
            match["image_filepath"] = filepath
            
        except Exception as e:
            print(f"Error saving image for match {i}: {str(e)}")

# Modify the main workflow to include image saving
if __name__ == "__main__":
    # Configuration
    pdf_path =""
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    output_file = "matched_captions_images_description.json"
    image_output_dir = "extracted_images_verification"  # New directory for verification images

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

    # Save images for verification
    print("\n=== Saving images for verification ===")
    save_images_for_verification(matches, image_output_dir)

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"matches": matches}, f, ensure_ascii=False, indent=2)
