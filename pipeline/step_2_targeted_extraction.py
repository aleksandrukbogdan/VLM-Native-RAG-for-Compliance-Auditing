import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import utils
import prompts

def crop_image(image: Image.Image, box: list) -> Image.Image:
    """
    Crops the image based on normalized coordinates [ymin, xmin, ymax, xmax] (0-1000).
    PIL crop expects (left, top, right, bottom).
    """
    width, height = image.size
    ymin, xmin, ymax, xmax = box

    left = (xmin / 1000) * width
    top = (ymin / 1000) * height
    right = (xmax / 1000) * width
    bottom = (ymax / 1000) * height

    # Add a small padding (optional)
    return image.crop((left, top, right, bottom))

def extract_data_from_page(image_path: Path, layout_path: Path, output_dir: Path):
    """
    Extracts data from specific blocks on a page based on layout analysis.
    """
    # Load image
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    # Load layout JSON
    try:
        with open(layout_path, "r", encoding="utf-8") as f:
            layout_data = json.load(f)
    except Exception as e:
        print(f"Error loading layout {layout_path}: {e}")
        return

    page_results = []
    
    # Create a temporary directory for crops (useful for debugging, optional)
    crops_dir = output_dir / "crops" / image_path.stem
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Processing {len(layout_data)} blocks for {image_path.name}...")

    for i, block in enumerate(layout_data):
        block_type = block.get("type")
        box = block.get("box")
        
        if not block_type or not box:
            continue

        # Crop the image
        cropped_image = crop_image(image, box)
        crop_path = crops_dir / f"block_{i}_{block_type}.png"
        cropped_image.save(crop_path)

        # Select Prompt
        prompt = ""
        if block_type == "title_block":
            prompt = prompts.TITLE_BLOCK_PROMPT
        elif block_type == "table":
            prompt = prompts.TABLE_PROMPT
        elif block_type == "drawing":
            prompt = prompts.DRAWING_PROMPT
        elif block_type == "text_block" or block_type == "header":
            prompt = prompts.TEXT_BLOCK_PROMPT
        else:
            continue # Skip unknown types

        # Call API with cropped image
        try:
            response_text = utils.call_qwen_vl(crop_path, prompt)
            
            # Process response based on type
            extracted_content = response_text
            if block_type in ["title_block", "drawing"]:
                try:
                    extracted_content = utils.parse_json_from_response(response_text)
                except:
                    # Fallback if JSON parsing fails, keep raw text
                    pass
            
            # Store result
            block_result = {
                "type": block_type,
                "box": box, # Keep original coordinates
                "content": extracted_content
            }
            page_results.append(block_result)

        except Exception as e:
            print(f"    Error extracting block {i} ({block_type}): {e}")

    return page_results

def run_targeted_extraction(images_dir: Path, layout_dir: Path) -> Path:
    """
    Main function for Step 2.
    """
    extraction_dir = images_dir / "extracted"
    extraction_dir.mkdir(exist_ok=True)
    
    layout_files = sorted(list(layout_dir.glob("*.json")))
    
    if not layout_files:
        print(f"No layout files found in {layout_dir}.")
        return extraction_dir

    print(f"Starting targeted extraction for {len(layout_files)} pages...")

    for layout_file in tqdm(layout_files, desc="Extracting Data"):
        # Determine corresponding image path
        # layout filename is "page_N.json", image is "page_N.png"
        image_filename = layout_file.stem + ".png" 
        image_path = images_dir / image_filename
        
        if not image_path.exists():
            print(f"Warning: Image not found for layout {layout_file.name}")
            continue
            
        output_json_path = extraction_dir / f"{layout_file.stem}_data.json"
        
        # Skip if already exists
        if output_json_path.exists():
            continue

        extracted_data = extract_data_from_page(image_path, layout_file, extraction_dir)
        
        if extracted_data:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=2)

    print(f"Extraction complete. Results saved in {extraction_dir}")
    return extraction_dir

