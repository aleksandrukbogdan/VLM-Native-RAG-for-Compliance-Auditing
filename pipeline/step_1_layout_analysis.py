import json
from pathlib import Path
from tqdm import tqdm
import utils
import prompts

def analyze_layout(images_dir: Path) -> Path:
    """
    Analyzes the layout of images in a directory using Qwen-VL.
    
    Args:
        images_dir (Path): Directory containing page images (e.g., page_1.png).
        
    Returns:
        Path: Directory where layout JSONs are saved.
    """
    
    # Create a subdirectory for layout JSONs
    layout_dir = images_dir / "layout"
    layout_dir.mkdir(exist_ok=True)
    
    # Find all PNG images
    image_files = sorted(list(images_dir.glob("*.png")))
    
    if not image_files:
        print(f"No images found in {images_dir} to analyze.")
        return layout_dir

    print(f"Analyzing layout for {len(image_files)} pages in {images_dir.name}...")

    for image_path in tqdm(image_files, desc="Analyzing Pages"):
        # Define output JSON path
        json_filename = image_path.stem + ".json" # e.g., page_1.json
        json_path = layout_dir / json_filename
        
        # Skip if already exists (simple caching mechanism)
        if json_path.exists():
            continue

        try:
            # Call Qwen-VL
            response_text = utils.call_qwen_vl(image_path, prompts.LAYOUT_ANALYSIS_PROMPT)
            
            # Parse JSON
            layout_data = utils.parse_json_from_response(response_text)
            
            # Save to file
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(layout_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error analyzing {image_path.name}: {e}")
            # Optionally, we could continue to the next page instead of crashing
            continue
            
    print(f"Layout analysis complete. Results saved in {layout_dir}")
    return layout_dir

