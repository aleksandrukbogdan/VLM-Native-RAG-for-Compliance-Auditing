import json
from pathlib import Path
from tqdm import tqdm
import re

def assemble_page_data(extracted_data: list, page_number: int) -> dict:
    """
    Assembles extracted blocks into a single structured JSON document.
    """
    final_doc = {
        "page_number": page_number,
        "metadata": {},
        "tables": [],
        "drawings": [],
        "text_blocks": [],
        "full_text_content": "" # Aggregate text for search indexing
    }

    full_text_parts = []

    for block in extracted_data:
        b_type = block.get("type")
        content = block.get("content")

        if not content:
            continue

        if b_type == "title_block":
            # Merge title block data into metadata
            if isinstance(content, dict):
                final_doc["metadata"].update(content)
                # Add values to full text for searchability
                full_text_parts.extend([str(v) for v in content.values() if v])
            else:
                # Fallback if it's just text
                final_doc["metadata"]["raw_text"] = str(content)

        elif b_type == "table":
            table_entry = {
                "markdown": content if isinstance(content, str) else str(content)
            }
            final_doc["tables"].append(table_entry)
            full_text_parts.append(str(content))

        elif b_type == "drawing":
            # Content might be a list of text items with boxes
            if isinstance(content, list):
                final_doc["drawings"].extend(content)
                # Extract just the text for indexing
                drawing_texts = [item.get("text", "") for item in content if isinstance(item, dict)]
                full_text_parts.extend(drawing_texts)
            elif isinstance(content, dict):
                 # Sometimes model returns a single object
                 final_doc["drawings"].append(content)
                 
                 # Add description to full text
                 if "description" in content:
                     full_text_parts.append(str(content["description"]))
                 
                 # Add extracted text content to full text
                 if "content" in content and isinstance(content["content"], list):
                      drawing_texts = [item.get("text", "") for item in content["content"] if isinstance(item, dict)]
                      full_text_parts.extend(drawing_texts)
            else:
                 # Fallback
                 pass

        elif b_type == "text_block" or b_type == "header":
             final_doc["text_blocks"].append(str(content))
             full_text_parts.append(str(content))

    # Construct full text content
    final_doc["full_text_content"] = "\n\n".join(full_text_parts)
    
    return final_doc

def run_assembly(images_dir: Path, extracted_dir: Path) -> Path:
    """
    Main function for Step 3: Assembly.
    """
    final_output_dir = images_dir / "final_json"
    final_output_dir.mkdir(exist_ok=True)
    
    # Find extraction results
    extracted_files = sorted(list(extracted_dir.glob("*_data.json")))
    
    if not extracted_files:
        print(f"No extracted data found in {extracted_dir}.")
        return final_output_dir

    print(f"Assembling final documents for {len(extracted_files)} pages...")

    for file_path in tqdm(extracted_files, desc="Assembling"):
        # Parse page number from filename (page_1_data.json)
        match = re.search(r"page_(\d+)", file_path.name)
        page_num = int(match.group(1)) if match else 0
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            final_doc = assemble_page_data(data, page_num)
            
            output_path = final_output_dir / f"page_{page_num}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_doc, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error assembling {file_path.name}: {e}")

    print(f"Assembly complete. Final JSONs saved in {final_output_dir}")
    return final_output_dir

