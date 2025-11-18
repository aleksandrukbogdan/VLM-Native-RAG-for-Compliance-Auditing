import json
from pathlib import Path
from tqdm import tqdm
import config

def create_chunk_object(content: str, chunk_type: str, metadata: dict, page_num: int) -> dict:
    """
    Helper to create a standardized chunk object.
    """
    return {
        "content": content,
        "metadata": {
            **metadata, # Include document/sheet metadata
            "page_number": page_number,
            "type": chunk_type, # 'text', 'table', 'drawing'
            "source": "AI_Parser"
        }
    }

def process_page_chunks(page_data: dict) -> list:
    """
    Dynamically splits a page JSON into logical chunks.
    """
    chunks = []
    page_num = page_data.get("page_number", 0)
    
    # 1. Base Metadata (Project Info from Title Block)
    # We attach this to EVERY chunk so the model knows context (e.g. "Project Number: 123")
    base_metadata = page_data.get("metadata", {})
    
    # Remove large raw text fields from metadata if they exist, to keep metadata lean
    if "raw_text" in base_metadata:
        del base_metadata["raw_text"]

    # 2. Tables -> Independent Chunks
    for table in page_data.get("tables", []):
        content = table.get("markdown", "")
        if len(content) > config.MIN_CHUNK_SIZE:
            chunks.append(create_chunk_object(content, "table", base_metadata, page_num))

    # 3. Drawings -> Independent Chunks
    for drawing in page_data.get("drawings", []):
        # Construct a rich description for the drawing chunk
        parts = []
        if "description" in drawing:
            parts.append(f"Описание чертежа: {drawing['description']}")
        
        # Add OCR content
        if "content" in drawing:
             # If it's a list of text boxes
            if isinstance(drawing["content"], list):
                texts = [item.get("text", "") for item in drawing["content"] if isinstance(item, dict)]
                if texts:
                    parts.append(f"Текст на чертеже: {', '.join(texts)}")
            # If it's just text (fallback)
            elif isinstance(drawing["content"], str):
                parts.append(f"Текст: {drawing['content']}")
        
        full_content = "\n".join(parts)
        if len(full_content) > config.MIN_CHUNK_SIZE:
            chunks.append(create_chunk_object(full_content, "drawing", base_metadata, page_num))

    # 4. Text Blocks -> Dynamic Grouping
    # We want to merge small paragraphs, but respect headers.
    text_buffer = []
    current_buffer_size = 0
    
    raw_blocks = page_data.get("text_blocks", [])
    
    for block in raw_blocks:
        # Clean the block
        text = block.strip()
        if not text:
            continue
            
        text_len = len(text)
        
        # Check if we should flush the current buffer
        # Conditions:
        # A. Buffer + New Block exceeds target size
        # B. (Optional) The new block looks like a distinct Header (heuristic: short, uppercase, ends with nothing or colon)
        
        is_likely_header = (text_len < 100 and text.isupper())
        
        if (current_buffer_size + text_len > config.TARGET_CHUNK_SIZE) or (is_likely_header and text_buffer):
            # Flush buffer
            if text_buffer:
                joined_content = "\n\n".join(text_buffer)
                chunks.append(create_chunk_object(joined_content, "text", base_metadata, page_num))
                text_buffer = []
                current_buffer_size = 0
        
        # Add current block to buffer
        text_buffer.append(text)
        current_buffer_size += text_len

    # Flush remaining buffer
    if text_buffer:
        joined_content = "\n\n".join(text_buffer)
        chunks.append(create_chunk_object(joined_content, "text", base_metadata, page_num))

    return chunks

def run_chunking(input_dir: Path) -> Path:
    """
    Main function for Step 4: Chunking.
    Reads assembled JSONs and outputs a flat list of chunks.
    """
    # Input: final_json folder from Step 3
    source_dir = input_dir / "final_json"
    if not source_dir.exists():
        print(f"Input directory {source_dir} does not exist. Run Step 3 first.")
        return None

    # Output: chunks folder
    output_dir = input_dir / "chunks"
    output_dir.mkdir(exist_ok=True)

    files = sorted(list(source_dir.glob("*.json")))
    print(f"Generating chunks from {len(files)} documents...")

    all_chunks = []
    
    for file_path in tqdm(files, desc="Chunking"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            page_chunks = process_page_chunks(data)
            all_chunks.extend(page_chunks)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # Save all chunks to a single file (good for small-medium projects)
    # For huge projects, you might want to save batches.
    output_file = output_dir / "all_chunks.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Chunking complete. Generated {len(all_chunks)} chunks.")
    print(f"Saved to: {output_file}")
    
    return output_file

