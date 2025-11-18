import config
from pipeline.step_0_preprocess import convert_pdf_to_images
from pipeline.step_1_layout_analysis import analyze_layout
from pipeline.step_2_targeted_extraction import run_targeted_extraction
from pipeline.step_3_assembly import run_assembly
from pipeline.step_4_chunking import run_chunking
from pipeline.step_5_indexing import run_indexing

def main():
    """
    Main function to run the entire OCR pipeline.
    """
    print("Starting OCR pipeline...")

    # --- Step 0: Pre-processing (PDF -> Images) ---
    # Find the first PDF file in the input directory
    try:
        pdf_file = next(config.INPUT_PATH.glob("*.pdf"))
        print(f"Found PDF file: {pdf_file.name}")
    except StopIteration:
        print(f"No PDF files found in {config.INPUT_PATH}")
        print("Please add a PDF file to the input directory and run again.")
        return

    # Convert the PDF to images
    image_folder = convert_pdf_to_images(pdf_file)
    
    # --- Step 1: Layout Analysis ---
    layout_folder = analyze_layout(image_folder)

    # --- Step 2: Targeted Extraction ---
    extracted_folder = run_targeted_extraction(image_folder, layout_folder)

    # --- Step 3: Assembly ---
    final_json_folder = run_assembly(image_folder, extracted_folder)

    # --- Step 4: Chunking ---
    chunk_file = run_chunking(image_folder)

    # --- Step 5: Indexing ---
    run_indexing(chunk_file)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()



