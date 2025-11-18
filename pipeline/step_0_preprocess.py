import config
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm


def convert_pdf_to_images(pdf_path: Path) -> Path:
    """
    Converts a PDF file to a series of images and saves them in a directory.

    Args:
        pdf_path (Path): The path to the input PDF file.

    Returns:
        Path: The path to the directory containing the output images.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    # Create an output directory named after the PDF file
    output_dir = config.OUTPUT_PATH / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {pdf_path.name} to images...")

    # Convert PDF to a list of PIL images
    images = convert_from_path(
        pdf_path=pdf_path,
        dpi=config.IMAGE_DPI,
        fmt=config.IMAGE_FORMAT,
    )

    # Save each image to the output directory
    for i, image in enumerate(tqdm(images, desc="Saving pages")):
        image_path = output_dir / f"page_{i + 1}.{config.IMAGE_FORMAT}"
        image.save(image_path)

    print(f"Successfully converted {len(images)} pages.")
    print(f"Images saved to: {output_dir}")

    return output_dir



