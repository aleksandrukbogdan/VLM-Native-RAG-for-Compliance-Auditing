from pathlib import Path
import os
from dotenv import load_dotenv

#  Get the root directory of the project
ROOT_PATH = Path(__file__).parent

#  Define paths for data directories
DATA_PATH = ROOT_PATH / "data"
INPUT_PATH = DATA_PATH / "input"
OUTPUT_PATH = DATA_PATH / "output"

#  Create directories if they don't exist
INPUT_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# --- PDF Preprocessing Settings ---
IMAGE_FORMAT = "png"  # Format for saved images (e.g., png, jpeg)
IMAGE_DPI = 300       # Dots per inch for rendering PDF pages


# --- Chunking Settings ---
TARGET_CHUNK_SIZE = 1000  # Target characters per text chunk
MIN_CHUNK_SIZE = 50       # Minimum characters to be considered a valid chunk


# --- Vector Store Settings ---
EMBEDDING_MODEL_NAME = "deepvk/USER-bge-m3"
CHROMA_DB_PATH = DATA_PATH / "chroma_db"
COLLECTION_NAME = "project_docs"

# --- API Settings ---
load_dotenv()  # Load environment variables from .env file

# API Key (required)
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "dummy").strip()

# Base URL: defaults to Alibaba Cloud, but can be overridden for local server
_base_url = os.getenv("QWEN_BASE_URL", "http://localhost:8880/v1").strip()
# Fix potential double quotes issue from .env
if _base_url.startswith('"') and _base_url.endswith('"'):
    _base_url = _base_url[1:-1]
if _base_url.startswith("'") and _base_url.endswith("'"):
    _base_url = _base_url[1:-1]

QWEN_BASE_URL = _base_url

# Model Name: defaults to qwen-vl-max, can be overridden for local model name
_model_name = os.getenv("QWEN_MODEL_NAME", "qwen-vl-max").strip()
if _model_name.startswith('"') and _model_name.endswith('"'):
    _model_name = _model_name[1:-1]
QWEN_MODEL_NAME = _model_name
