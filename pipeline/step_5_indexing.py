import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import config
from tqdm import tqdm
import shutil

# Custom Embedding Function class to wrap SentenceTransformer for Chroma
class LocalEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model_name):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        return self.model.encode(input, batch_size=4, convert_to_tensor=False).tolist()

def run_indexing(chunks_file: str):
    """
    Reads chunks from JSON and indexes them into ChromaDB using a local DeepVK model.
    """
    if not chunks_file.exists():
        print(f"Chunks file not found: {chunks_file}")
        return

    print("Loading chunks...")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    if not chunks:
        print("No chunks to index.")
        return
        
    print(f"Loaded {len(chunks)} chunks. initializing Vector Store...")

    # Initialize Chroma Client
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
    
    # Initialize Embedding Function (DeepVK)
    embedding_fn = LocalEmbeddingFunction(config.EMBEDDING_MODEL_NAME)

    # Get or Create Collection
    # We delete the existing one to ensure a fresh index
    try:
        client.delete_collection(name=config.COLLECTION_NAME)
        print(f"Deleted existing collection '{config.COLLECTION_NAME}'")
    except Exception:
        pass # Collection didn't exist or error deleting

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    # Prepare Batch Data
    ids = []
    documents = []
    metadatas = []

    print("Indexing chunks...")
    for i, chunk in enumerate(tqdm(chunks)):
        # Create unique ID
        chunk_id = f"chunk_{i}"
        
        # Prepare content
        content = chunk.get("content", "")
        if not content:
            continue

        # Prepare metadata (ensure flat dictionary)
        # Chroma metadata values must be str, int, float, or bool
        meta = chunk.get("metadata", {}).copy()
        
        # Flatten or stringify complex metadata if necessary
        clean_meta = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        
        ids.append(chunk_id)
        documents.append(content)
        metadatas.append(clean_meta)

    # Add to Chroma in batches (to avoid memory issues)
    batch_size = 5 # Smaller batch size for local embedding
    total_batches = (len(ids) + batch_size - 1) // batch_size

    for b in tqdm(range(total_batches), desc="Vectorizing"):
        start_idx = b * batch_size
        end_idx = start_idx + batch_size
        
        batch_ids = ids[start_idx:end_idx]
        batch_docs = documents[start_idx:end_idx]
        batch_meta = metadatas[start_idx:end_idx]
        
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta
        )

    print(f"Successfully indexed {len(ids)} chunks into '{config.CHROMA_DB_PATH}'")
    return config.CHROMA_DB_PATH
