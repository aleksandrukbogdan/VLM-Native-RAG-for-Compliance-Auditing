import chromadb
from sentence_transformers import SentenceTransformer
import config
import argparse

# Re-use the same embedding function wrapper
class LocalEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        return self.model.encode(input, convert_to_tensor=False).tolist()

def query_database(query_text: str, n_results: int = 3):
    """
    Queries the ChromaDB for relevant chunks.
    """
    print(f"Querying: '{query_text}'...")
    
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_PATH))
    
    embedding_fn = LocalEmbeddingFunction(config.EMBEDDING_MODEL_NAME)
    
    try:
        collection = client.get_collection(
            name=config.COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    # Display Results
    print("\n--- Found Results ---")
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        content = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i] if 'distances' in results else "N/A"

        print(f"\nResult #{i+1} (ID: {doc_id})")
        print(f"Page: {meta.get('page_number', '?')} | Type: {meta.get('type', '?')}")
        print("-" * 40)
        print(content[:300] + "..." if len(content) > 300 else content)
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the RAG system.")
    parser.add_argument("query", type=str, help="The question to ask.")
    args = parser.parse_args()
    
    query_database(args.query)

