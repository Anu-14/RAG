import argparse
from sentence_transformers import SentenceTransformer
import chromadb

# Import your function
from rag_agent import agentic_rag_query  
from config import COLLECTION_NAME, EMBEDDING_MODEL_NAME

def main():
    parser = argparse.ArgumentParser(description="Run Agentic RAG Query")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Complex query to ask the RAG system"
    )

    args = parser.parse_args()

    # Initialize embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Initialize ChromaDB (in-memory for now, or connect to persistent one)
    client = chromadb.Client()
    collection = client.get_collection(COLLECTION_NAME)

    # Load other models
    decomposition_model = genai.GenerativeModel('gemini-2.5-flash')
    synthesis_model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # Run the query
    complex_query = args.query
    agentic_result_json = agentic_rag_query(
        complex_query, collection, embedding_model, decomposition_model, synthesis_model
    )

    print(json.dumps(agentic_result_json, indent=4))


if __name__ == "__main__":
    main()
