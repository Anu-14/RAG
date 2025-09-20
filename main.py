import argparse
from sentence_transformers import SentenceTransformer
import chromadb
import json
import google.generativeai as genai
from rag_agent import agentic_rag_query # Assuming agentic_rag_query returns a Python dict
from config import COLLECTION_NAME, EMBEDDING_MODEL_NAME
from embedding import create_embeddings

def main():
    parser = argparse.ArgumentParser(description="Run Agentic RAG Query with multiple queries.")
    parser.add_argument(
        "--query_file",
        type=str,
        default="sample_queries.txt",
        help="Path to a text file containing queries, one per line"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="rag_results.json",
        help="Path to save the output JSON file"
    )

    args = parser.parse_args()

    # Load queries from the text file, one per line
    try:
        with open(args.query_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()] # Read lines and remove empty ones
    except FileNotFoundError:
        print(f"Error: Query file not found at {args.query_file}")
        return
    except Exception as e:
        print(f"Error reading query file: {e}")
        return

    if not queries:
        print("No queries found in the file.")
        return

    # Initialize embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Initialize ChromaDB (in-memory for now, or connect to persistent one)
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Error getting collection: {e}")
        print(f"Creating the collection '{COLLECTION_NAME}' ")
        create_embeddings(COLLECTION_NAME)
        return


    # Load other models
    decomposition_model = genai.GenerativeModel('gemini-2.5-flash')
    synthesis_model = genai.GenerativeModel('gemini-2.5-flash')

    results = []
    for query in queries:
        print(f"Processing query: {query}")
        # Assuming agentic_rag_query now returns a Python dictionary
        agentic_result_dict = agentic_rag_query(
            query, collection, embedding_model, decomposition_model, synthesis_model
        )
        results.append(agentic_result_dict)

    # Save results to a JSON file
    try:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output_file}")
    except IOError as e:
        print(f"Error saving results to {args.output_file}: {e}")


if __name__ == "__main__":
    main()
