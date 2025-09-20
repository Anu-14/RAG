def rag_query(query_text, collection, embedding_model, n_results=10):
    # Generate embedding for the query
    query_embedding = embedding_model.encode(query_text).tolist()

    # Perform similarity search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas']
    )

    # Extract relevant documents and metadatas and return them as a list of dictionaries
    retrieved_sources = []
    if results and results['documents'] and results['metadatas']:
        for chunk, metadata in zip(results['documents'][0], results['metadatas'][0]):
            retrieved_sources.append({
                "content": chunk,
                "metadata": metadata
            })

    return retrieved_sources


if __name__=="__main__":
    query = "What are the key risks for Microsoft in 2023?"
    retrieved_context = rag_query(query, collection, embedding_model)
    print(retrieved_context) # This will now print a list of dictionaries