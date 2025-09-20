from sentence_transformers import SentenceTransformer
import chromadb
import os
from config import EMBEDDING_MODEL_NAME, COLLECTION_NAME

# Initialize the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize ChromaDB client 
client = chromadb.Client()

# Create or get a collection
collection_name = COLLECTION_NAME
try:
    collection = client.create_collection(name=collection_name)
except:
    collection = client.get_collection(name=collection_name)


# Prepare data for ChromaDB
ids = []
documents = []
metadatas = []

for i, chunk in enumerate(all_chunks):
    ids.append(f"chunk_{i}")
    documents.append(chunk["content"])
    
    # Process metadata: remove 'coordinates' and convert lists to strings
    processed_metadata = {}
    for key, value in chunk["metadata"].items():
        if key == 'coordinates':
            continue # Skip the coordinates field
        if isinstance(value, list):
            processed_metadata[key] = ", ".join(map(str, value)) # Convert list to string
        else:
            processed_metadata[key] = value
    metadatas.append(processed_metadata)


# Add data to ChromaDB in batches to avoid exceeding payload limits
batch_size = 100  # Adjust batch size as needed
for i in range(0, len(ids), batch_size):
    batch_ids = ids[i:i+batch_size]
    batch_documents = documents[i:i+batch_size]
    batch_metadatas = metadatas[i:i+batch_size]

    # Generate embeddings for the current batch
    batch_embeddings = embedding_model.encode(batch_documents).tolist()

    # Add batch to ChromaDB
    collection.add(
        embeddings=batch_embeddings,
        documents=batch_documents,
        metadatas=batch_metadatas,
        ids=batch_ids
    )
    print(f"Added batch {i//batch_size + 1} to ChromaDB")

print(f"Successfully added {len(ids)} chunks to ChromaDB.")