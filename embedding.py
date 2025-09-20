import os
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements
import json
import re
from sentence_transformers import SentenceTransformer
import chromadb
from config import DATA_DIR, EMBEDDING_MODEL_NAME, COLLECTION_NAME

data_dir = DATA_DIR
pdf_files = os.listdir(data_dir)

all_chunks = []
overlap_size = 100 # Define overlap size (in characters)

for file_name in pdf_files:
    file_path = os.path.join(data_dir, file_name)
    try:
        elements = partition_pdf(
            filename=file_path,
            # Keeping basic chunking strategy for initial element extraction,
            # we will handle overlapping and custom headers manually.
            chunking_strategy=None, # Disable automatic chunking here
            extract_tables_as_elements=True # Extract tables as separate elements
        )

        # Extract company and year from filename
        match = re.match(r'([a-zA-Z]+)-10-k-(\d{4})\.pdf', file_name)
        company = match.group(1).upper() if match else file_name
        year = match.group(2) if match else 'N/A'

        # Process elements to create chunks with metadata, custom header, and overlap
        current_chunk_text = ""
        current_metadata = {}
        for i, element in enumerate(elements):
            element_text = str(element)
            element_metadata = element.metadata.to_dict()

            # Update metadata with source and page number
            element_metadata['source'] = file_name
            element_metadata['page_number'] = element_metadata.get('page_number', 'N/A')

            # Create the custom header
            header = f"This excerpt is from company {company} FY {year}.\n"

            # Add element to the current chunk text
            current_chunk_text += element_text + "\n" # Add a newline for separation

            # Update metadata (simple approach: use metadata from the last element in the chunk)
            current_metadata = element_metadata

            # If current chunk is large enough or it's the last element, create a chunk
            if len(current_chunk_text) >= 300 or i == len(elements) - 1:
                # Add the header to the beginning of the chunk content
                chunk_content = header + current_chunk_text.strip()

                chunk = {
                    "content": chunk_content,
                    "metadata": current_metadata
                }
                all_chunks.append(chunk)

                # Prepare for the next chunk with overlap
                # Keep the last 'overlap_size' characters for the next chunk
                current_chunk_text = current_chunk_text[-overlap_size:].lstrip() # Use lstrip to avoid leading whitespace from overlap


        # Handle remaining text if any
        if current_chunk_text.strip():
             header = f"This excerpt is from company {company} FY {year}.\n"
             chunk_content = header + current_chunk_text.strip()
             chunk = {
                "content": chunk_content,
                "metadata": current_metadata # Use metadata of the last processed element
             }
             all_chunks.append(chunk)


    except Exception as e:
        print(f"Error processing {file_name}: {e}")

print(f"Created {len(all_chunks)} chunks.")

with open("chunks_with_metadata.json", "w") as f:
    json.dump(all_chunks, f, indent=4)

print("Chunks saved to chunks_with_metadata.json")

# Initialize the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize ChromaDB client 
client = chromadb.PersistentClient(path="./chroma_db")

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
