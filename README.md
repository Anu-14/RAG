# Agentic RAG

A Python implementation for Agentic Retrieval-Augmented Generation (RAG) — letting you augment large language model responses with relevant information retrieved from an external corpus or knowledge base.

This project demonstrates a basic pipeline for performing RAG: you provide a dataset (documents), chunk it, compute embeddings, retrieve relevant chunks for a user query, then generate an answer grounded in the retrieved context.

Use-cases include question answering, chatbots, knowledge-base search, etc

### Features

Splitting documents into chunks to allow more fine-grained retrieval.

Computing embeddings for chunks.

Decomposing user queries (optionally) to improve relevance.

A RAG agent to integrate retrieval + generation.

### Getting Started

Here are the steps to run the project locally:

1. Clone the repository
```
git clone https://github.com/Anu-14/RAG.git
cd RAG
```
2. Install requirements
```
pip install -r requirements.txt
```
3. Prepare your data
Put documents you want to use for retrieval inside the data/ folder (or update config to point to your data folder).
Make sure they are in a form the chunker can handle (pdf)
4. Configure settings
Open config.py to set parameters such as embedding model, data_dir, etc.
5. Configure GOOGLE_API_KEY:
For colab:
```
from google.colab import userdata
import os
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

Otherwise set the GOOGLE_API_KEY in config.py and import in query_decomposition.py and rag_agent.py, 
```
6. Run the pipeline with user query
```
python main.py --query "Compare the key risk factors of Microsoft and Google in 2023"
```

### How It Works (High-Level Pipeline)

Document ingestion
Load documents from data/.

Chunking
Split documents into smaller chunks to increase retrieval granularity.

Embedding
Transform text chunks to vector embeddings with (for example) a pre-trained embedding model.

Indexing / Storage
Store embeddings + metadata in a structure or database enabling nearest-neighbor searches.

Query processing
  a. Possibly decompose query into sub-queries.
  b. Embed query.
  c. Retrieve top-k relevant chunks.

Generation
Forward retrieved context to LLM for answer.
