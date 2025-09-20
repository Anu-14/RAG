import json
import google.generativeai as genai
from google.colab import userdata
import os
import re

# Ensure the Gemini model for synthesis is initialized
# Use an appropriate Gemini model name, e.g., 'gemini-1.5-flash-latest' or 'gemini-1.0-pro'
try:
    synthesis_model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    print(f"Could not initialize synthesis model: {e}")
    print("Please ensure you have access to the specified Gemini model and your API key is correct.")
    synthesis_model = None # Set to None if model initialization fails


def agentic_rag_query(complex_query: str, collection, embedding_model, decomposition_model: genai.GenerativeModel, synthesis_model: genai.GenerativeModel, n_results_per_subquery: int = 3):
    """
    Executes an agentic RAG query by decomposing the complex query, performing
    multi-step retrieval, and synthesizing the results.

    Args:
        complex_query: The original complex user query.
        collection: The ChromaDB collection.
        embedding_model: The embedding model for RAG queries.
        decomposition_model: The Gemini model for query decomposition.
        synthesis_model: The Gemini model for synthesizing the final answer.
        n_results_per_subquery: The number of results to retrieve for each sub-query.

    Returns:
        A dictionary containing the question, answer, reasoning, sub-queries, and sources.
    """
    if decomposition_model is None or synthesis_model is None:
        print("LLM models not initialized. Cannot perform agentic RAG query.")
        return {
            "question": complex_query,
            "answer": "Error: LLM models not initialized.",
            "reasoning": "LLM models required for decomposition and synthesis are not available.",
            "sub_queries": [],
            "sources": []
        }

    # Step 1: Query Decomposition
    print("Performing query decomposition...")
    sub_queries = decompose_query(complex_query, decomposition_model)
    print(f"Decomposed into sub-queries: {sub_queries}")

    # Step 2: Multi-step Retrieval
    print("Performing multi-step retrieval...")
    all_retrieved_sources = []
    for sub_query in sub_queries:
        print(f"Retrieving for sub-query: {sub_query}")
        retrieved_sources = rag_query(sub_query, collection, embedding_model, n_results=n_results_per_subquery)
        all_retrieved_sources.extend(retrieved_sources)

    # Prepare combined context and structured sources for synthesis and output
    combined_context = ""
    structured_sources = []
    for source in all_retrieved_sources:
        # Extract company and year from filename (assuming filename format like 'company-10-k-year.pdf')
        filename = source['metadata'].get('source', '')
        match = re.match(r'([a-zA-Z]+)-10-k-(\d{4})\.pdf', filename)
        company = match.group(1).upper() if match else filename
        year = match.group(2) if match else 'N/A'

        structured_sources.append({
            "company": company,
            "year": year,
            "excerpt": source['content'],
            "page": source['metadata'].get('page_number', 'N/A')
        })
        combined_context += f"Source: {company} ({year}), Page: {source['metadata'].get('page_number', 'N/A')}\nContent: {source['content']}\n\n"


    # Step 3: Synthesis
    print("Synthesizing final answer...")
    try:
        synthesis_prompt = f"""\
        You are an expert financial assistant. You are given a user query and a set of retrieved context chunks that contain relevant information. 
        Your task is to generate a clear, concise, and well-structured final answer.

        Instructions:
        1. Read the user query carefully.
        2. Review all provided chunks of text. Use only the information in these chunks.
        3. Identify the most relevant parts of the chunks to answer the query.
        4. If multiple chunks contribute, synthesize them into a single coherent response.
        5. If the query is comparative, highlight differences, similarities, or trends.
        6. If information is missing or not found in the chunks, state that clearly instead of guessing.

        Output Requirements:
        - Final answer should be factual, concise, and directly address the user query.
        - Present information in a structured form (bullet points, tables, or short paragraphs) if appropriate.
        - Do not include irrelevant details or repeat text verbatim from chunks unless necessary.
        - Never invent information not supported by the chunks.

        Context:
        {combined_context}

        Synthesized Answer:
        """
        synthesis_response = synthesis_model.generate_content(synthesis_prompt)
        final_answer = synthesis_response.text.strip()
        print("Synthesis complete.")
    except Exception as e:
        print(f"Error during synthesis: {e}")
        final_answer = "Error during synthesis."


    # Step 4: Format Output as JSON
    result = {
        "question": complex_query,
        "answer": final_answer,
        "reasoning": "Query decomposed, multi-step retrieval performed, and results synthesized.",
        "sub_queries": sub_queries,
        "sources": structured_sources # Use the structured_sources list
    }

    return json.dumps(result, indent=4)

if __name__=="__main__":
    complex_query = "Compare the revenue growth and key risks of Microsoft and Google in 2023."
    agentic_result_json = agentic_rag_query(complex_query, collection, embedding_model, decomposition_model, synthesis_model)
    print(agentic_result_json)
