import os
from dotenv import load_dotenv

import google.generativeai as genai
from google.colab import userdata
import re # Import regular expression module

# Initialize the Gemini API
# Make sure you have added your GOOGLE_API_KEY to Colab secrets
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Generative Model
# Use an appropriate Gemini model name, e.g., 'gemini-1.5-flash-latest' or 'gemini-1.0-pro'
try:
    decomposition_model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    print(f"Could not initialize model: {e}")
    print("Please ensure you have access to the specified Gemini model and your API key is correct.")
    decomposition_model = None # Set to None if model initialization fails


def decompose_query(query: str, model: genai.GenerativeModel) -> list[str]:
    """
    Decomposes a complex query into simpler sub-queries using a Gemini LLM.

    Args:
        query: The complex user query.
        model: The initialized Gemini GenerativeModel.

    Returns:
        A list of sub-queries.
    """
    if model is None:
        print("LLM model is not initialized. Cannot perform query decomposition.")
        return [query] # Return original query if model is not available

    try:
        prompt = f"""\
        You are an intelligent query agent designed to answer complex questions using a structured, multi-step approach. 
        Your task is to break down complex queries, perform multiple retrievals if needed, and synthesize coherent, accurate responses.

        Query Decomposition:
          - Analyze the user’s question to determine if it involves multiple aspects or comparative elements.
          - Break the question into smaller, queries if they need retrieval.
          - Preserve logical dependencies between sub-queries.
          - Do not hallucinate.
          - Do not leave any information.
          - Do not add anything extra.
        {query}

        Provide the sub-queries as a numbered list.
"""
        response = model.generate_content(prompt)

        # Parse the response to extract sub-queries using regex for more robust parsing
        sub_queries_text = response.text
        # Look for lines starting with a number followed by a period and space (e.g., "1. ")
        sub_queries = re.findall(r'^\d+\.\s*(.*)', sub_queries_text, re.MULTILINE)
        sub_queries = [q.strip() for q in sub_queries if q.strip()] # Ensure no empty strings

        return sub_queries
    except Exception as e:
        print(f"Error during query decomposition: {e}")
        return [query] 

complex_query = "Compare the revenue growth and key risks of Microsoft and Google in 2023."
sub_queries = decompose_query(complex_query, decomposition_model)
print("Original Query:", complex_query)
print("Decomposed Sub-queries:", sub_queries)