from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch
from .llm import client

def rag(query):
    movies = load_movies()

    hs = HybridSearch(movies)

    docs = hs.rrf_search(query, k=60, limit=6)
    

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored                
                to Hoopla users. Hoopla is a movie streaming service.

                Query: {query}

                Documents:
                {docs}

                Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

    print("Search Results:")

    for doc in docs:
        print(f"\t- {doc["title"]}")

    print(f"\nRAG Response:\n{response.text}")
