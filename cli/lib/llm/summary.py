from lib.search_utils import load_movies, PROMPT_PATH
from lib.hybrid_search import HybridSearch
from .llm import client


def summary(query, limit):
    movies = load_movies()

    hs = HybridSearch(movies)

    docs = hs.rrf_search(query, limit)

    with open(PROMPT_PATH/"summary.md", "r") as f:
        prompt = f.read()

    prompt = prompt.format(query=query, results=docs)

    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

    print("Search results:")
    for doc in docs:
        print(f"\t- {doc["title"]}")

    print(f"LLM Summary:\n{response.text}")
