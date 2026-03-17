from lib.search_utils import load_movies, PROMPT_PATH
from lib.hybrid_search import HybridSearch
from .llm import client


def summary(query, limit):
    movies = load_movies()

    hs = HybridSearch(movies)

    docs = hs.rrf_search(query, limit)

    with open(PROMPT_PATH / "summary.md", "r") as f:
        prompt = f.read()

    prompt = prompt.format(query=query, documents=docs[:5])

    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

    print("Search results:")
    for doc in docs:
        print(f"\t- {doc['title']}")

    print(f"LLM Summary:\n{response.text}")


def answer_question(question, limit):
    movies = load_movies()

    hs = HybridSearch(movies)

    docs = hs.rrf_search(question, limit)

    with open(PROMPT_PATH / "question.md", "r") as f:
        prompt = f.read()

    prompt = prompt.format(question=question, context=docs[:5])

    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

    print("Search results:")
    for doc in docs:
        print(f"\t- {doc['title']}")

    print(f"Answer:\n{response.text}")
