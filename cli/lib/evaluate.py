import json
from .search_utils import PROJECT_ROOT, load_movies
from .hybrid_search import HybridSearch


def load_test_case():
    with open(PROJECT_ROOT / "data" / "golden_dataset.json", "r") as f:
        test_cases = json.load(f)

    return test_cases["test_cases"]


def evaluate(limit):
    test_cases = load_test_case()

    movies = load_movies()

    hs = HybridSearch(movies)

    for test_case in test_cases:
        query = test_case["query"]
        relevent_docs = test_case["relevant_docs"]

        rrf_results = hs.rrf_search(query, k=60, limit=limit)

        relevent_cnt = 0

        for rrf_result in rrf_results:
            relevent_cnt += rrf_result["title"] in relevent_docs

        precision = relevent_cnt / limit
        retrieved = ", ".join([r["title"] for r in rrf_results])

        print(f"Query: {query}")
        print(f"\t- Precision@{limit}: {precision}")
        print(f"\t- Retrieved: {retrieved}")
        print(f"\t- Relevent: {relevent_docs}\n")

