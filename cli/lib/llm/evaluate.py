import json
from lib.search_utils import PROJECT_ROOT, load_movies
from lib.hybrid_search import HybridSearch


def load_test_case():
    with open(PROJECT_ROOT / "data" / "golden_dataset.json", "r") as f:
        test_cases = json.load(f)

    return test_cases["test_cases"]


# Evaluation Metrices (Precision, Recall, F1 score)
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
        recall = relevent_cnt / len(relevent_docs)

        retrieved = ", ".join([r["title"] for r in rrf_results])

        f1_score = 2 * (precision * recall) / (precision + recall)

        print(f"Query: {query}")
        print(f"\t- Precision@{limit}: {precision:.4f}")
        print(f"\t- Recall@{limit}: {recall:.4f}")
        print(f"\t- F1 Score: {f1_score}")
        print(f"\t- Retrieved: {retrieved}")
        print(f"\t- Relevent: {relevent_docs}\n")
