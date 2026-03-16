import os

from lib.prompt.re_rank import individual_rerank, batch_rerank, cross_encoder

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from lib.search_utils import load_movies
from lib.llm import augment_prompt, error_analysis


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_search_results = self._bm25_search(query, limit * 500)
        semantic_search_results = self.semantic_search.search_chunks(query, limit * 500)

        combined_results = combine_search_results(
            bm25_search_results, semantic_search_results, alpha
        )

        return combined_results[:limit]

    def rrf_search(self, query, k, limit=10):
        bm25_search_results = self._bm25_search(query, limit * 500)
        semantic_search_results = self.semantic_search.search_chunks(query, limit * 500)

        results = combine_result_rrf(bm25_search_results, semantic_search_results, k)

       # print("Results after RRF search\n")
       # for i, res in enumerate(results[:limit]):
       #     print(f"{i}. {res['title']}")
       #     print(f"RRF Score: {res['rrf_score']:.3f}")
       #     print(
       #         f"BM25 rank: {res['bm25_rank']} Semantic Rank: {res['semantic_rank']}"
       #     )
       #     print(f"{res['description'][:100]}\n")

        return results[:limit]


def rrf_search_command(query, limit, k, evaluate, enhance=None, re_rank_method=None):
    movies = load_movies()
    hs = HybridSearch(movies)

    print(f"Query: {query}")

    if enhance:
        new_query = augment_prompt(query, enhance)
        print(f"Enhanced query ({enhance}): '{query}' -> '{new_query}'\n")

        query = new_query

    rrf_limit = limit * 5 if re_rank_method else limit

    results = hs.rrf_search(query, k, rrf_limit)

    if re_rank_method:
        match re_rank_method:
            case "individual":
                results = individual_rerank(query, results, rrf_limit)
                print(f"Re-ranking top {limit} results using individual method...")

            case "batch":
                results = batch_rerank(query, results, limit)
                print(f"Re-ranking top {limit} results using batch method...")

            case "cross_encoder":
                results = cross_encoder(query, results, limit)
                print(f"Re-ranking top {limit} results using cross_encoder method...")

        print(f"Reciprocal Rank Fusion Results for {query} (k={k}):\n")

    for i, res in enumerate(results):
        print(f"{i}. {res['title']}")
        if re_rank_method == "cross_encoder":
            print(f"Cross Encoder Score: {res['cross_encoder_score']}")
        print(f"RRF Score: {res['rrf_score']:.3f}")
        print(f"BM25 rank: {res['bm25_rank']} Semantic Rank: {res['semantic_rank']}")
        print(f"{res['description'][:100]}\n")

    if evaluate:
        error_analysis(results, query)


def weighted_search_command(query, alpha, limit):
    movies = load_movies()
    hs = HybridSearch(movies)

    results = hs.weighted_search(query, alpha, limit)

    for i, res in enumerate(results):
        print(
            f"{i}. id: {res['doc_id']}\n{res['title']}\nHYbrid Score: {res['hybrid_score']:.3f}\nBM25: {res['keyword_score']:.3f}, Semantic: {res['semantic_score']:.3f}\n{res['description']}"
        )


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_search_results(results):
    scores = [r["score"] for r in results]
    norm_scores = normalization(scores)

    for idx, result in enumerate(results):
        result["normalized_score"] = norm_scores[idx]

    return results


def combine_result_rrf(bm25_results, semantic_results, k):

    bm25_rank = rank(bm25_results)
    semantic_rank = rank(semantic_results)

    docs_maps = {}

    for doc in bm25_rank:
        doc_id = doc["doc_id"]

        docs_maps[doc_id] = {
            "doc_id": doc_id,
            "bm25_rank": doc["rank"],
            "semantic_rank": None,
            "title": doc["title"],
            "description": doc["description"],
        }
    for doc in semantic_rank:
        doc_id = doc["id"]

        if doc_id not in docs_maps:
            docs_maps[doc_id] = {
                "doc_id": doc_id,
                "bm25_rank": None,
                "semantic_rank": doc["rank"],
                "title": doc["title"],
                "description": doc["description"],
            }
        else:
            docs_maps[doc_id]["semantic_rank"] = doc["rank"]

    for v in docs_maps.values():
        v["rrf_score"] = rrf_score(v["bm25_rank"], k) + rrf_score(v["semantic_rank"], k)

    results = sorted(docs_maps.values(), key=lambda x: x["rrf_score"], reverse=True)

    return results


def rank(scores):
    ranked = sorted(scores, key=lambda x: x["score"], reverse=True)

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    return ranked


def rrf_score(rank, k=60):
    if not isinstance(rank, int):
        return 0
    return 1 / (k + rank)


def combine_search_results(bm25_results, semantic_results, alpha):
    bm25_norm = normalize_search_results(bm25_results)
    semantic_norm = normalize_search_results(semantic_results)

    docs_maps = {}

    for norm in bm25_norm:
        doc_id = norm["doc_id"]

        docs_maps[doc_id] = {
            "doc_id": doc_id,
            "keyword_score": norm["normalized_score"],
            "semantic_score": 0.0,
            "title": norm["title"],
            "description": None,
        }
    for norm in semantic_norm:
        doc_id = norm["id"]

        if doc_id not in docs_maps:
            docs_maps[doc_id] = {
                "doc_id": doc_id,
                "keyword_score": 0.0,
                "semantic_score": norm["normalized_score"],
                "title": norm["title"],
                "description": norm["description"][:100],
            }

        docs_maps[doc_id]["semantic_score"] = norm["normalized_score"]

    for k, v in docs_maps.items():
        docs_maps[k]["hybrid_score"] = hybrid_score(
            v["keyword_score"], v["semantic_score"], alpha
        )

    results = sorted(docs_maps.values(), key=lambda x: x["hybrid_score"], reverse=True)

    return results


def normalization(scores):
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]


def normalize_command(scores):
    result = normalization(scores)

    for res in result:
        print(f"{res:.4f}")
