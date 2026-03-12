from collections import defaultdict
from sentence_transformers import SentenceTransformer
from .search_utils import CACHE_PATH, load_movies
import numpy as np
import os
import re
import json

SCORE_PRECISION = 4


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.embedding_path = CACHE_PATH / "movie_embedding.npy"

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarity_scores = []

        for embedding, doc in zip(self.embeddings, self.documents):
            # zip() combines two or more iterables (lists, arrays, etc.) element
            # by element into pairs (or tuples).
            # numbers = [1, 2, 3]        ---> 1 a
            # letters = ['a', 'b', 'c']  ---> 2 b

            _similarity = cosine_similarity(query_embedding, embedding)
            similarity_scores.append((_similarity, doc))

        similarity_scores.sort(key=lambda x: x[0], reverse=True)

        scores = similarity_scores[:limit]

        format_results = []
        for score, document in scores:
            format_results.append(
                {
                    "doc_id": document["id"],
                    "title": document["title"],
                    "score": score,
                    "description": document["description"],
                }
            )

        return format_results

    def generate_embedding(self, text):
        if not text:
            raise ValueError("Text cannot be empty")

        res = self.model.encode([text])

        return res[0]

    def build_embedding(self, documents):
        self.documents = documents
        document_strings = []

        for document in documents:
            self.document_map[document["id"]] = document

            document_strings.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(document_strings, show_progress_bar=True)
        np.save(self.embedding_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeding(self, documents):
        self.documents = documents

        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists(self.embedding_path):
            self.embeddings = np.load(self.embedding_path)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embedding(documents)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self.chunk_embeddings_path = CACHE_PATH / "chunk_embeddings.npy"
        self.chunk_metadata_path = CACHE_PATH / "chunk_metadata.json"

    def search_chunks(self, query: str, limit: int = 10):

        if self.chunk_embeddings is None:
            raise ValueError(
                "Chunk embeddings not loaded. Call load_or_create_chunk_embeddings first."
            )
        
        query_embedding = self.generate_embedding(query)

        chunk_score = []

        movie_score = defaultdict(lambda: 0) # default score is 0

        for i, embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(embedding, query_embedding)

            metadata = self.chunk_metadata["chunks"][i]
            
            mid, cid = metadata["movie_id"], metadata["chunk_id"]

            chunk_score.append({
                    "chunk_id": cid,
                    "movie_id": mid,
                    "score": similarity,
                })
            movie_score[mid] = max(similarity, movie_score[mid]) # here if no value is added
                                                                 # default value is 0, hence
                                                                 # max value i.e new value 
                                                                 # will be added
        
        #for chunk_sc in chunk_score:
        #    if chunk_sc["movie_id"] not in movie_score:
        #        movie_score[chunk_sc["movie_id"]] = chunk_sc["score"]

        #    if chunk_sc["score"] > movie_score[chunk_sc["movie_id"]]:
        #        movie_score[chunk_sc["movie_id"]] = chunk_sc["score"]

        movie_score_sorted = sorted(movie_score.items(), key= lambda x: x[1], reverse=True)

        top_results = movie_score_sorted[:limit]
        
        final_result = []
        for movie_id, score in top_results:
            doc = self.documents[movie_id]

            final_result.append({
                    "id": movie_id,
                    "title": doc["title"],
                    "description": doc["description"][:100],
                    "score": round(score, SCORE_PRECISION),
                    "metadata": doc.get("metadata", {})
                })

        return final_result

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        all_chunks = []
        chunk_metadata = []

        for movie_id, document in enumerate(documents):
            self.document_map[document["id"]] = document

            description = document["description"]

            if description.strip() == "":
                continue

            chunks = semantic_chunk(description, max_chunk_size=4, overlap=1)

            all_chunks += chunks

            for chunk_id in range(len(chunks)):
                chunk_metadata.append(
                    {
                        "movie_id": movie_id,
                        "chunk_id": chunk_id,
                        "total_chunks": len(chunks),
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:

        self.documents = documents

        for document in documents:
            self.document_map[document["id"]] = document

        if os.path.exists(self.chunk_metadata_path) and os.path.exists(
            self.chunk_embeddings_path
        ):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)

            with open(self.chunk_metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


def search_chunked_command(query, limit):
    movies = load_movies()

    css = ChunkedSemanticSearch()

    css.load_or_create_chunk_embeddings(movies)

    result = css.search_chunks(query, limit)
    
    for i, res in enumerate(result):
        print(f"\n{i}. {res["title"]} (score: {res["score"]:.4f})")
        print(f"   {res["description"]}...")

def embed_chunks_command():
    movies = load_movies()

    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)

    print(f"Generated {len(embeddings)} chunked embeddings")


def semantic_chunk(text, max_chunk_size, overlap):
    text = text.strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not re.search(r"[.!?]$", sentences[0]):
        sentences = [text]

    chunks = []
    step_size = max_chunk_size - overlap

    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i : i + max_chunk_size]
        
        cleaned_sentences = []

        for s in chunk_sentences:
            s = s.strip()

            if s:
                cleaned_sentences.append(s)

        if not cleaned_sentences:
            continue 

        chunks.append(" ".join(cleaned_sentences))

        if i + max_chunk_size >= len(sentences):
            break

    return chunks


def semantic_chunk_command(text, max_chunk_size, overlap):
    chunks = semantic_chunk(text, max_chunk_size, overlap)

    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i}. {chunk}\n")


def fixed_size_chunking(text, chunk_size, overlap):
    text = text.split()

    chunks = []

    if overlap >= chunk_size:
        raise ValueError("Overlap size cannot be greater than chunk size")

    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size]

        if not chunk:
            break

        if len(chunk) <= overlap:
            break

        chunks.append(" ".join(chunk))

        i += chunk_size - overlap

    return chunks


def chunk_command(text, chunk_size, overlap=0):
    chunks = fixed_size_chunking(text, chunk_size, overlap)

    print(f"chunking {len(text)} characters\n")
    for i, chunk in enumerate(chunks):
        print(f"{i}. {chunk}\n")


def search_command(query, limit):
    ss = SemanticSearch()

    movies = load_movies()

    ss.load_or_create_embeding(movies)

    res = ss.search(query, limit)

    for i, r in enumerate(res):
        print(
            f"{i}. {r['title']} (score: {r['score']:.4f})\n{r['description'][:100]}\n\n"
        )


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)  # magnitude of vec1
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_query_text(query):
    ss = SemanticSearch()

    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    ss = SemanticSearch()

    documents = load_movies()

    embeddings = ss.load_or_create_embeding(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")
