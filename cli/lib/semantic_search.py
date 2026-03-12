from sentence_transformers import SentenceTransformer
from .search_utils import CACHE_PATH, load_movies
import numpy as np
import os
import re
import json
import numpy as np


class SemanticSearch:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
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
            raise ValueError("Text cann be empty")

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
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self.chunk_embeddings_path = CACHE_PATH / "chunk_embeddings.npy"
        self.chunk_metadata_path = CACHE_PATH / "chunk_metadata.json"

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


def embed_chunks_command():
    movies = load_movies()

    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)

    print(f"Generated {len(embeddings)} chunked embeddings")


def semantic_chunk(text, max_chunk_size, overlap):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    chunks = []
    step_size = max_chunk_size - overlap

    for i in range(0, len(sentences), step_size):
        chunk = sentences[i : i + max_chunk_size]

        if not chunk:
            break

        chunks.append(" ".join(chunk))
        
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
