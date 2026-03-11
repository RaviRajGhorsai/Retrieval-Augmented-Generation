from sentence_transformers import SentenceTransformer
from .search_utils import CACHE_PATH, load_movies
import numpy as np
import os


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.embedding_path = CACHE_PATH / "movie_embedding.npy"

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
