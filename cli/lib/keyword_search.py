from .search_utils import load_movies, load_stop_words, CACHE_PATH
from nltk.stem import PorterStemmer
import string
import os
import pickle
from collections import defaultdict

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap = {}  # maps document ID to document
        self.index_path = CACHE_PATH / "index.pkl"
        self.docmap_path = CACHE_PATH / "docmap.pkl"

    def __add_document(self, doc_id, text):
        tokens = tokenization(text)

        for token in set(tokens):
            self.index[token].add(doc_id)

    def get_documents(self, term):
        return sorted(list(self.index[term]))

    def build(self):

        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"

            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")

    print(f"First document for token 'merida' = {docs[0]}")

def clean_text(text):
    text = text.lower()

    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


# Convert search data and query intoword based tokens
def tokenization(text):
    text = clean_text(text)
    tokens = [tok for tok in text.split() if tok]

    tokens = remove_stop_words(tokens)

    # After removal of stop words we need to stemm tokens,
    # like convert running, runs, etc to run. so related searching is easy

    stemmed_tokens = []

    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))

    return stemmed_tokens


# remove stop words like a, the, etc. that may not have specific meaning to any search
def remove_stop_words(tokens):
    stop_words = load_stop_words()

    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def has_matching_token(query_tokens, movie_tokens):

    for q_tok in query_tokens:
        for m_tok in movie_tokens:
            if q_tok in m_tok:
                return True
    return False


def search_command(query, n_results):
    data = load_movies()
    result_movie = []

    query_tokens = tokenization(query)

    for movie in data:
        movie_tokens = tokenization(movie["title"])

        if has_matching_token(query_tokens, movie_tokens):
            result_movie.append(movie)
        if len(result_movie) == n_results:
            break
    return result_movie
