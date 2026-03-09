from .search_utils import load_movies, load_stop_words, CACHE_PATH
from nltk.stem import PorterStemmer
import string
import os
import pickle
from collections import defaultdict, Counter

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)  # help: {1, 2, 10, 34}
        self.docmap = {}  # maps document ID to document
        self.term_frequencies = defaultdict(Counter)

        """ term frequency is stored as:
                {
                    1: Counter({'dragon': 3, 'hero': 1}),
                    2: Counter({'batman': 2, 'crime': 1}),
                }
                first 1 == doc_id
        """

        self.index_path = CACHE_PATH / "index.pkl"
        self.docmap_path = CACHE_PATH / "docmap.pkl"
        self.term_frequencies_path = CACHE_PATH / "term_frequencies.pkl"

    def __add_document(self, doc_id, text):
        # converts text into tokens and add correcponding doc_id to token
        # eg. "help" token is mapped to corresponding doc_ids
        # help: {1, 2, 10, 34}
        # it becomes easy to search in which document does help keyword exists on

        tokens = tokenization(text)

        for token in set(tokens):
            
            self.index[token].add(doc_id)
        
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term):
        # returns sorted list of doc_ids that contains that token
        # {1, 2, 10, 34}

        return sorted(list(self.index[term]))

    def get_tf(self, doc_id, term):
        token = tokenization(term)
        
        if len(token) != 1:
            raise ValueError("Can only have one token")
        
        return self.term_frequencies[doc_id][token[0]]

    def build(self):

        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]

            # while searching for movies we search both title and description

            text = f"{movie['title']} {movie['description']}"

            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        # saves the index and docmap cache to an file

        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

def tf_command(doc_id, term):
    idx = InvertedIndex()
    
    idx.load()

    print(idx.get_tf(doc_id, term))

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()


def clean_text(text):
    text = text.lower()

    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenization(text):
    # Convert search data and query intoword based tokens

    text = clean_text(text)
    tokens = [tok for tok in text.split() if tok]

    tokens = remove_stop_words(tokens)

    # After removal of stop words we need to stemm tokens,
    # like convert running, runs, etc to run. so related searching is easy

    stemmed_tokens = []

    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))

    return stemmed_tokens


def remove_stop_words(tokens):
    # remove stop words like a, the, etc. that may not have specific meaning to any search

    stop_words = load_stop_words()

    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def has_matching_token(query_tokens, movie_tokens):

    for q_tok in query_tokens:
        for m_tok in movie_tokens:
            if q_tok in m_tok:
                return True
    return False


def search_command(query, n_results=5):
    idx = InvertedIndex()

    idx.load()

    seen, result = set(), []

    query_tokens = tokenization(query)

    for token in query_tokens:
        matching_doc_ids = idx.get_documents(token)

        for matching_doc_id in matching_doc_ids:
            if matching_doc_id in seen:
                continue

            seen.add(matching_doc_id)
            matching_doc = idx.docmap[matching_doc_id]
            result.append(matching_doc)

            if len(result) >= n_results:
                return result

    return result
