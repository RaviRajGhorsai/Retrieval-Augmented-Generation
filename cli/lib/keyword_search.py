from .search_utils import load_movies, load_stop_words
import string


def clean_text(text):
    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenization(text):
    text = clean_text(text)
    tokens = [tok for tok in text.split() if tok]
    return tokens

def has_matching_token(query_tokens, movie_tokens):
    query_tokens = remove_stop_words(query_tokens)
    movie_tokens = remove_stop_words(movie_tokens)

    for q_tok in query_tokens:
        for m_tok in movie_tokens:
            if q_tok in m_tok:
                return True
    return False

def remove_stop_words(tokens):
    stop_words = load_stop_words()

    tokens = [token for token in tokens if token not in stop_words]
    return tokens

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
