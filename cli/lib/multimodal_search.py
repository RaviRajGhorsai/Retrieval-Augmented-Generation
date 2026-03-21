from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
from lib.search_utils import load_movies


class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []

        for doc in documents:
            self.texts.append(f"{doc["title"]}: {doc["description"]}")

        
        self.text_embedding = self.model.encode(self.texts, show_progress_bar=True) 

    def embed_image(self, image_path):
        image = Image.open(image_path).convert("RGB")

        embedding = self.model.encode([image], show_progress_bar=True)

        return embedding[0]
    
    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)

        text_embedding = self.text_embedding
        
        similarities = []
        for idx, emb in enumerate(text_embedding):
            similarities.append((idx, cosine_similarity(emb, image_embedding)))

        similarity_score = sorted(similarities, key=lambda x:x[1], reverse=True)

        results = []
        for id, score in similarity_score:
            doc = self.documents[id]
            
            results.append({
                    "doc_id": id,
                    "title": doc["title"],
                    "similarity": score,
                    "description": doc["description"]
                })

        return results[:5]


def verify_embedding(image_path):
    movies = load_movies()

    ms = MultimodalSearch(movies)
    
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    movies = load_movies()

    ms = MultimodalSearch(movies)

    result = ms.search_with_image(image_path)

    for idx, res in enumerate(result):
        print(f"{idx}. {res["title"]} (similarity: {res["similarity"]})\n{res["description"][:100]}\n\n")

