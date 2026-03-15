import os
import time
from dotenv import load_dotenv
from google import genai
from lib.search_utils import PROMPT_PATH

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")


client = genai.Client(api_key=api_key)


def individual_rerank(query, documents, limit):
    with open(PROMPT_PATH / "re_rank.md", "r") as f:
        prompt = f.read()

    results = []
    for doc in documents:
        prompt = prompt.format(
            query=query, title=doc["title"], description=doc["description"]
        )

        response = client.models.generate_content(
            model="gemma-3-27b-it", contents=prompt
        )

        results.append({**doc, "rerank_response": int(response.text)})
        time.sleep(3)

    results = sorted(results, key=lambda x: x["rerank_response"], reverse=True)
    limit = int(limit / 5)
    return results[:limit]
