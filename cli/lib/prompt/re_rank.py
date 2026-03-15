import os
import json
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
    with open(PROMPT_PATH / "re_rank_individual.md", "r") as f:
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


def batch_rerank(query, documents, limit):

    with open(PROMPT_PATH / "re_rank_batch.md", "r") as f:
        prompt = f.read()

    doc_list_str = ""
    _temp = """<movie id={idx}>{title}:\n{desc}\n</movie>\n"""
    for idx, doc in enumerate(documents):
        doc_list_str += _temp.format(
            idx=idx, title=doc["title"], desc=doc["description"]
        )

    _prompt = prompt.format(query=query, doc_list_str=doc_list_str, len=len(documents))

    response = client.models.generate_content(model="gemma-3-27b-it", contents=_prompt)

    response_parsed = json.loads(response.text)

    results = []

    for idx, doc in enumerate(documents):
        results.append({**doc, "rerank_score": response_parsed.index(idx)})

    results = sorted(results, key=lambda x: x["rerank_score"], reverse=False)

    return results
