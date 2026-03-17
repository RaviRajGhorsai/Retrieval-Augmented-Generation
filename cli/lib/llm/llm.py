import os
import json
from dotenv import load_dotenv
from google import genai
from lib.search_utils import PROMPT_PATH

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")


client = genai.Client(api_key=api_key)


def augment_prompt(query, enhance):
    with open(PROMPT_PATH / f"{enhance}.md", "r") as f:
        prompt = f.read()

    return generate_text(prompt, query)


def generate_text(prompt, query):
    prompt = prompt.format(query=query)
    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)
    return response.text


def error_analysis(results, query):
    with open(PROMPT_PATH / "evaluation.md", "r") as f:
        prompt = f.read()

    formatted_results = []

    for result in results:
        formatted_results.append(result["title"])

    formatted_results_str = "\n".join(formatted_results)

    prompt = prompt.format(query=query, formatted_results=formatted_results_str)

    response = client.models.generate_content(model="gemma-3-27b-it", contents=prompt)

    judge = json.loads(response.text)

    for idx, r in enumerate(results):
        print(f"{idx + 1}. {r['title']}: {judge[idx]}/3")
