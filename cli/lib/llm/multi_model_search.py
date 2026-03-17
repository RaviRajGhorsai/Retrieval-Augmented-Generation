from lib.llm.llm import client
from lib.search_utils import PROMPT_PATH
import mimetypes
from google.genai import types


def describe_image(image_path, query):

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as f:
        image = f.read()

    with open(PROMPT_PATH / "system_prompt_for_img.md", "r") as f:
        prompt = f.read()

    parts = [
        prompt,
        types.Part.from_bytes(data=image, mime_type=mime),
        query.strip(),
    ]

    response = client.models.generate_content(model="gemma-3-27b-it", contents=parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
