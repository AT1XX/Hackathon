import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("MODEL_API_KEY")
BASE = (os.getenv("MODEL_ENDPOINT") or "").rstrip("/")

# BASE should be host only (no /v1)
client = OpenAI(base_url=f"{BASE}/v1", api_key=API_KEY)

models = client.models.list()
for m in models.data:
    print(m.id)
