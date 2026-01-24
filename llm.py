import os
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

MODEL_API_KEY = os.getenv("MODEL_API_KEY")
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")

print("USING ENDPOINT:", MODEL_ENDPOINT)


def generate_insights(patterns: dict) -> str:
    if not MODEL_ENDPOINT or not MODEL_API_KEY:
        return "LLM not configured: set MODEL_ENDPOINT and MODEL_API_KEY in .env"

    prompt = f"""
You are an AI habit coach.
Based on the user patterns below:
1) Explain the key issues clearly
2) Identify likely root causes
3) Give 3 short, actionable recommendations
4) Give a 'tomorrow plan' in 3 bullet points

User patterns (JSON):
{patterns}

Tone: supportive, practical, concise. No fluff.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful habit coach."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }

    # This is the only part you may need to change depending on the API format:
    r = requests.post(
        MODEL_ENDPOINT,
        headers={"Authorization": f"Bearer {MODEL_API_KEY}",
                 "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    r.raise_for_status()

    data = r.json()

    # Common response shapes:
    # OpenAI-like: data["choices"][0]["message"]["content"]
    if "choices" in data:
        return data["choices"][0]["message"]["content"]

    # Fallback:
    return str(data)
