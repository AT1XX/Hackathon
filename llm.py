import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("MODEL_API_KEY")
RAW_BASE = (os.getenv("MODEL_ENDPOINT") or "").rstrip("/")  # env may include junk paths
MODEL = os.getenv("MODEL_NAME", "google/gemma-3-27b-it")

_client = None

def _normalize_base(raw: str) -> str:
    """
    Accepts:
      https://host
      https://host/v1
      https://host/v1/
      https://host/v1/chat
      https://host/v1/chat/
    Returns:
      https://host/v1
    """
    if not raw:
        return ""

    # strip common accidental suffixes
    for suffix in ("/v1/chat", "/v1/chat/", "/chat", "/chat/", "/v1", "/v1/"):
        if raw.endswith(suffix.rstrip("/")):
            raw = raw[: -len(suffix.rstrip("/"))]
            raw = raw.rstrip("/")
            break

    return f"{raw}/v1"

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError("Missing MODEL_API_KEY in .env")
        if not RAW_BASE:
            raise RuntimeError("Missing MODEL_ENDPOINT in .env")

        base_url = _normalize_base(RAW_BASE)

        # Helpful debug print (shows up in your Streamlit logs / terminal)
        print(f"[LLM] Using base_url={base_url} model={MODEL}")

        _client = OpenAI(base_url=base_url, api_key=API_KEY)
    return _client

def generate_insights(patterns: dict) -> str:
    try:
        client = _get_client()

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
""".strip()

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful habit coach."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=350,
        )

        return (resp.choices[0].message.content or "").strip()

    except Exception as e:
        return f"AI insights (fallback mode — LLM error): {e}"
