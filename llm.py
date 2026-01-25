import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("MODEL_API_KEY")
RAW_BASE = (os.getenv("MODEL_ENDPOINT") or "").rstrip(
    "/")  # env may include junk paths
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
You are an AI habit coach specializing in habit interdependence and behavioral science.

Based on the user patterns below, provide SPECIFIC, ACTIONABLE recommendations:

**USER PROFILE SUMMARY:**
- Average Sleep: {patterns.get('avg_sleep_hours', 0)} hours (Goal: 7.5)
- Average Steps: {patterns.get('avg_steps', 0)} (Goal: 10,000)
- Average Study: {patterns.get('avg_study_hours', 0)} hours (Goal: 3.0)
- Average Mood: {patterns.get('avg_mood_score', 0)}/10 (Goal: 7.0)
- Risk Level: {patterns.get('tomorrow_risk_band', 'Unknown')}

**ANALYSIS:**
1) **Sleep Analysis**: {patterns.get('sleep_trend', '')} trend, {patterns.get('low_sleep_rate_<6h', 0)*100:.0f}% days < 6h sleep
2) **Activity Impact**: Sleep-Mood correlation: {patterns.get('sleep_mood_correlation', 0):.2f}, Activity-Sleep correlation: {patterns.get('active_good_sleep_days', 0)} good days
3) **Consistency Issues**: Sleep consistency: {patterns.get('sleep_consistency_index', 0):.2f}, Study consistency: {patterns.get('study_consistency_index', 0):.2f}
4) **Primary Risk Factors**: {', '.join(patterns.get('primary_risk_factors', [])) if patterns.get('primary_risk_factors') else 'None identified'}

**GENERATE A STRUCTURED RESPONSE WITH THESE SECTIONS:**

1. **HABIT INTERDEPENDENCIES** (2-3 key relationships from their data):
   - How sleep affects their mood/productivity specifically
   - How activity impacts their sleep quality
   - How study habits relate to mood fluctuations

2. **TOP 3 ACTIONABLE RECOMMENDATIONS** (specific to their patterns):
   - Include exact numbers (e.g., "Aim for 7.5 hours on weekdays")
   - Include timing suggestions based on best_wake_hour: {patterns.get('best_wake_hour_for_study', 'N/A')}
   - Address their specific risk factors

3. **WEEKLY IMPROVEMENT PLAN** (focus on consistency):
   - One key habit to build this week
   - One habit to reduce/eliminate
   - Weekly check-in strategy

4. **TOMORROW'S PRIORITY** (single most impactful action):
   - Based on risk score: {patterns.get('tomorrow_risk_score_0_100', 0)}/100
   - What to do first thing in the morning
   - One evening ritual to prepare

5. **MEASURABLE GOALS FOR NEXT 7 DAYS:**
   - Sleep target (hours/night)
   - Activity target (minimum steps/day)
   - Study consistency target (hours/day)
   - Mood improvement target

**Tone**: Supportive but data-driven. Use "we" language. Reference their specific numbers.
**Format**: Use clear headings, bullet points, and specific numbers.
"""
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a supportive but data-driven habit coach. Provide specific, actionable advice based on the user's actual data patterns."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )

        return (resp.choices[0].message.content or "").strip()

    except Exception as e:
        return f"AI insights (fallback mode — LLM error): {e}"
