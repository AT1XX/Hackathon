import os
import traceback
from dotenv import load_dotenv
from openai import OpenAI  # Groq uses OpenAI-compatible SDK

load_dotenv()

API_KEY = os.getenv("MODEL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")

_client = None

def _get_client():
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError("Missing MODEL_API_KEY in .env")
        
        # Groq endpoint
        _client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=API_KEY
        )
        print(f"[LLM] Using Groq with model: {MODEL_NAME}")
    return _client

def generate_insights(patterns: dict) -> str:
    try:
        client = _get_client()

        prompt = f"""
You are an AI habit coach specializing in habit interdependence and behavioral science.

Based on the user patterns below, provide SPECIFIC, ACTIONABLE recommendations:

**USER PROFILE SUMMARY:**
- Last 7 Days Sleep: {patterns.get('avg_sleep_hours_7d', 0)} hours (Goal: 7.5)
- Last 30 Days Sleep: {patterns.get('avg_sleep_hours_30d', 0)} hours
- Last 7 Days Steps: {patterns.get('avg_steps_7d', 0)} (Goal: 10,000)
- Last 30 Days Steps: {patterns.get('avg_steps_30d', 0)}
- Last 7 Days Study: {patterns.get('avg_study_hours_7d', 0)} hours (Goal: 3.0)
- Last 30 Days Study: {patterns.get('avg_study_hours_30d', 0)} hours
- Last 7 Days Mood: {patterns.get('avg_mood_score_7d', 0)}/10 (Goal: 7.0)
- Last 30 Days Mood: {patterns.get('avg_mood_score_30d', 0)}/10
- Risk Level: {patterns.get('tomorrow_risk_band', 'Unknown')}

**ANALYSIS:**
1) **Sleep Trends**: Recent trend: {patterns.get('sleep_trend', '')}, 7-day avg: {patterns.get('avg_sleep_hours_7d', 0)}h, 30-day avg: {patterns.get('avg_sleep_hours_30d', 0)}h
2) **Activity Impact**: 7-day steps: {patterns.get('avg_steps_7d', 0)}, 30-day steps: {patterns.get('avg_steps_30d', 0)}
3) **Study Patterns**: 7-day study: {patterns.get('avg_study_hours_7d', 0)}h, 30-day study: {patterns.get('avg_study_hours_30d', 0)}h
4) **Correlations (7-day)**: Sleep-Mood: {patterns.get('sleep_mood_correlation_7d', 0):.2f}, Steps-Mood: {patterns.get('steps_mood_correlation_7d', 0):.2f}
5) **Correlations (30-day)**: Sleep-Mood: {patterns.get('sleep_mood_correlation_30d', 0):.2f}, Steps-Mood: {patterns.get('steps_mood_correlation_30d', 0):.2f}
6) **Risk Factors**: {', '.join(patterns.get('primary_risk_factors', [])) if patterns.get('primary_risk_factors') else 'None identified'}
7) **Consistency**: Sleep consistency: {patterns.get('sleep_consistency_index', 0):.2f}, Study consistency: {patterns.get('study_consistency_index', 0):.2f}

**GENERATE A STRUCTURED RESPONSE WITH THESE SECTIONS:**

1. **HABIT INTERDEPENDENCIES** (2-3 key relationships from their data):
   - Compare 7-day vs 30-day trends
   - How sleep affects their mood/productivity specifically
   - How activity impacts their sleep quality
   - How study habits relate to mood fluctuations

2. **TOP 3 ACTIONABLE RECOMMENDATIONS** (specific to their patterns):
   - Include exact numbers (e.g., "Aim for 7.5 hours on weekdays")
   - Include timing suggestions based on best_wake_hour: {patterns.get('best_wake_hour_for_study', 'N/A')}
   - Address their specific risk factors
   - Compare 7-day vs 30-day performance

3. **WEEKLY IMPROVEMENT PLAN** (focus on consistency):
   - One key habit to build this week
   - One habit to reduce/eliminate
   - Weekly check-in strategy

4. **TOMORROW'S PRIORITY** (single most impactful action):
   - Based on risk score: {patterns.get('tomorrow_risk_score_0_100', 0)}/100
   - What to do first thing in the morning
   - One evening ritual to prepare

5. **MEASURABLE GOALS FOR NEXT 7 DAYS:**
   - Sleep target (hours/night) - consider 7-day average: {patterns.get('avg_sleep_hours_7d', 0)}h
   - Activity target (minimum steps/day) - consider 7-day average: {patterns.get('avg_steps_7d', 0)}
   - Study consistency target (hours/day) - consider 7-day average: {patterns.get('avg_study_hours_7d', 0)}h
   - Mood improvement target

**Tone**: Supportive but data-driven. Use "we" language. Reference their specific numbers for both 7-day and 30-day periods.
**Format**: Use clear headings, bullet points, and specific numbers.
"""

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a supportive but data-driven habit coach."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2500,
        )
        
        return (resp.choices[0].message.content or "").strip()
        
    except Exception as e:
        print("=" * 60)
        print("ERROR in generate_insights:")
        traceback.print_exc()
        print("=" * 60)
        return f"""⚠️ **AI insights (fallback mode – LLM error)**

**Error details:**  
`{type(e).__name__}: {e}'
"""