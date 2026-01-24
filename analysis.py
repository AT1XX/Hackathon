import pandas as pd
import numpy as np


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Parse date + useful features
    df["Date"] = pd.to_datetime(df["Date"])
    df["DayOfWeek"] = df["Date"].dt.day_name()

    # Wake up time -> hour
    # Wake_Up_Time is like "07:02"
    df["WakeHour"] = pd.to_datetime(
        df["Wake_Up_Time"], format="%H:%M", errors="coerce").dt.hour

    # Basic “success” proxy (you can tweak for your story)
    # Example: success day if Sleep >= 7, Steps >= 8000, Study >= 2
    df["Success"] = (
        (df["Sleep_Hours"] >= 7).astype(int)
        + (df["Steps"] >= 8000).astype(int)
        + (df["Study_Hours"] >= 2).astype(int)
    )
    df["SuccessLabel"] = np.where(df["Success"] >= 2, "Good Day", "Risk Day")
    return df


def compute_patterns(df: pd.DataFrame, user_id: str) -> dict:
    u = df[df["User_ID"] == user_id].copy()
    if u.empty:
        return {"error": f"No rows for user {user_id}"}

    # Core stats
    avg_sleep = float(u["Sleep_Hours"].mean())
    avg_steps = float(u["Steps"].mean())
    avg_study = float(u["Study_Hours"].mean())
    avg_mood = float(u["Mood_Score"].mean())

    # Best focus window (by WakeHour vs Study_Hours)
    by_wake = u.groupby("WakeHour")[
        "Study_Hours"].mean().sort_values(ascending=False)
    best_wake_hour = int(by_wake.index[0]) if len(by_wake) else None

    # “Risk factors” signals
    low_sleep_days = float((u["Sleep_Hours"] < 6).mean())
    low_mood_days = float((u["Mood_Score"] <= 3).mean())

    # Relationship: sleep vs mood
    corr_sleep_mood = float(u["Sleep_Hours"].corr(
        u["Mood_Score"])) if u["Sleep_Hours"].nunique() > 1 else 0.0

    # Simple risk score for “tomorrow”
    # (Hackathon-friendly heuristic)
    recent = u.sort_values("Date").tail(7)
    risk_score = 0
    risk_score += 35 if recent["Sleep_Hours"].mean() < 6.5 else 0
    risk_score += 25 if recent["Steps"].mean() < 6000 else 0
    risk_score += 20 if recent["Mood_Score"].mean() < 5 else 0
    risk_score += 20 if recent["Study_Hours"].mean() < 1.5 else 0

    risk_band = "Low" if risk_score < 25 else "Medium" if risk_score < 60 else "High"

    return {
        "user_id": user_id,
        "rows": int(len(u)),
        "avg_sleep_hours": round(avg_sleep, 2),
        "avg_steps": round(avg_steps, 0),
        "avg_study_hours": round(avg_study, 2),
        "avg_mood_score": round(avg_mood, 2),
        "best_wake_hour_for_study": best_wake_hour,
        "low_sleep_rate_<6h": round(low_sleep_days, 2),
        "low_mood_rate_<=3": round(low_mood_days, 2),
        "sleep_mood_correlation": round(corr_sleep_mood, 2),
        "tomorrow_risk_score_0_100": int(min(100, risk_score)),
        "tomorrow_risk_band": risk_band,
    }
