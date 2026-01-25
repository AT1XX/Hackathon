import pandas as pd
import numpy as np
from datetime import datetime


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Parse date + useful features
    df["Date"] = pd.to_datetime(df["Date"])
    df["DayOfWeek"] = df["Date"].dt.day_name()
    df["Weekend"] = df["DayOfWeek"].isin(["Saturday", "Sunday"])

    # Wake up time -> hour
    df["WakeHour"] = pd.to_datetime(
        df["Wake_Up_Time"], format="%H:%M", errors="coerce").dt.hour

    # Add time-based features
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week

    # Energy score calculation
    df["Energy_Score"] = (
        (df["Sleep_Hours"] / 10 * 40) +  # 40% weight
        (df["Steps"] / 20000 * 30) +      # 30% weight
        (df["Water_Intake_ml"] / 5000 * 20) +  # 20% weight
        (df["Calories_Burned"] / 4000 * 10)    # 10% weight
    )

    # Productivity score
    df["Productivity_Score"] = (
        (df["Study_Hours"] / 8 * 60) +   # 60% weight
        (df["Mood_Score"] / 10 * 40)     # 40% weight
    )

    # Basic "success" proxy
    df["Success"] = (
        (df["Sleep_Hours"] >= 7).astype(int) +
        (df["Steps"] >= 8000).astype(int) +
        (df["Study_Hours"] >= 2).astype(int)
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
    avg_water = float(u["Water_Intake_ml"].mean())
    avg_calories = float(u["Calories_Burned"].mean())

    # Energy and productivity scores
    avg_energy = float(u["Energy_Score"].mean())
    avg_productivity = float(u["Productivity_Score"].mean())

    # Best focus window (by WakeHour vs Study_Hours)
    by_wake = u.groupby("WakeHour")[
        "Study_Hours"].mean().sort_values(ascending=False)
    best_wake_hour = int(by_wake.index[0]) if len(by_wake) else None

    # Weekend vs weekday patterns
    weekday_avg_sleep = float(u[~u["Weekend"]]["Sleep_Hours"].mean())
    weekend_avg_sleep = float(u[u["Weekend"]]["Sleep_Hours"].mean())

    weekday_avg_study = float(u[~u["Weekend"]]["Study_Hours"].mean())
    weekend_avg_study = float(u[u["Weekend"]]["Study_Hours"].mean())

    # "Risk factors" signals
    low_sleep_days = float((u["Sleep_Hours"] < 6).mean())
    very_low_sleep_days = float((u["Sleep_Hours"] < 5).mean())
    low_mood_days = float((u["Mood_Score"] <= 3).mean())
    low_water_days = float((u["Water_Intake_ml"] < 2000).mean())

    # Pattern detection: Sleep -> Mood relationship
    high_sleep_high_mood = u[(u["Sleep_Hours"] >= 7)
                             & (u["Mood_Score"] >= 7)].shape[0]
    low_sleep_low_mood = u[(u["Sleep_Hours"] < 6) &
                           (u["Mood_Score"] <= 4)].shape[0]

    # Activity -> Sleep relationship
    active_good_sleep = u[(u["Steps"] >= 10000) &
                          (u["Sleep_Hours"] >= 7)].shape[0]
    inactive_poor_sleep = u[(u["Steps"] < 5000) &
                            (u["Sleep_Hours"] < 6)].shape[0]

    # Study -> Mood relationship
    studied_good_mood = u[(u["Study_Hours"] >= 3) &
                          (u["Mood_Score"] >= 7)].shape[0]
    no_study_low_mood = u[(u["Study_Hours"] < 1) &
                          (u["Mood_Score"] <= 4)].shape[0]

    # Correlations
    corr_sleep_mood = float(u["Sleep_Hours"].corr(
        u["Mood_Score"])) if u["Sleep_Hours"].nunique() > 1 else 0.0
    corr_steps_mood = float(u["Steps"].corr(
        u["Mood_Score"])) if u["Steps"].nunique() > 1 else 0.0
    corr_study_mood = float(u["Study_Hours"].corr(
        u["Mood_Score"])) if u["Study_Hours"].nunique() > 1 else 0.0
    corr_sleep_study = float(u["Sleep_Hours"].corr(
        u["Study_Hours"])) if u["Sleep_Hours"].nunique() > 1 else 0.0

    # Trend analysis (last 7 vs last 30 days)
    recent_7 = u.sort_values("Date").tail(7)
    recent_30 = u.sort_values("Date").tail(30)

    sleep_trend = "improving" if recent_7["Sleep_Hours"].mean(
    ) > recent_30["Sleep_Hours"].mean() else "declining"
    mood_trend = "improving" if recent_7["Mood_Score"].mean(
    ) > recent_30["Mood_Score"].mean() else "declining"
    study_trend = "improving" if recent_7["Study_Hours"].mean(
    ) > recent_30["Study_Hours"].mean() else "declining"

    # Risk score calculation
    risk_score = 0
    risk_factors = []

    if recent_7["Sleep_Hours"].mean() < 6.5:
        risk_score += 30
        risk_factors.append("low_sleep")
    if recent_7["Steps"].mean() < 6000:
        risk_score += 20
        risk_factors.append("low_activity")
    if recent_7["Mood_Score"].mean() < 5:
        risk_score += 25
        risk_factors.append("low_mood")
    if recent_7["Study_Hours"].mean() < 1.5:
        risk_score += 15
        risk_factors.append("low_study")
    if recent_7["Water_Intake_ml"].mean() < 2000:
        risk_score += 10
        risk_factors.append("low_water")

    risk_band = "Low" if risk_score < 25 else "Medium" if risk_score < 60 else "High"

    # Most productive days of week
    by_day = u.groupby("DayOfWeek")["Productivity_Score"].mean()
    best_day = by_day.idxmax() if len(by_day) else None
    worst_day = by_day.idxmin() if len(by_day) else None

    # Consistency metrics
    sleep_consistency = float(
        1 - (u["Sleep_Hours"].std() / u["Sleep_Hours"].mean())) if u["Sleep_Hours"].mean() > 0 else 0
    study_consistency = float(
        1 - (u["Study_Hours"].std() / u["Study_Hours"].mean())) if u["Study_Hours"].mean() > 0 else 0

    return {
        "user_id": user_id,
        "rows": int(len(u)),

        # Core averages
        "avg_sleep_hours": round(avg_sleep, 2),
        "avg_steps": round(avg_steps, 0),
        "avg_study_hours": round(avg_study, 2),
        "avg_mood_score": round(avg_mood, 2),
        "avg_water_intake_ml": round(avg_water, 0),
        "avg_calories_burned": round(avg_calories, 0),

        # Derived scores
        "avg_energy_score": round(avg_energy, 2),
        "avg_productivity_score": round(avg_productivity, 2),

        # Time patterns
        "best_wake_hour_for_study": best_wake_hour,
        "weekday_avg_sleep": round(weekday_avg_sleep, 2),
        "weekend_avg_sleep": round(weekend_avg_sleep, 2),
        "weekday_avg_study": round(weekday_avg_study, 2),
        "weekend_avg_study": round(weekend_avg_study, 2),

        # Risk factors
        "low_sleep_rate_<6h": round(low_sleep_days, 2),
        "very_low_sleep_rate_<5h": round(very_low_sleep_days, 2),
        "low_mood_rate_<=3": round(low_mood_days, 2),
        "low_water_rate_<2000ml": round(low_water_days, 2),

        # Habit relationships (counts)
        "high_sleep_high_mood_days": high_sleep_high_mood,
        "low_sleep_low_mood_days": low_sleep_low_mood,
        "active_good_sleep_days": active_good_sleep,
        "inactive_poor_sleep_days": inactive_poor_sleep,
        "studied_good_mood_days": studied_good_mood,
        "no_study_low_mood_days": no_study_low_mood,

        # Correlations
        "sleep_mood_correlation": round(corr_sleep_mood, 2),
        "steps_mood_correlation": round(corr_steps_mood, 2),
        "study_mood_correlation": round(corr_study_mood, 2),
        "sleep_study_correlation": round(corr_sleep_study, 2),

        # Trends
        "sleep_trend": sleep_trend,
        "mood_trend": mood_trend,
        "study_trend": study_trend,
        "recent_7_days_sleep": round(recent_7["Sleep_Hours"].mean(), 2),
        "recent_30_days_sleep": round(recent_30["Sleep_Hours"].mean(), 2),

        # Risk assessment
        "tomorrow_risk_score_0_100": int(min(100, risk_score)),
        "tomorrow_risk_band": risk_band,
        "primary_risk_factors": risk_factors,

        # Day patterns
        "most_productive_day": best_day,
        "least_productive_day": worst_day,

        # Consistency
        "sleep_consistency_index": round(sleep_consistency, 2),
        "study_consistency_index": round(study_consistency, 2),

        # Recommendations data
        "sleep_goal": 7.5,
        "steps_goal": 10000,
        "study_goal": 3.0,
        "water_goal": 3000,
        "mood_goal": 7.0,
    }
