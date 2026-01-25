from datetime import datetime, timedelta
import numpy as np
import pandas as pd


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

    # Sort by date
    u = u.sort_values("Date")

    # Get recent data
    recent_7 = u.tail(7)
    recent_30 = u.tail(30)

    # Function to compute stats for a given dataframe
    def compute_stats_for_period(data, period_name):
        if len(data) == 0:
            return {}

        # Core stats
        avg_sleep = float(data["Sleep_Hours"].mean())
        avg_steps = float(data["Steps"].mean())
        avg_study = float(data["Study_Hours"].mean())
        avg_mood = float(data["Mood_Score"].mean())
        avg_water = float(data["Water_Intake_ml"].mean())
        avg_calories = float(data["Calories_Burned"].mean())

        # Energy and productivity scores
        avg_energy = float(data["Energy_Score"].mean())
        avg_productivity = float(data["Productivity_Score"].mean())

        # Risk factors
        low_sleep_days = float((data["Sleep_Hours"] < 6).mean())
        very_low_sleep_days = float((data["Sleep_Hours"] < 5).mean())
        low_mood_days = float((data["Mood_Score"] <= 3).mean())
        low_water_days = float((data["Water_Intake_ml"] < 2000).mean())

        # Correlations
        corr_sleep_mood = float(data["Sleep_Hours"].corr(
            data["Mood_Score"])) if data["Sleep_Hours"].nunique() > 1 else 0.0
        corr_steps_mood = float(data["Steps"].corr(
            data["Mood_Score"])) if data["Steps"].nunique() > 1 else 0.0
        corr_study_mood = float(data["Study_Hours"].corr(
            data["Mood_Score"])) if data["Study_Hours"].nunique() > 1 else 0.0
        corr_sleep_study = float(data["Sleep_Hours"].corr(
            data["Study_Hours"])) if data["Sleep_Hours"].nunique() > 1 else 0.0

        return {
            # Core averages
            f"avg_sleep_hours_{period_name}": round(avg_sleep, 2),
            f"avg_steps_{period_name}": round(avg_steps, 0),
            f"avg_study_hours_{period_name}": round(avg_study, 2),
            f"avg_mood_score_{period_name}": round(avg_mood, 2),
            f"avg_water_intake_ml_{period_name}": round(avg_water, 0),
            f"avg_calories_burned_{period_name}": round(avg_calories, 0),

            # Derived scores
            f"avg_energy_score_{period_name}": round(avg_energy, 2),
            f"avg_productivity_score_{period_name}": round(avg_productivity, 2),

            # Risk factors
            f"low_sleep_rate_<6h_{period_name}": round(low_sleep_days, 2),
            f"very_low_sleep_rate_<5h_{period_name}": round(very_low_sleep_days, 2),
            f"low_mood_rate_<=3_{period_name}": round(low_mood_days, 2),
            f"low_water_rate_<2000ml_{period_name}": round(low_water_days, 2),

            # Correlations
            f"sleep_mood_correlation_{period_name}": round(corr_sleep_mood, 2),
            f"steps_mood_correlation_{period_name}": round(corr_steps_mood, 2),
            f"study_mood_correlation_{period_name}": round(corr_study_mood, 2),
            f"sleep_study_correlation_{period_name}": round(corr_sleep_study, 2),
        }

    # Compute stats for all data, last 7 days, and last 30 days
    all_stats = compute_stats_for_period(u, "all")
    stats_7 = compute_stats_for_period(recent_7, "7d")
    stats_30 = compute_stats_for_period(recent_30, "30d")

    # Combine all stats
    patterns = {}
    patterns.update(all_stats)
    patterns.update(stats_7)
    patterns.update(stats_30)

    # Add user info
    patterns["user_id"] = user_id
    patterns["rows"] = int(len(u))

    # Best focus window (using all data)
    by_wake = u.groupby("WakeHour")[
        "Study_Hours"].mean().sort_values(ascending=False)
    patterns["best_wake_hour_for_study"] = int(
        by_wake.index[0]) if len(by_wake) else None

    # Weekend vs weekday patterns
    patterns["weekday_avg_sleep"] = round(
        float(u[~u["Weekend"]]["Sleep_Hours"].mean()), 2)
    patterns["weekend_avg_sleep"] = round(
        float(u[u["Weekend"]]["Sleep_Hours"].mean()), 2)
    patterns["weekday_avg_study"] = round(
        float(u[~u["Weekend"]]["Study_Hours"].mean()), 2)
    patterns["weekend_avg_study"] = round(
        float(u[u["Weekend"]]["Study_Hours"].mean()), 2)

    # Habit relationships (counts using all data)
    patterns["high_sleep_high_mood_days"] = u[(
        u["Sleep_Hours"] >= 7) & (u["Mood_Score"] >= 7)].shape[0]
    patterns["low_sleep_low_mood_days"] = u[(
        u["Sleep_Hours"] < 6) & (u["Mood_Score"] <= 4)].shape[0]
    patterns["active_good_sleep_days"] = u[(
        u["Steps"] >= 10000) & (u["Sleep_Hours"] >= 7)].shape[0]
    patterns["inactive_poor_sleep_days"] = u[(
        u["Steps"] < 5000) & (u["Sleep_Hours"] < 6)].shape[0]
    patterns["studied_good_mood_days"] = u[(
        u["Study_Hours"] >= 3) & (u["Mood_Score"] >= 7)].shape[0]
    patterns["no_study_low_mood_days"] = u[(
        u["Study_Hours"] < 1) & (u["Mood_Score"] <= 4)].shape[0]

    # Trend analysis
    patterns["sleep_trend"] = "improving" if recent_7["Sleep_Hours"].mean(
    ) > recent_30["Sleep_Hours"].mean() else "declining"
    patterns["mood_trend"] = "improving" if recent_7["Mood_Score"].mean(
    ) > recent_30["Mood_Score"].mean() else "declining"
    patterns["study_trend"] = "improving" if recent_7["Study_Hours"].mean(
    ) > recent_30["Study_Hours"].mean() else "declining"
    patterns["recent_7_days_sleep"] = round(recent_7["Sleep_Hours"].mean(), 2)
    patterns["recent_30_days_sleep"] = round(
        recent_30["Sleep_Hours"].mean(), 2)

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

    patterns["tomorrow_risk_score_0_100"] = int(min(100, risk_score))
    patterns["tomorrow_risk_band"] = "Low" if risk_score < 25 else "Medium" if risk_score < 60 else "High"
    patterns["primary_risk_factors"] = risk_factors

    # Most productive days of week
    by_day = u.groupby("DayOfWeek")["Productivity_Score"].mean()
    patterns["most_productive_day"] = by_day.idxmax() if len(by_day) else None
    patterns["least_productive_day"] = by_day.idxmin() if len(by_day) else None

    # Consistency metrics (using all data)
    sleep_consistency = float(
        1 - (u["Sleep_Hours"].std() / u["Sleep_Hours"].mean())) if u["Sleep_Hours"].mean() > 0 else 0
    study_consistency = float(
        1 - (u["Study_Hours"].std() / u["Study_Hours"].mean())) if u["Study_Hours"].mean() > 0 else 0

    patterns["sleep_consistency_index"] = round(sleep_consistency, 2)
    patterns["study_consistency_index"] = round(study_consistency, 2)

    # Recommendations data
    patterns["sleep_goal"] = 7.5
    patterns["steps_goal"] = 10000
    patterns["study_goal"] = 3.0
    patterns["water_goal"] = 3000
    patterns["mood_goal"] = 7.0

    return patterns
