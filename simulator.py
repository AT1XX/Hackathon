# simulator.py
import random
import pandas as pd
from datetime import datetime, timedelta


def _rand_time_str():
    # Wake-up time: between 5:00 and 9:59
    hour = random.randint(5, 9)
    minute = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}"


def generate_day(user_id: str, date: pd.Timestamp) -> dict:
    # Constraints from your dataset description
    sleep_hours = round(random.uniform(4.0, 9.5), 1)
    steps = random.randint(500, 15000)
    calories = random.randint(1200, 3800)
    water = random.randint(500, 5000)
    study = round(random.uniform(0.0, 6.0), 1)
    mood = random.randint(1, 10)

    return {
        "User_ID": user_id,
        "Date": date.strftime("%Y-%m-%d"),
        "Wake_Up_Time": _rand_time_str(),
        "Sleep_Hours": sleep_hours,
        "Steps": steps,
        "Calories_Burned": calories,
        "Water_Intake_ml": water,
        "Study_Hours": study,
        "Mood_Score": mood,
    }


def init_live_df(user_id: str, start_date: str = "2025-01-01") -> pd.DataFrame:
    # Start with 7 days so charts aren’t empty
    d0 = pd.to_datetime(start_date)
    rows = [generate_day(user_id, d0 + timedelta(days=i)) for i in range(7)]
    return pd.DataFrame(rows)


def append_new_day(df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    if df.empty:
        return init_live_df(user_id)

    last_date = pd.to_datetime(df["Date"]).max()
    next_date = last_date + timedelta(days=1)

    new_row = generate_day(user_id, next_date)
    df2 = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # keep it light: store last 30 “days”
    df2["Date"] = pd.to_datetime(df2["Date"])
    df2 = df2.sort_values("Date").tail(30).reset_index(drop=True)

    # Convert back to string format like original CSV (optional)
    df2["Date"] = df2["Date"].dt.strftime("%Y-%m-%d")
    return df2
