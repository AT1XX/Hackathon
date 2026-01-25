# AI Habit Coach (Hackathon MVP)

A lightweight, end-to-end “AI habit coach” that turns daily habit logs into **patterns + risk signals + actionable coaching** using a clean Streamlit dashboard and an LLM endpoint.

✅ Works in **two modes**:
- **CSV (Static)**: analyze the provided dataset (`Daily_Habit_Tracker.csv`)
- **Simulated Live**: generates a new “day” every few seconds (great for demos), pauses on Analyze, resumes on demand

---

## Demo Highlights
- **Last 7 days trend chart** (select metric from sidebar)
- **User snapshot + metrics** (sleep, steps, study, mood, risk level)
- **Pattern analysis** (correlations, consistency, best/worst day)
- **AI coaching report** (structured recommendations + download)

---

## Project Structure
Hackathon/
├─ app.py # Streamlit UI (CSV + Live Simulation)
├─ analysis.py # Feature engineering + pattern extraction
├─ llm.py # LLM client + prompt + response handling
├─ simulator.py # Live habit generator (5s = 1 day)
├─ Daily_Habit_Tracker.csv # Dataset
├─ .env # API keys + endpoint (NOT committed)
└─ README.md

---

## Dataset
`Daily_Habit_Tracker.csv` contains simulated daily activities for **500 virtual users** across 2025.

**Columns**
- `User_ID`
- `Date`
- `Wake_Up_Time` (HH:MM)
- `Sleep_Hours` (5.0–9.5)
- `Steps` (500–20000)
- `Calories_Burned` (1200–3800)
- `Water_Intake_ml` (500–5000)
- `Study_Hours` (0–8)
- `Mood_Score` (1–10)

`analysis.py` adds derived signals like:
- `Energy_Score`
- `Productivity_Score`
- `SuccessLabel`
- trends / correlations / risk score

---

## Requirements

**Python**: 3.10+ recommended  
**Libraries**:
- streamlit
- pandas
- numpy
- python-dotenv
- openai (for OpenAI-compatible endpoints)
- streamlit-autorefresh (optional for live mode)
- markdown (optional if rendering markdown to HTML inside a scroll box)

---

## Setup

### 1) Create & activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Mac/Linux
```
python3 -m venv .venv
source .venv/bin/activate
```
2) Install dependencies
```pip install -r requirements.txt```


If you don’t have a requirements.txt yet, install manually:
```
pip install streamlit pandas numpy python-dotenv openai streamlit-autorefresh markdown
```

Configure the LLM (.env)

Create a file named .env in the project root:
```
MODEL_API_KEY=your_key_here
MODEL_ENDPOINT=https://your-endpoint-host/v1
MODEL_NAME=google/gemma-3-27b-it
```

Notes:

- MODEL_ENDPOINT should point to the OpenAI-compatible base URL (usually ending in /v1)
- MODEL_NAME must match a model returned by your provider’s /models endpoint

Run the App
```
streamlit run app.py
```
Open:
```
Local: http://localhost:8501
```

How Live Simulation Works

In Simulated Live mode:
- A new “day” is appended every N seconds (5s = 1 day)
- The dashboard updates automatically (using streamlit-autorefresh)
- Clicking Generate Personalized Recommendations will:
  - Pause the simulation
  - Run pattern extraction
  - Call the LLM for coaching
  - Show Start Simulation ▶ button to resume

