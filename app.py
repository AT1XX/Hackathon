import streamlit as st
import pandas as pd
import tempfile
import mistune  # add at the top with other imports

from analysis import load_data, compute_patterns
from llm import generate_insights
from simulator import init_live_df, append_new_day
import markdown as md

# Optional helper for auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="AI Habit Coach", layout="wide")
CSV_PATH = "Daily_Habit_Tracker.csv"


# ----------------------------
# Session State Defaults (for simulation + AI output)
# ----------------------------
if "sim_running" not in st.session_state:
    st.session_state.sim_running = True

if "ai_output" not in st.session_state:
    st.session_state.ai_output = ""

if "last_analyzed_user" not in st.session_state:
    st.session_state.last_analyzed_user = None

if "live_user" not in st.session_state:
    st.session_state.live_user = None

if "live_df" not in st.session_state:
    st.session_state.live_df = None


# ----------------------------
# Helpers
# ----------------------------
def normalize_text(text: str) -> str:
    # Collapse 3+ newlines into max 2
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


# ----------------------------
# UI Header
# ----------------------------
st.title("AI Habit Coach (Hackathon MVP)")
st.caption("CSV → Pattern discovery → Live simulation → LLM coaching → Clean demo")


# ----------------------------
# Sidebar Controls
# ----------------------------
mode = st.sidebar.radio(
    "Data Mode", ["CSV (Static)", "Simulated Live (5s = 1 day)"])

metric_options = {
    "Sleep Hours": "Sleep_Hours",
    "Steps": "Steps",
    "Calories Burned": "Calories_Burned",
    "Water Intake (ml)": "Water_Intake_ml",
    "Study Hours": "Study_Hours",
    "Mood Score": "Mood_Score",
}
metric_label = st.sidebar.selectbox(
    "Chart Metric", list(metric_options.keys()))
metric_col = metric_options[metric_label]


# ----------------------------
# Load Data
# ----------------------------
if mode == "CSV (Static)":
    df = load_data(CSV_PATH)
    user_ids = sorted(df["User_ID"].unique().tolist())
    user_id = st.selectbox("Select a user", user_ids)

else:
    user_id = st.selectbox("Select simulated user", [
                           "SIM_USER_001", "SIM_USER_002", "SIM_USER_003"])
    refresh_secs = st.sidebar.slider("Update interval (seconds)", 1, 10, 5)

    # Initialize / switch user
    if st.session_state.live_df is None or st.session_state.live_user != user_id:
        st.session_state.live_user = user_id
        st.session_state.live_df = init_live_df(
            user_id=user_id, start_date="2025-01-01")

        # reset analysis/controls when switching users
        st.session_state.ai_output = ""
        st.session_state.last_analyzed_user = None
        st.session_state.sim_running = True

    # Auto-update only if sim is running
    if st.session_state.sim_running:
        if HAS_AUTOREFRESH:
            st_autorefresh(interval=refresh_secs * 1000, key="live_refresh")
        else:
            st.sidebar.warning(
                "Install streamlit-autorefresh for true auto-updates: pip install streamlit-autorefresh")

        # Each refresh adds a new "day"
        st.session_state.live_df = append_new_day(
            st.session_state.live_df, user_id=user_id)

    # ✅ Use your load_data() to generate Energy/Productivity/etc for simulated data
    live_raw = st.session_state.live_df.copy()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        live_raw.to_csv(tmp.name, index=False)
        df = load_data(tmp.name)


# ----------------------------
# Compute patterns once
# ----------------------------
patterns = compute_patterns(df, user_id)


# ----------------------------
# Trend Chart (Last 7 Days)
# ----------------------------
st.subheader("📈 Last 7 Days Trend")

user_df = df[df["User_ID"] == user_id].copy().sort_values("Date").tail(7)
chart_df = user_df[["Date", metric_col]].copy()
chart_df["Date"] = pd.to_datetime(chart_df["Date"])
chart_df = chart_df.set_index("Date")

st.line_chart(chart_df, height=260)


# ----------------------------
# 3-column layout (your v1 UI)
# ----------------------------
col1, col2, col3 = st.columns([1, 1, 1])


# ===== Column 1 (v1 snapshot + metrics) =====
with col1:
    st.subheader("📊 Data Snapshot")
    st.dataframe(
        df[df["User_ID"] == user_id].sort_values("Date").tail(15),
        width="stretch"
    )

    st.subheader("📈 Key Metrics")

    # Create tabs for different time periods
    metric_tab1, metric_tab2 = st.tabs(["📅 Last 7 Days", "📅 Last 30 Days"])

    with metric_tab1:
        st.caption("Average values from the last 7 days")
        col1_7d, col2_7d = st.columns(2)

        with col1_7d:
            avg_sleep_7d = patterns.get('avg_sleep_hours_7d', 0)
            st.metric("Avg Sleep", f"{avg_sleep_7d:.1f}h",
                      f"Goal: {patterns.get('sleep_goal', 7.5)}h")

            avg_steps_7d = patterns.get('avg_steps_7d', 0)
            st.metric("Avg Steps", f"{avg_steps_7d:,.0f}",
                      f"Goal: {patterns.get('steps_goal', 10000):,.0f}")

            avg_study_7d = patterns.get('avg_study_hours_7d', 0)
            st.metric("Avg Study", f"{avg_study_7d:.1f}h")

        with col2_7d:
            avg_mood_7d = patterns.get('avg_mood_score_7d', 0)
            st.metric("Avg Mood", f"{avg_mood_7d:.1f}/10",
                      f"Goal: {patterns.get('mood_goal', 7.0)}/10")

            avg_water_7d = patterns.get('avg_water_intake_ml_7d', 0)
            st.metric("Avg Water", f"{avg_water_7d:,.0f}ml",
                      f"Goal: {patterns.get('water_goal', 3000):,.0f}ml")

            st.metric("Risk Level", patterns.get('tomorrow_risk_band', 'N/A'))

    with metric_tab2:
        st.caption("Average values from the last 30 days")
        col1_30d, col2_30d = st.columns(2)

        with col1_30d:
            avg_sleep_30d = patterns.get('avg_sleep_hours_30d', 0)
            st.metric("Avg Sleep", f"{avg_sleep_30d:.1f}h",
                      f"Goal: {patterns.get('sleep_goal', 7.5)}h")

            avg_steps_30d = patterns.get('avg_steps_30d', 0)
            st.metric("Avg Steps", f"{avg_steps_30d:,.0f}",
                      f"Goal: {patterns.get('steps_goal', 10000):,.0f}")

            avg_study_30d = patterns.get('avg_study_hours_30d', 0)
            st.metric("Avg Study", f"{avg_study_30d:.1f}h")

        with col2_30d:
            avg_mood_30d = patterns.get('avg_mood_score_30d', 0)
            st.metric("Avg Mood", f"{avg_mood_30d:.1f}/10",
                      f"Goal: {patterns.get('mood_goal', 7.0)}/10")

            avg_water_30d = patterns.get('avg_water_intake_ml_30d', 0)
            st.metric("Avg Water", f"{avg_water_30d:,.0f}ml",
                      f"Goal: {patterns.get('water_goal', 3000):,.0f}ml")

            st.metric("Sleep Trend", patterns.get('sleep_trend', 'N/A'))


# ===== Column 2 (v1 pattern analysis) =====
with col2:
    st.subheader("🔍 Pattern Analysis")

    risk_factors = patterns.get('primary_risk_factors', [])
    if risk_factors:
        st.warning(f"⚠️ Primary Risk Factors: {', '.join(risk_factors)}")

    # Create tabs for correlation analysis
    corr_tab1, corr_tab2 = st.tabs(["📊 Last 7 Days", "📊 Last 30 Days"])

    with corr_tab1:
        st.write("**Habit Correlations (Last 7 Days):**")
        col_a, col_b = st.columns(2)

        with col_a:
            corr = patterns.get('sleep_mood_correlation_7d', 0)
            color = "🟢" if corr > 0.3 else "🟡" if corr > 0 else "🔴"
            st.write(f"{color} Sleep→Mood: {corr:.2f}")

            corr = patterns.get('sleep_study_correlation_7d', 0)
            color = "🟢" if corr > 0.3 else "🟡" if corr > 0 else "🔴"
            st.write(f"{color} Sleep→Study: {corr:.2f}")

        with col_b:
            corr = patterns.get('steps_mood_correlation_7d', 0)
            color = "🟢" if corr > 0.3 else "🟡" if corr > 0 else "🔴"
            st.write(f"{color} Steps→Mood: {corr:.2f}")

            corr = patterns.get('study_mood_correlation_7d', 0)
            color = "🟢" if corr > 0.3 else "🟡" if corr > 0 else "🔴"
            st.write(f"{color} Study→Mood: {corr:.2f}")

    with corr_tab2:
        st.write("**Habit Correlations (Last 30 Days):**")
        col_a, col_b = st.columns(2)

        with col_a:
            corr = patterns.get('sleep_mood_correlation_30d', 0)
            color = "🟢" if corr > 0.3 else "🟡" if corr > 0 else "🔴"
            st.write(f"{color} Sleep→Mood: {corr:.2f}")

            corr = patterns.get('sleep_study_correlation_30d', 0)
            color = "🟢" if corr > 0.3 else "🟡" if corr > 0 else "🔴"
            st.write(f"{color} Sleep→Study: {corr:.2f}")

        with col_b:
            corr = patterns.get('steps_mood_correlation_30d', 0)
            color = "🟢" if corr > 0.3 else "🟡" if corr > 0 else "🔴"
            st.write(f"{color} Steps→Mood: {corr:.2f}")

            corr = patterns.get('study_mood_correlation_30d', 0)
            color = "🟢" if corr > 0.3 else "🟡" if corr > 0 else "🔴"
            st.write(f"{color} Study→Mood: {corr:.2f}")

    st.write("**Risk Factors by Period:**")
    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        low_sleep_7d = patterns.get('low_sleep_rate_<6h_7d', 0) * 100
        st.write(f"**Last 7 Days:**")
        st.write(f"Low Sleep Days: {low_sleep_7d:.0f}%")
        low_mood_7d = patterns.get('low_mood_rate_<=3_7d', 0) * 100
        st.write(f"Low Mood Days: {low_mood_7d:.0f}%")

    with risk_col2:
        low_sleep_30d = patterns.get('low_sleep_rate_<6h_30d', 0) * 100
        st.write(f"**Last 30 Days:**")
        st.write(f"Low Sleep Days: {low_sleep_30d:.0f}%")
        low_mood_30d = patterns.get('low_mood_rate_<=3_30d', 0) * 100
        st.write(f"Low Mood Days: {low_mood_30d:.0f}%")

    st.write("**Consistency Scores (Overall):**")
    cons_col1, cons_col2 = st.columns(2)

    with cons_col1:
        sleep_cons = patterns.get('sleep_consistency_index', 0)
        sleep_cons = max(0.0, min(1.0, float(sleep_cons)))
        st.progress(sleep_cons, text=f"Sleep: {sleep_cons:.0%}")

    with cons_col2:
        study_cons = patterns.get('study_consistency_index', 0)
        study_cons = max(0.0, min(1.0, float(study_cons)))
        st.progress(study_cons, text=f"Study: {study_cons:.0%}")

    best_day = patterns.get('most_productive_day', 'N/A')
    worst_day = patterns.get('least_productive_day', 'N/A')
    if best_day != 'N/A' and worst_day != 'N/A':
        st.write(f"**Most Productive Day:** {best_day}")
        st.write(f"**Least Productive Day:** {worst_day}")

    with st.expander("View Detailed Patterns (JSON)"):
        st.json(patterns)


# ===== Column 3 (v1 AI block + v2 pause/resume) =====
with col3:
    st.subheader("🤖 AI Coach Insights")

    # In live mode: Analyze + Start Simulation
    if mode == "Simulated Live (5s = 1 day)":
        b1, b2 = st.columns([1, 1])

        with b1:
            analyze_clicked = st.button(
                "🎯 Generate Personalized Recommendations", type="primary", width="stretch")

        with b2:
            start_clicked = False
            if not st.session_state.sim_running:
                start_clicked = st.button(
                    "Start Simulation ▶", width="stretch")
    else:
        analyze_clicked = st.button(
            "🎯 Generate Personalized Recommendations", type="primary", width="stretch")
        start_clicked = False

    if analyze_clicked:
        # Pause simulation when analyzing (live mode only)
        if mode == "Simulated Live (5s = 1 day)":
            st.session_state.sim_running = False

        with st.spinner("Analyzing your habits and generating recommendations..."):
            insights = generate_insights(patterns)

        st.session_state.ai_output = insights
        st.session_state.last_analyzed_user = user_id

        st.rerun()  # show immediately

    if start_clicked:
        st.session_state.sim_running = True
        st.rerun()

    # ---- Output (same behavior as your v1 app.py) ----
    if st.session_state.ai_output and st.session_state.last_analyzed_user == user_id:
        st.success("Recommendations generated!")

        clean = normalize_text(st.session_state.ai_output)

        st.markdown("---")
        st.markdown("Output")


       # At the top of app.py (or before the AI output section)
        st.markdown("""
        <style>
            .scrollable-container ul, .scrollable-container ol {
                list-style-position: inside;
                margin-left: -20px;
            }
            .scrollable-container li {
                display: list-item;
                list-style-type: disc;
            }
        </style>
        """, unsafe_allow_html=True)

        html_content = mistune.html(clean)
        styled_html = f'''
        <div class="scrollable-container" style="height:620px; overflow-y:auto; padding:12px; border:1px solid #3333; border-radius:8px;">
            {html_content}
       
        '''
        st.markdown(styled_html, unsafe_allow_html=True)

    
        st.download_button(
            label="📥 Download Recommendations",
            data=clean,
            file_name=f"habit_recommendations_{user_id}.txt",
            mime="text/plain"
        )

        if mode == "Simulated Live (5s = 1 day)":
            if st.session_state.sim_running:
                st.info("Simulation running.")
            else:
                st.info("Simulation paused. Click **Start Simulation ▶** to resume.")

    else:
        st.info("""
**Click the button to get personalized recommendations:**

The AI will analyze:
- Your sleep patterns and quality
- Activity levels and consistency
- Study/work productivity
- Mood correlations
- Risk factors and trends

You'll receive specific, actionable advice tailored to your data!
""")

        st.markdown("---")
        st.markdown("**💡 Quick Tips Based on Your Data:**")

        # Use 7-day data for quick tips
        avg_sleep_7d = patterns.get('avg_sleep_hours_7d', 0)
        avg_steps_7d = patterns.get('avg_steps_7d', 0)
        study_cons = patterns.get('study_consistency_index', 0)

        if avg_sleep_7d < 7:
            st.write(
                "• **Sleep Priority**: Your average sleep over last 7 days is below 7 hours. Try going to bed 30 minutes earlier this week.")

        if avg_steps_7d < 8000:
            st.write(
                "• **Activity Boost**: Add a 15-minute walk to your morning or evening routine to increase your daily steps.")

        if study_cons < 0.5:
            st.write(
                "• **Study Consistency**: Try the '25-minute focus, 5-minute break' Pomodoro technique to improve consistency.")

        best_hour = patterns.get('best_wake_hour_for_study')
        if best_hour is not None:
            try:
                if int(best_hour) <= 7:
                    st.write(
                        f"• **Optimal Timing**: You're most productive when waking up around {int(best_hour)}:00 AM.")
            except Exception:
                pass

        if mode == "Simulated Live (5s = 1 day)":
            if st.session_state.sim_running:
                st.info(
                    "Simulation running. Click **Generate Personalized Recommendations** to pause and analyze.")
            else:
                st.info("Simulation paused. Click **Start Simulation ▶** to resume.")
