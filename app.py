import streamlit as st
from analysis import load_data, compute_patterns
from llm import generate_insights

st.set_page_config(page_title="AI Habit Coach", layout="wide")

CSV_PATH = "Daily_Habit_Tracker.csv"

st.title("AI Habit Coach (Hackathon MVP)")
st.caption("CSV → Pattern discovery → LLM coaching → Clean demo")

df = load_data(CSV_PATH)

user_ids = sorted(df["User_ID"].unique().tolist())
user_id = st.selectbox("Select a user", user_ids)

# Calculate patterns once
patterns = compute_patterns(df, user_id)

# Create three columns for better layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("📊 Data Snapshot")
    st.dataframe(df[df["User_ID"] == user_id].sort_values(
        "Date").tail(15), use_container_width=True)

    # Quick stats
    st.subheader("📈 Key Metrics")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Avg Sleep", f"{patterns.get('avg_sleep_hours', 0):.1f}h",
                  f"Goal: {patterns.get('sleep_goal', 7.5)}h")
        st.metric("Avg Steps", f"{patterns.get('avg_steps', 0):,.0f}",
                  f"Goal: {patterns.get('steps_goal', 10000):,.0f}")
        st.metric("Risk Level", patterns.get('tomorrow_risk_band', 'N/A'))

    with metric_col2:
        st.metric("Avg Study", f"{patterns.get('avg_study_hours', 0):.1f}h",
                  f"Goal: {patterns.get('study_goal', 3.0)}h")
        st.metric("Avg Mood", f"{patterns.get('avg_mood_score', 0):.1f}/10",
                  f"Goal: {patterns.get('mood_goal', 7.0)}/10")
        st.metric("Sleep Trend", patterns.get('sleep_trend', 'N/A'))

with col2:
    st.subheader("🔍 Pattern Analysis")

    # Risk factors
    risk_factors = patterns.get('primary_risk_factors', [])
    if risk_factors:
        st.warning(f"⚠️ Primary Risk Factors: {', '.join(risk_factors)}")

    # Habit relationships
    st.write("**Habit Correlations:**")
    col_a, col_b = st.columns(2)
    with col_a:
        corr = patterns.get('sleep_mood_correlation', 0)
        color = "🟢" if corr > 0.3 else "🟡" if corr > 0 else "🔴"
        st.write(f"{color} Sleep→Mood: {corr:.2f}")

        corr = patterns.get('sleep_study_correlation', 0)
        color = "🟢" if corr > 0.3 else "🟡" if corr > 0 else "🔴"
        st.write(f"{color} Sleep→Study: {corr:.2f}")

    with col_b:
        corr = patterns.get('steps_mood_correlation', 0)
        color = "🟢" if corr > 0.3 else "🟡" if corr > 0 else "🔴"
        st.write(f"{color} Steps→Mood: {corr:.2f}")

        corr = patterns.get('study_mood_correlation', 0)
        color = "🟢" if corr > 0.3 else "🟡" if corr > 0 else "🔴"
        st.write(f"{color} Study→Mood: {corr:.2f}")

    # Consistency metrics
    st.write("**Consistency Scores:**")
    cons_col1, cons_col2 = st.columns(2)
    with cons_col1:
        sleep_cons = patterns.get('sleep_consistency_index', 0)
        st.progress(sleep_cons, text=f"Sleep: {sleep_cons:.0%}")
    with cons_col2:
        study_cons = patterns.get('study_consistency_index', 0)
        st.progress(study_cons, text=f"Study: {study_cons:.0%}")

    # Best/Worst days
    best_day = patterns.get('most_productive_day', 'N/A')
    worst_day = patterns.get('least_productive_day', 'N/A')
    if best_day != 'N/A' and worst_day != 'N/A':
        st.write(f"**Most Productive:** {best_day}")
        st.write(f"**Least Productive:** {worst_day}")

    # Expandable detailed patterns
    with st.expander("View Detailed Patterns (JSON)"):
        st.json(patterns)

with col3:
    st.subheader("🤖 AI Coach Insights")

    if st.button("🎯 Generate Personalized Recommendations", type="primary", use_container_width=True):
        with st.spinner("Analyzing your habits and generating recommendations..."):
            insights = generate_insights(patterns)

        st.success("Recommendations generated!")

        # Display insights with better formatting
        st.markdown("---")
        st.markdown(insights)

        # Add a download button for the recommendations
        st.download_button(
            label="📥 Download Recommendations",
            data=insights,
            file_name=f"habit_recommendations_{user_id}.txt",
            mime="text/plain"
        )
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

        # Show quick tips while waiting
        st.markdown("---")
        st.markdown("**💡 Quick Tips Based on Your Data:**")

        if patterns.get('avg_sleep_hours', 0) < 7:
            st.write(
                "• **Sleep Priority**: Your average sleep is below 7 hours. Try going to bed 30 minutes earlier this week.")

        if patterns.get('avg_steps', 0) < 8000:
            st.write(
                "• **Activity Boost**: Add a 15-minute walk to your morning or evening routine.")

        if patterns.get('study_consistency_index', 0) < 0.5:
            st.write(
                "• **Study Consistency**: Try the '25-minute focus, 5-minute break' Pomodoro technique.")

        best_hour = patterns.get('best_wake_hour_for_study')
        if best_hour and best_hour <= 7:
            st.write(
                f"• **Optimal Timing**: You're most productive when waking up around {best_hour}:00 AM.")
