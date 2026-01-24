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

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Data snapshot")
    st.dataframe(df[df["User_ID"] == user_id].sort_values(
        "Date").tail(15), use_container_width=True)

    patterns = compute_patterns(df, user_id)
    st.subheader("Extracted patterns (signals)")
    st.json(patterns)

with col2:
    st.subheader("AI Coach Insights")
    if st.button("Analyze & Coach"):
        with st.spinner("Generating insights..."):
            text = generate_insights(patterns)
        st.write(text)
    else:
        st.info("Click **Analyze & Coach** to generate insights for the selected user.")
