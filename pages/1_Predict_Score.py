import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from pipeline import extract_features_from_input

st.set_page_config(page_title="Predict Score", page_icon="📈")

@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.joblib")

model = load_model()

st.title("📈 Predict Question Success")
st.write("Enter your Stack Overflow draft details below.")

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Question Title", "How to invert a binary tree in Python?")
    tags = st.text_input("Tags (comma separated)", "python, binary-tree, data-structures")
    rep = st.number_input("Your Stack Overflow Reputation", min_value=1, value=150)
    age = st.number_input("Account Age (Days)", min_value=1, value=365)
    
with col2:
    hour = st.slider("Time of Post (Hour 0-23)", min_value=0, max_value=23, value=14)
    is_weekend = st.checkbox("Posting on a Weekend?")
    body = st.text_area("Question Body", "I am trying to invert a binary tree in Python, here is my code: \n```python\n# code here\n```\nWhat am I doing wrong?", height=150)

if st.button("Predict Success Score", type="primary"):
    # Extract features using our pandas pipeline
    features_df = extract_features_from_input(title, body, tags, rep, age, hour, is_weekend)
    
    # Predict using the XGBoost model
    score = model.predict(features_df)[0]
    
    st.divider()
    st.subheader("Predicted Engagement Score")
    
    # Plotly Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Success Score (0-100)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1E88E5"},
            'steps': [
                {'range': [0, 40], 'color': "#FFCDD2"},
                {'range': [40, 70], 'color': "#FFF9C4"},
                {'range': [70, 100], 'color': "#C8E6C9"}],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)