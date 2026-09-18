import streamlit as st

st.set_page_config(
    page_title="Stack Watcher",
    page_icon="👀",
    layout="wide"
)

st.title("👀 Stack Watcher")
st.markdown("""
Welcome to **Stack Watcher**, a machine learning application designed to predict the engagement success of a Stack Overflow question before you even post it!

### 🧭 Navigation
* **👈 Predict Score:** Go to the sidebar and select **Predict Score** to input your draft question and see how it ranks.
* **👈 Model Explainability:** Select **Model Explainability** to dive deep into the SHAP (SHapley Additive exPlanations) values and see exactly *why* the XGBoost model makes its predictions.
""")

st.info("Built with Python, Pandas, XGBoost, and Streamlit.")