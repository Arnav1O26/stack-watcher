import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Explainability", page_icon="🧠", layout="wide")

@st.cache_resource
def load_artifacts():
    explainer = joblib.load("models/shap_explainer.joblib")
    background = joblib.load("models/shap_background.joblib")
    return explainer, background

explainer, background_data = load_artifacts()

st.title("🧠 XGBoost Model Explainability (SHAP)")
st.write("Understand the inner workings of our engagement prediction model.")

st.subheader("Global Feature Importance")
st.write("This chart shows which features have the biggest impact on predicting a question's success across historical data.")

with st.spinner("Calculating SHAP values..."):
    # Generate SHAP values
    shap_values = explainer(background_data)
    
    # Render standard SHAP summary plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, background_data, show=False)
    st.pyplot(fig)

st.divider()
st.markdown("""
### 📖 How to read this chart:
* **Features on the y-axis:** Ordered by their overall impact on the model. Top features drive the predictions the most.
* **Color (Red vs. Blue):** Represents the feature's value. Red means the value is high (e.g., high user reputation), Blue means low.
* **Position on the x-axis:** Shows whether that specific feature pushed the success score *up* (right of center) or *down* (left of center).
""")