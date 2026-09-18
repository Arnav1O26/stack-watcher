# 👀 Stack Watcher

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Multipage-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-F7931E.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-2496ED.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)

An end-to-end Machine Learning web application designed to predict the engagement success score (0-100) of a Stack Overflow question before it is posted. It evaluates post quality, user authority, and temporal signals to forecast community response.

## 🚀 Features
* **XGBoost Prediction Engine:** Utilizes advanced gradient boosting to capture non-linear relationships in engagement metrics, achieving high predictive accuracy.
* **Explainable AI (XAI) via SHAP:** Demystifies the "black box" model by visually breaking down exactly how each feature (e.g., reputation, tag count, time of day) impacts the final success score.
* **Multipage Streamlit Dashboard:** Clean, decoupled UI architecture separating the interactive prediction tool from the analytical explainability dashboard.
* **Automated Data Pipeline:** Generates a statistically grounded synthetic dataset of 3,000+ historical posts and engineers predictive features natively in Pandas.

## 🏗️ Project Architecture
* `app.py`: The main entry point and landing page for the application.
* `pages/`: Contains the modular UI views (`1_Predict_Score.py` and `2_Model_Explainability.py`).
* `pipeline.py`: The backend engine responsible for data synthesis, feature engineering, model training, and saving joblib artifacts.
* `models/`: Stores the persisted XGBoost model and SHAP TreeExplainer.
* `Dockerfile`: Configures the lightweight Python 3.12 container for cloud deployment.

## 💻 Local Installation & Setup

1. **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/stack-watcher.git](https://github.com/YOUR-USERNAME/stack-watcher.git)
    cd stack-watcher
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Train the model and generate artifacts:**
    ```bash
    python pipeline.py
    ```

5. **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 🐳 Docker & Cloud Deployment
This application is fully containerized and ready for deployment on platforms like Render, AWS, or Google Cloud. 

To build and run the Docker container locally:
```bash
docker build -t stack-watcher .
docker run -p 8501:8501 stack-watcher
```

## 🔮 Future Advancements (V2 Roadmap)
* **SQL-Native Data Pipeline:** Transition feature engineering off of Pandas and rewrite the data processing layer entirely in SQL. This will utilize **Recursive CTEs** and **Window Functions** for highly optimized, database-level analytics.
* **Real Dataset Integration:** Replace the synthetic data generation pipeline by ingesting the official Stack Exchange Data Dump (or querying via Google BigQuery) to train the model on millions of real-world interactions.