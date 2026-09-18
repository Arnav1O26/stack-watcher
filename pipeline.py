import os
import re
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

FEATURE_COLS = [
    "title_len_words",
    "body_len_words",
    "has_code_snippet",
    "tag_count",
    "user_reputation",
    "account_age_days",
    "post_hour",
    "is_weekend"
]

def generate_synthetic_stackoverflow_data(n_samples=3000, random_state=42):
    """
    Generates a realistic historical dataset of Stack Overflow posts
    with natural correlations to post engagement success.
    """
    np.random.seed(random_state)
    
    # 1. Author Authority
    user_reputation = np.random.exponential(scale=800, size=n_samples).clip(1, 150000).astype(int)
    account_age_days = np.random.randint(1, 4000, size=n_samples)
    
    # 2. Post Quality
    title_len_words = np.random.normal(loc=9, scale=3, size=n_samples).clip(3, 25).astype(int)
    body_len_words = np.random.normal(loc=140, scale=60, size=n_samples).clip(20, 800).astype(int)
    has_code_snippet = np.random.binomial(n=1, p=0.72, size=n_samples)
    tag_count = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.25, 0.4, 0.2, 0.05], size=n_samples)
    
    # 3. Temporal Signals
    post_hour = np.random.randint(0, 24, size=n_samples)
    is_weekend = np.random.binomial(n=1, p=0.28, size=n_samples)
    
    # Engagement calculation: ground-truth success formula (0 to 100)
    # Rewards concise titles, substantial body with code, established rep, and weekday posting
    base_score = (
        (np.log1p(user_reputation) * 4.5) +
        (has_code_snippet * 18.0) +
        (tag_count * 3.2) +
        (np.where((title_len_words >= 6) & (title_len_words <= 14), 10.0, 2.0)) +
        (np.where((body_len_words >= 80) & (body_len_words <= 300), 12.0, 4.0)) +
        (np.where((post_hour >= 13) & (post_hour <= 19), 6.0, 1.0)) -
        (is_weekend * 7.0)
    )
    
    # Add natural variance/noise
    noise = np.random.normal(0, 5, size=n_samples)
    success_score = np.clip(base_score + noise, 0, 100).round(1)
    
    df = pd.DataFrame({
        "title_len_words": title_len_words,
        "body_len_words": body_len_words,
        "has_code_snippet": has_code_snippet,
        "tag_count": tag_count,
        "user_reputation": user_reputation,
        "account_age_days": account_age_days,
        "post_hour": post_hour,
        "is_weekend": is_weekend,
        "success_score": success_score
    })
    return df

def extract_features_from_input(title, body, tags, user_reputation, account_age_days, post_hour, is_weekend):
    """
    Transforms raw user submission inputs into a model-ready pandas DataFrame.
    """
    title_words = len(title.strip().split()) if title else 0
    body_words = len(body.strip().split()) if body else 0
    
    # Check for Markdown or HTML code formatting
    has_code = 1 if ("```" in body or "<code>" in body or re.search(r"    [a-zA-Z0-9_]", body)) else 0
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    num_tags = max(1, min(len(tag_list), 5))
    
    record = {
        "title_len_words": [title_words],
        "body_len_words": [body_words],
        "has_code_snippet": [has_code],
        "tag_count": [num_tags],
        "user_reputation": [int(user_reputation)],
        "account_age_days": [int(account_age_days)],
        "post_hour": [int(post_hour)],
        "is_weekend": [1 if is_weekend else 0]
    }
    return pd.DataFrame(record)[FEATURE_COLS]

def train_and_save_pipeline():
    """
    Trains the XGBoost Regressor and initializes the SHAP explainer,
    saving artifacts for production inference.
    """
    print("1. Generating historical Stack Overflow training data...")
    data = generate_synthetic_stackoverflow_data(n_samples=4000)
    data.to_csv("models/training_data_sample.csv", index=False)
    
    X = data[FEATURE_COLS]
    y = data["success_score"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("2. Training XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate performance
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    print(f"   Train R2: {train_r2:.3f} | Test R2: {test_r2:.3f}")
    
    print("3. Fitting SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    print("4. Persisting artifacts to 'models/'...")
    joblib.dump(model, "models/xgb_model.joblib")
    joblib.dump(explainer, "models/shap_explainer.joblib")
    # Save a small reference sample for SHAP background baseline
    joblib.dump(X_train.sample(100, random_state=42), "models/shap_background.joblib")
    
    print("Pipeline build complete! Artifacts saved successfully.")

if __name__ == "__main__":
    train_and_save_pipeline()