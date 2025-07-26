import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# Set Streamlit page config
st.set_page_config(page_title="🎬 Movie Hit Predictor", layout="wide")

# ---- Load model or fallback ----
MODEL_PATH = "model.pkl"
COLS_PATH = "model_columns.pkl"

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
        model = joblib.load(MODEL_PATH)
        columns = joblib.load(COLS_PATH)
        return model, columns
    return None, None

def save_model(model, columns):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(columns, COLS_PATH)

model, model_columns = load_model()

# ---- Header ----
st.markdown("<h1 style='color:#f63366;'>🎬 Movie Hit or Flop Predictor</h1>", unsafe_allow_html=True)
st.markdown("Fill in the details to predict whether a movie will be a HIT or a FLOP.")

# ---- Training CSV Upload ----
with st.expander("📁 Upload CSV to Retrain Model"):
    uploaded_file = st.file_uploader("Upload CSV with features like budget, popularity, genres etc.", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'success' not in df.columns:
            st.error("❌ Your CSV must include a 'success' column (1=Hit, 0=Flop).")
        else:
            y = df["success"]
            X = df.drop("success", axis=1)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            save_model(model, list(X.columns))
            model_columns = list(X.columns)
            st.success("✅ Model retrained and saved successfully!")

# ---- Input Form ----
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        budget = st.number_input("💰 Budget (USD)", min_value=0, value=5000000, step=500000)
        popularity = st.slider("📊 Popularity (0–300)", 0.0, 300.0, 50.0)
        runtime = st.slider("⏱️ Runtime (minutes)", 30.0, 240.0, 120.0)
        vote_average = st.slider("⭐ Average Vote", 0.0, 10.0, 6.0)
    with col2:
        vote_count = st.number_input("🗳️ Vote Count", min_value=0, value=500, step=10)
        cast_popularity = st.slider("🌟 Cast Popularity (1–10)", 1, 10, 5)

        genres = [
            "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
            "Family", "Fantasy", "Foreign", "History", "Horror", "Music", "Mystery",
            "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
        ]
        selected_genres = st.multiselect("🎞️ Select Genres", genres)

    submit = st.form_submit_button("🔍 Predict")

# ---- Prediction Logic ----
if submit:
    if model is None or model_columns is None:
        st.error("❌ Model not loaded or trained yet. Please upload a CSV to train.")
    else:
        input_data = {col: 0 for col in model_columns}

        # Set numeric inputs
        input_data["budget"] = budget
        input_data["popularity"] = min(popularity, 100)  # normalize
        input_data["runtime"] = runtime
        input_data["vote_average"] = vote_average
        input_data["vote_count"] = vote_count
        input_data["cast_popularity"] = cast_popularity

        # Set genre flags
        for genre in selected_genres:
            col_name = f"genres_{genre}"
            if col_name in input_data:
                input_data[col_name] = 1

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probas = model.predict_proba(input_df)[0]

        # ---- Output Result ----
        result = "✅ HIT" if prediction == 1 else "❌ FLOP"
        confidence = round(np.max(probas) * 100, 2)

        st.markdown(f"<h3>📈 Prediction Result</h3><h4>{result} (Confidence: {confidence}%)</h4>",
                    unsafe_allow_html=True)

        # ---- Bar Plot ----
        fig, ax = plt.subplots(figsize=(4, 2))
        categories = ['Flop', 'Hit']
        ax.bar(categories, probas, color=['#FF4B4B', '#4CAF50'])
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# ---- Footer ----
st.markdown("---")
st.markdown("🔁 You can retrain the model anytime using your own dataset.")
