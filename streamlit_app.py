import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="🎬 Movie Hit Predictor", layout="wide")

# Load or fallback model
MODEL_PATH = "model.pkl"
COLS_PATH = "model_columns.pkl"

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
        return joblib.load(MODEL_PATH), joblib.load(COLS_PATH)
    return None, None

def save_model(model, columns):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(columns, COLS_PATH)

model, model_columns = load_model()

# Header
st.markdown("<h1 style='text-align:center; color:#f63366;'>🎬 Movie Hit or Flop Predictor</h1>", unsafe_allow_html=True)

# CSV Upload
with st.expander("📁 Upload CSV to Retrain Model"):
    uploaded_file = st.file_uploader("Upload your movie dataset CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "success" not in df.columns:
            st.error("CSV must include a 'success' column (1=Hit, 0=Flop)")
        else:
            y = df["success"]
            X = df.drop("success", axis=1)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            save_model(model, list(X.columns))
            model_columns = list(X.columns)
            st.success("Model retrained with your data!")

# Input Form
with st.form("predict_form"):
    st.markdown("### 📋 Enter Movie Details")
    col1, col2 = st.columns([1, 1])

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
        selected_genres = st.multiselect("🎞️ Genres", genres)

    submit = st.form_submit_button("🔍 Predict")

# Predict
if submit:
    if not model or not model_columns:
        st.error("❌ Model not available. Please upload a CSV.")
    else:
        input_data = {col: 0 for col in model_columns}
        input_data["budget"] = budget
        input_data["popularity"] = min(popularity, 100)  # Normalize to 100 max
        input_data["runtime"] = runtime
        input_data["vote_average"] = vote_average
        input_data["vote_count"] = vote_count
        input_data["cast_popularity"] = cast_popularity

        for genre in selected_genres:
            col = f"genres_{genre}"
            if col in input_data:
                input_data[col] = 1

        df_input = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0]
        result = "✅ HIT" if prediction == 1 else "❌ FLOP"
        confidence = round(np.max(proba) * 100, 2)

        # Display
        st.markdown(f"<h3 style='color:lightgreen;'>Prediction: {result}</h3>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        # Bar Plot
        st.markdown("### 🔢 Confidence Breakdown")
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.bar(['Flop', 'Hit'], proba, color=["#ff4b4b", "#4caf50"])
        ax.set_ylim([0, 1])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
