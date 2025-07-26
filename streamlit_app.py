import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="🎬 Movie Hit or Flop Predictor", layout="centered")

# Custom styling
st.markdown("""
    <style>
        body, .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
            font-family: "Segoe UI", sans-serif;
        }

        input, select, textarea {
            background-color: #f9f9f9 !important;
            color: #000000 !important;
            border: 1px solid #cccccc !important;
            border-radius: 6px;
            padding: 6px;
        }

        .stSlider > div[data-baseweb="slider"] > div {
            background-color: transparent !important;
        }

        .stSlider > div[data-baseweb="slider"] > div > div:first-child {
            background-color: #4a90e2 !important;
            border-radius: 4px;
        }

        .stSlider > div[data-baseweb="slider"] span[role="slider"] {
            background-color: #ffffff !important;
            border: 2px solid #4a90e2 !important;
            border-radius: 50%;
            width: 20px !important;
            height: 20px !important;
            box-shadow: 0 0 4px rgba(0, 0, 0, 0.15);
            margin-top: -8px;
        }

        .stMultiSelect, .stSelectbox {
            background-color: #f7f7f7 !important;
            color: #000000 !important;
        }

        svg text {
            fill: #000000 !important;
        }

        .stSlider mark,
        .stSlider mark::before,
        .stSlider mark::after {
            background: none !important;
            color: #000000 !important;
            border: none !important;
            box-shadow: none !important;
            font-weight: normal !important;
            padding: 0 !important;
            margin: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎬 Movie Hit or Flop Predictor")

# Load model and columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.markdown("Enter the details about the movie below:")

# Input form
with st.form("input_form"):
    budget = st.number_input("💰 Budget (in USD)", min_value=0, value=1000000, step=100000)
    popularity = st.slider("📊 Popularity (0-100 scale)", 0.0, 100.0, 50.0)
    runtime = st.slider("⏱️ Runtime (minutes)", 30.0, 240.0, 120.0)
    vote_average = st.slider("⭐ Average Vote", 0.0, 10.0, 5.0)
    vote_count = st.number_input("🗳️ Vote Count", min_value=0, value=100, step=10)
    cast_popularity = st.slider("🌟 Cast Popularity (1-10)", 1, 10, 5)

    genres = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
        "Family", "Fantasy", "Foreign", "History", "Horror", "Music", "Mystery",
        "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
    ]
    selected_genres = st.multiselect("🎞️ Select Genres", genres)

    submit = st.form_submit_button("Predict")

# On submit
if submit:
    input_data = {col: 0 for col in model_columns}

    # Numeric features
    input_data["budget"] = budget
    input_data["popularity"] = popularity
    input_data["runtime"] = runtime
    input_data["vote_average"] = vote_average
    input_data["vote_count"] = vote_count
    input_data["cast_popularity"] = cast_popularity

    # One-hot encode genres
    for genre in selected_genres:
        col_name = f"genres_{genre}"
        if col_name in input_data:
            input_data[col_name] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # Display result
    st.subheader("📈 Prediction Result")
    result = "✅ HIT" if prediction == 1 else "❌ FLOP"
    confidence = round(np.max(probability) * 100, 2)

    st.success(f"**Predicted: {result} (Confidence: {confidence}%)**")

    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(["Flop", "Hit"], probability, color=['#FF4B4B', '#00CC96'])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)
