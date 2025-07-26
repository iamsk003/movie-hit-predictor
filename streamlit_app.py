import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

# App Title
st.title("🎬 Movie Hit or Flop Predictor")

st.markdown("Enter the details about the movie below:")

# Input form
with st.form("input_form"):
    budget = st.number_input("💰 Budget (in USD)", min_value=0, value=1000000, step=100000)
    popularity = st.slider("📊 Popularity (Raw Scale, e.g. max ~100)", 0.0, 100.0, 5.0)
    runtime = st.slider("⏱️ Runtime (minutes)", 30.0, 240.0, 120.0)
    vote_average = st.slider("⭐ Average Vote", 0.0, 10.0, 5.0)
    vote_count = st.number_input("🗳️ Vote Count", min_value=0, value=100, step=10)
    cast_popularity = st.slider("🌟 Cast Popularity (1-10)", 1, 10, 5)


    # Genre selection
    genres = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
        "Family", "Fantasy", "Foreign", "History", "Horror", "Music", "Mystery",
        "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
    ]
    selected_genres = st.multiselect("🎞️ Select Genres", genres)

    submit = st.form_submit_button("Predict")

# When form is submitted
if submit:
    # Initialize data dictionary with zeros
    input_data = {col: 0 for col in model_columns}

    # Set numerical inputs
    input_data["budget"] = budget
    input_data["popularity"] = (popularity / 300) * 100
    input_data["runtime"] = runtime
    input_data["vote_average"] = vote_average
    input_data["vote_count"] = vote_count
    input_data["cast_popularity"] = cast_popularity / 2  # Scale from 1–10 to 0.5–5

    # Set selected genres
    for genre in selected_genres:
        col_name = f"genres_{genre}"
        if col_name in input_data:
            input_data[col_name] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # Output
    st.subheader("📈 Prediction Result")
    result = "✅ HIT" if prediction == 1 else "❌ FLOP"
    confidence = round(np.max(probability) * 100, 2)

    st.success(f"**Predicted: {result} (Confidence: {confidence}%)**")
