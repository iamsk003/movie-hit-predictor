import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64

# Set page config
st.set_page_config(page_title="🎬 Movie Hit or Flop Predictor", layout="centered")

# Custom CSS for dark mode and layout
st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: #FFFFFF;
        }
        .stApp {
            background-color: #000000;
        }
        .css-18e3th9 {
            background-color: #000000;
        }
        .css-1d391kg, .css-1v3fvcr, .css-1cpxqw2, .css-ffhzg2, .css-1n543e5 {
            color: white;
        }
        .stSlider > div[data-baseweb="slider"] > div {
            background-color: #444;
        }
        .css-1v0mbdj.edgvbvh3 {
            color: #FFF;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎬 Movie Hit or Flop Predictor")

# Upload model section
st.sidebar.header("📦 Upload Your Model")
model_file = st.sidebar.file_uploader("Upload Model (.pkl)", type=["pkl"])
columns_file = st.sidebar.file_uploader("Upload Model Columns (.pkl)", type=["pkl"])

# Load default or uploaded model
if model_file is not None and columns_file is not None:
    model = joblib.load(model_file)
    model_columns = joblib.load(columns_file)
else:
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
    input_data["popularity"] = popularity
    input_data["runtime"] = runtime
    input_data["vote_average"] = vote_average
    input_data["vote_count"] = vote_count
    input_data["cast_popularity"] = cast_popularity

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

    # Bar chart visualization
    fig, ax = plt.subplots()
    labels = ['Flop', 'Hit']
    colors = ['#FF4B4B', '#00CC96']
    ax.bar(labels, probability, color=colors)
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)
