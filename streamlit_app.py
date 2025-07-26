import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Movie Hit Predictor", page_icon="🎬")
st.title("🎬 Movie Hit or Flop Prediction")

# Input fields
budget = st.number_input("💰 Budget ($)", min_value=10000, max_value=500000000, step=100000, value=50000000)
popularity = st.slider("📊 Popularity", 0.0, 100.0, 50.0)
runtime = st.slider("🎞 Runtime (minutes)", 60, 300, 120)
vote_average = st.slider("⭐ Average Vote", 0.0, 10.0, 7.0)
vote_count = st.number_input("🗳 Vote Count", 0, 100000, 2000)
cast_popularity = st.slider("🎭 Cast Popularity", 0.0, 10.0, 5.0)

# Genre checkboxes (dynamic from model_columns)
genre_cols = [col for col in model_columns if col.startswith("genres_")]
genres_input = {
    col: st.checkbox(col.replace("genres_", ""), value=False)
    for col in genre_cols
}

# Build input vector
input_data = {col: 0 for col in model_columns}  # start with all zeros
input_data.update({
    "budget": budget,
    "popularity": popularity,
    "runtime": runtime,
    "vote_average": vote_average,
    "vote_count": vote_count,
    "cast_popularity": cast_popularity,
    **genres_input
})

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("🎯 Predict Movie Success"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.subheader("📈 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Predicted: **HIT** (Confidence: {round(proba[1]*100, 2)}%)")
    else:
        st.error(f"❌ Predicted: **FLOP** (Confidence: {round(proba[0]*100, 2)}%)")

    st.markdown("---")
    st.write("🔍 Debug Info (Input to Model):")
    st.dataframe(input_df)
