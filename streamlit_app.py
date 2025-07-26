import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model and expected columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🎬 Movie Hit or Flop Predictor")
st.markdown("Fill in the movie details to predict if it will be a **Hit or Flop**.")

# Basic numeric inputs
budget = st.number_input("Budget (in INR)", value=50000000)
popularity = st.slider("Popularity Score (0–100)", 0.0, 100.0, 50.0)
runtime = st.number_input("Runtime (minutes)", value=120)
vote_average = st.slider("Average Vote (0–10)", 0.0, 10.0, 7.0)
vote_count = st.number_input("Vote Count", value=2000)
cast_popularity = st.slider("Cast Popularity (0–10)", 0.0, 10.0, 5.0)

# Genre checkboxes
st.markdown("🎭 Select applicable genres:")
genres_selected = {
    "genres_Action": st.checkbox("Action"),
    "genres_Adventure": st.checkbox("Adventure"),
    "genres_Sci-Fi": st.checkbox("Sci-Fi"),
    "genres_Drama": st.checkbox("Drama"),
    "genres_Comedy": st.checkbox("Comedy")
}

# Prepare user input dictionary
input_data = {
    "budget": budget,
    "popularity": popularity,
    "runtime": runtime,
    "vote_average": vote_average,
    "vote_count": vote_count,
    "cast_popularity": cast_popularity,
    **genres_selected
}

# Create DataFrame and align columns
X_input = pd.DataFrame([input_data])
X_input = X_input.reindex(columns=model_columns, fill_value=0)

# Prediction
if st.button("Predict Movie Success"):
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    st.subheader("🎯 Prediction:")
    if prediction == 1:
        st.success("✅ This movie is likely to be a **HIT** 🎉")
    else:
        st.error("❌ This movie is likely to be a **FLOP**.")

    # Plot bar chart
    st.subheader("📊 Confidence Level:")
    labels = ["Flop", "Hit"]
    fig, ax = plt.subplots()
    ax.bar(labels, proba * 100, color=["red", "green"])
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)
