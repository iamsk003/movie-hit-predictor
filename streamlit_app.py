import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    popularity = st.slider("📊 Popularity (normalized 0–100)", 0.0, 100.0, 50.0)
    runtime = st.slider("⏱️ Runtime (minutes)", 30.0, 240.0, 120.0)
    vote_average = st.slider("⭐ Average Vote", 0.0, 10.0, 5.0)
    vote_count = st.number_input("🗳️ Vote Count", min_value=0, value=100, step=10)
    cast_popularity = st.slider("🌟 Cast Popularity (1–10)", 1, 10, 5)

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
    input_data = {col: 0 for col in model_columns}
    input_data["budget"] = budget
    input_data["popularity"] = popularity
    input_data["runtime"] = runtime
    input_data["vote_average"] = vote_average
    input_data["vote_count"] = vote_count
    input_data["cast_popularity"] = cast_popularity

    for genre in selected_genres:
        col_name = f"genres_{genre}"
        if col_name in input_data:
            input_data[col_name] = 1

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    result = "✅ HIT" if prediction == 1 else "❌ FLOP"
    confidence = round(np.max(probability) * 100, 2)

    st.subheader("📈 Prediction Result")
    st.success(f"**Predicted: {result} (Confidence: {confidence}%)**")

    # 📊 Confidence Bar Chart
    st.subheader("📊 Confidence Breakdown")
    labels = ["FLOP", "HIT"]
    colors = ['#FF6B6B', '#4CAF50']
    fig, ax = plt.subplots()
    ax.bar(labels, probability, color=colors)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # 🔍 Show Inputs
    st.markdown("### 🔍 Your Inputs:")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}))

# 📁 CSV Upload for Bulk Predictions
st.markdown("---")
st.markdown("## 📂 Bulk Prediction from CSV")
uploaded_file = st.file_uploader("Upload a CSV with the same feature format", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        pred = model.predict(df)
        probs = model.predict_proba(df)
        df['Prediction'] = ['HIT' if p == 1 else 'FLOP' for p in pred]
        df['Confidence (%)'] = np.max(probs, axis=1) * 100
        st.success("✅ Predictions done!")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error processing file: {e}")
