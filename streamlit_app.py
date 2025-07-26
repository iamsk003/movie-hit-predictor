
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Toggle theme mode
dark_mode = st.toggle("🌗 Dark Mode", value=False)

# Apply dark/light theme styles
if dark_mode:
    bg_color = "#111"
    text_color = "#eee"
    pred_bg = "#263238"
    hit_color = "#00e676"
    flop_color = "#ff5252"
else:
    bg_color = "#f2f2f2"
    text_color = "#111"
    pred_bg = "#ffffff"
    hit_color = "#4CAF50"
    flop_color = "#f44336"

st.markdown(f'''
<style>
    body, .main {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .stButton > button {{
        background-color: {hit_color};
        color: white;
    }}
    .prediction-box {{
        background-color: {pred_bg};
        padding: 1em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }}
</style>
''', unsafe_allow_html=True)

# App Title
st.title("🎬 Movie Hit or Flop Predictor")
st.markdown("Enter the details about the movie below:")

# Input form
with st.form("input_form"):
    budget = st.number_input("💰 Budget (in USD)", min_value=0, value=1000000, step=100000)
    popularity = st.slider("📊 Popularity", 0.0, 300.0, 50.0)
    runtime = st.slider("⏱️ Runtime (minutes)", 30.0, 240.0, 120.0)
    vote_average = st.slider("⭐ Average Vote", 0.0, 10.0, 5.0)
    vote_count = st.number_input("🗳️ Vote Count", min_value=0, value=100, step=10)
    cast_popularity = st.slider("🌟 Cast Popularity (1-10)", 1, 10, 3)

    genres = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
        "Family", "Fantasy", "Foreign", "History", "Horror", "Music", "Mystery",
        "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
    ]
    selected_genres = st.multiselect("🎞️ Select Genres", genres)
    submit = st.form_submit_button("Predict")

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

    if prediction == 1:
        hit_animation = "🎉💥🔥🌟 Blockbuster!"
        color = hit_color
    else:
        hit_animation = "💔 Better luck next time!"
        color = flop_color

    st.markdown(f'''
    <div class='prediction-box' style='color:{color};'>
        Predicted: <span style='color:{color};'>{result}</span><br>
        Confidence: <strong>{confidence}%</strong><br>
        {hit_animation}
    </div>
    ''', unsafe_allow_html=True)

    # Show and download prediction result
    result_df = pd.DataFrame([input_data])
    result_df["Prediction"] = result
    result_df["Confidence (%)"] = confidence

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Result as CSV",
        data=csv,
        file_name="prediction_result.csv",
        mime="text/csv"
    )
