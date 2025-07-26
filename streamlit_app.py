import streamlit as st
import pandas as pd
import joblib

# Load trained model and expected feature columns
with open("model.pkl", "rb") as f:
    model = joblib.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = joblib.load(f)

st.set_page_config(page_title="Movie Hit Predictor", layout="centered")
st.title("🎬 Movie Hit/Flop Prediction App")

st.markdown("""
Enter the movie details below to predict whether your movie will be a **HIT** or **FLOP** based on machine learning.
""")

# Step 1: Initialize input data with 0s
input_data = {col: 0 for col in model_columns}

# Step 2: Collect numeric inputs
input_data["budget"] = st.number_input("💰 Budget (in ₹)", value=50000000)
input_data["popularity"] = st.number_input("🔥 Popularity Score", value=50.0)
input_data["runtime"] = st.number_input("🎞 Runtime (minutes)", value=120)
input_data["vote_average"] = st.slider("⭐ Average Rating", 0.0, 10.0, 7.0)
input_data["vote_count"] = st.number_input("🗳 Vote Count", value=2000)
input_data["cast_popularity"] = st.number_input("👥 Cast Popularity Score", value=5.0)

# Step 3: Genre selection based on available columns
st.markdown("🎭 **Select Genres**")
genre_cols = [col for col in model_columns if col.startswith("genres_")]
for genre_col in genre_cols:
    genre_name = genre_col.replace("genres_", "")
    if st.checkbox(genre_name):
        input_data[genre_col] = 1

# Step 4: Build input dataframe
input_df = pd.DataFrame([input_data])

# Optional Debug: show model inputs
# st.write("🔍 Input DataFrame:")
# st.dataframe(input_df)

# Step 5: Predict
if st.button("🚀 Predict Movie Success"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][prediction] * 100

    st.subheader("📈 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Predicted: **HIT** (Confidence: {proba:.1f}%)")
    else:
        st.error(f"❌ Predicted: **FLOP** (Confidence: {proba:.1f}%)")

    # Extra: show full probability breakdown
    st.markdown("### 🔢 Prediction Probabilities")
    st.write({
        "Flop (0)": f"{model.predict_proba(input_df)[0][0]*100:.1f}%",
        "Hit (1)": f"{model.predict_proba(input_df)[0][1]*100:.1f}%"
    })
