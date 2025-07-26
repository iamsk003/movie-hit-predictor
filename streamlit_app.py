import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import base64
import os

# Page config
st.set_page_config(page_title="🎬 Movie Hit or Flop Predictor", layout="centered")

# Custom CSS loader
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

st.title("🎬 Movie Hit or Flop Predictor")

# Load default model
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Upload CSV for training
st.subheader("📤 Train Your Own Model from CSV")
uploaded_csv = st.file_uploader("Upload a CSV file with features and a 'success' column", type=["csv"])
train_button = st.button("🧠 Train Model")

if uploaded_csv is not None and train_button:
    try:
        df = pd.read_csv(uploaded_csv)
        if 'success' not in df.columns:
            st.error("❌ CSV must contain a 'success' column (1=Hit, 0=Flop)")
        else:
            st.success("✅ CSV uploaded. Training model...")

            y = df["success"]
            X = df.drop("success", axis=1)

            X = pd.get_dummies(X)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            joblib.dump(clf, "model.pkl")
            joblib.dump(X.columns.tolist(), "model_columns.pkl")
            model = clf
            model_columns = X.columns.tolist()

            st.success("🎉 Model trained and loaded for predictions!")

            st.markdown("### 🧪 Training Metrics")
            st.write(f"✅ **Accuracy:** {accuracy:.2f}")
            st.write(f"✅ **Precision:** {precision:.2f}")
            st.write(f"✅ **Recall:** {recall:.2f}")
            st.write(f"✅ **F1 Score:** {f1:.2f}")

            def get_download_link(file_path, file_label):
                with open(file_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:file/pkl;base64,{b64}" download="{file_path}">📥 Download {file_label}</a>'
                    return href

            st.markdown("---")
            st.markdown("### 📥 Download Trained Artifacts")
            st.markdown(get_download_link("model.pkl", "Model"), unsafe_allow_html=True)
            st.markdown(get_download_link("model_columns.pkl", "Model Columns"), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# Input form
st.markdown("### 🎯 Enter movie details for prediction")
with st.form("input_form"):
    budget = st.number_input("💰 Budget (USD)", min_value=0, value=1000000, step=100000)
    popularity = st.slider("📊 Popularity (0-100)", 0.0, 100.0, 50.0)
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

# Prediction section
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

    st.subheader("📈 Prediction Result")
    result = "✅ HIT" if prediction == 1 else "❌ FLOP"
    confidence = round(np.max(probability) * 100, 2)
    st.success(f"**Predicted: {result} (Confidence: {confidence}%)**")

    # Bar chart
    fig, ax = plt.subplots()
    labels = ['Flop', 'Hit']
    colors = ['#FF4B4B', '#00CC96']
    ax.bar(labels, probability, color=colors)
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)
