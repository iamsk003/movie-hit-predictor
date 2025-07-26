{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import numpy as np\
import joblib\
import pandas as pd\
import matplotlib.pyplot as plt\
\
# Load model\
model = joblib.load("model.pkl")\
\
st.title("\uc0\u55356 \u57260  Movie Hit or Flop Predictor")\
\
st.markdown("Enter your movie details below:")\
\
# User input\
budget = st.number_input("Budget (in INR)", value=50000000)\
popularity = st.slider("Popularity (0\'96100)", 0.0, 100.0, 50.0)\
runtime = st.number_input("Runtime (in minutes)", value=120)\
vote_average = st.slider("Vote Average (0\'9610)", 0.0, 10.0, 7.0)\
vote_count = st.number_input("Vote Count", value=2000)\
cast_popularity = st.slider("Cast Popularity (0\'9610)", 0.0, 10.0, 5.0)\
\
st.markdown("Select movie genres:")\
\
genres = \{\
    "genres_Action": st.checkbox("Action"),\
    "genres_Adventure": st.checkbox("Adventure", value=True),\
    "genres_Sci-Fi": st.checkbox("Sci-Fi"),\
    "genres_Drama": st.checkbox("Drama", value=True),\
    "genres_Comedy": st.checkbox("Comedy", value=True),\
\}\
\
# Combine all inputs\
input_data = \{\
    "budget": budget,\
    "popularity": popularity,\
    "runtime": runtime,\
    "vote_average": vote_average,\
    "vote_count": vote_count,\
    "cast_popularity": cast_popularity,\
    **genres\
\}\
\
# Convert to DataFrame\
X_input = pd.DataFrame([input_data])\
\
# Predict\
if st.button("Predict"):\
    prediction = model.predict(X_input)[0]\
    proba = model.predict_proba(X_input)[0]\
\
    # Output\
    st.subheader("\uc0\u55356 \u57263  Prediction:")\
    if prediction == 1:\
        st.success("\uc0\u9989  This movie is likely to be a **HIT**! \u55356 \u57225 ")\
    else:\
        st.error("\uc0\u10060  This movie is likely to be a **FLOP**.")\
\
    # Bar chart\
    st.subheader("\uc0\u55357 \u56522  Confidence:")\
    labels = ["Flop", "Hit"]\
    fig, ax = plt.subplots()\
    ax.bar(labels, proba * 100, color=["red", "green"])\
    ax.set_ylabel("Probability (%)")\
    ax.set_ylim([0, 100])\
    st.pyplot(fig)\
}