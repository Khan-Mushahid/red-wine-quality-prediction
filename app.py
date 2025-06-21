# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and scaler
clf = joblib.load("models/wine_quality_classifier.pkl")
reg = joblib.load("models/wine_quality_regressor.pkl")
scaler = joblib.load("models/feature_scaler.pkl")

# App Title
st.title("🍷 Red Wine Quality Prediction")
st.markdown("Predict wine **quality score** and **quality label** based on physicochemical tests.")

# Sidebar Inputs
st.sidebar.header("Enter Wine Characteristics:")
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
            'pH', 'sulphates', 'alcohol']

user_input = {}
for feat in features:
    user_input[feat] = st.sidebar.slider(
        label=f"{feat.title()}",
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        value=5.0 if feat != 'alcohol' else 10.0
    )

# Create DataFrame from inputs
input_df = pd.DataFrame([user_input])
scaled_input = scaler.transform(input_df)

# Predict
if st.button("🔮 Predict Wine Quality"):
    pred_label = clf.predict(scaled_input)[0]
    pred_score = reg.predict(scaled_input)[0]

    label_map = {0: "Low", 1: "Medium", 2: "High"}
    predicted_label = label_map.get(pred_label, "Unknown")

    st.success(f"💡 Predicted Quality Label: **{predicted_label}**")
    st.info(f"📈 Predicted Quality Score (regression): **{pred_score:.2f} / 10**")

    # Optional: Show inputs back
    st.subheader("Wine Input Summary")
    st.write(input_df)