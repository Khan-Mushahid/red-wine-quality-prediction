# 🍷 Red Wine Quality Prediction

This project is a complete **end-to-end machine learning pipeline** built to predict the quality of red wine based on its physicochemical properties. Using the **UCI Wine Quality dataset**, it implements classification and regression models to evaluate wine quality and exposes an interactive web app via **Streamlit**.

---

## 🔗 Live Web App

👉 [Click here to try the deployed app](https://your-streamlit-link.streamlit.app)

> _Note: If the app doesn’t open, it may not be deployed yet. You can clone this repo to run it locally._

---

## 📌 Project Overview

- 🔍 **Problem**: Wine quality is typically judged by taste testers—this project aims to automate the prediction using chemical properties of the wine.
- 🎯 **Objective**:
  - Predict **quality score** (regression)
  - Predict **quality label** - Low / Medium / High (classification)
  - Provide an interactive UI for users to test predictions
- 🚀 **Tech Stack**:
  - Python, Pandas, NumPy, Scikit-learn
  - Streamlit for app deployment
  - Joblib for model serialization

---

## 📂 Dataset

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)
- **File**: `winequality-red.csv`
- **Records**: 1,599 red wine samples
- **Features**: 11 chemical properties + target label (`quality`)

---

## 📊 Features Used

- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`

---

## 🧠 Models Used

| Model Type        | Algorithm         | Accuracy / Performance |
|-------------------|-------------------|-------------------------|
| Classification    | Random Forest     | ~79% Accuracy           |
| Regression        | Random Forest     | RMSE ≈ 0.63, R² ≈ 0.53 |

---

## 📁 Project Structure

```bash
📦 red-wine-quality-prediction
├── app.py                  # Streamlit application
├── requirements.txt        # All required dependencies
├── models/
│   ├── wine_quality_classifier.pkl
│   ├── wine_quality_regressor.pkl
│   └── feature_scaler.pkl
├── Suggestions_Dataset.csv # Feedback responses (optional)
