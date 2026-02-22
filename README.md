# 🌫️ AQI Research & Forecast Platform

An interactive, explainable Air Quality Index (AQI) forecasting and research dashboard built using **XGBoost** and **Streamlit**.  
This platform supports **multi-horizon forecasting (24h / 7d / 30d)**, **scenario simulation (What-If analysis)**, and **historical validation (Actual vs Predicted)**.

---

## 🔗 Live Demo

https://aqiprediction-c73dkekehsulnpsean7c5x.streamlit.app/

---

## 📌 Project Objective

To build an explainable machine learning system that can:
- Forecast future AQI levels (2026)
- Validate predictions on historical data (2000–2025)
- Provide feature-level interpretability using SHAP
- Allow controlled experimentation via scenario simulation

---

## 🚀 Key Features

### 🔮 1. Forecast Mode (2026)
- Multi-horizon forecasting:
  - 24 Hours
  - 7 Days
  - 30 Days
- Recursive (autoregressive) forecasting using lag features (t-1, t-2, t-3)
- 95% Confidence Interval bands
- Drift control for stability

---

### 🧪 2. Research Mode (What-If + Explainability)
- Adjust:
  - AQI Lag values
  - Weather conditions
  - Pollutant levels
- Instant prediction response
- AQI category classification
- SHAP-based feature contribution visualization

---

### 📈 3. Historical Validation Mode
- Compare **Actual vs Predicted AQI**
- Hourly resampling
- One-step ahead validation
- Demonstrates real-world reliability

---

## 🧠 Model Details

### Model Used
- **XGBoost Regressor**

### Why XGBoost?
- Strong performance on structured time-series data
- Handles non-linear relationships
- Works well with SHAP for explainability
- Faster and more interpretable than deep LSTM models

---

## 📊 Feature Engineering

### Time Features
- Year
- Month
- Day
- Dayofweek
- Hour_sin / Hour_cos (cyclical encoding)

### Meteorological Features
- Temperature
- Humidity
- Pressure
- Wind Speed
- Wind Direction

### Pollutant Features
- PM2.5
- PM10
- NO2
- SO2
- O3
- CO

### Autoregressive Lag Features
- aqi_lag1
- aqi_lag2
- aqi_lag3
- Rolling average (aqi_roll3)

These lag features help capture short-term AQI trends.

---

## 📊 Model Performance

Performance metrics are displayed inside the app UI:
- RMSE
- R² Score
- Residual Standard Deviation (used for confidence bands)

---

## 🏗 Project Structure
