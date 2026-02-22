# 🌫️ AQI Research & Forecast Platform

An interactive and explainable Air Quality Index (AQI) forecasting system built using XGBoost and Streamlit.

This platform supports:
- Multi-horizon forecasting (24h / 7d / 30d)
- Historical validation (Actual vs Predicted comparison)
- Scenario simulation (What-If analysis)
- SHAP-based model explainability

---

## 🔗 Live Demo

https://aqiprediction-c73dkekehsulnpsean7c5x.streamlit.app/

---

## 📌 Project Objective

To build an explainable machine learning system capable of forecasting AQI trends and allowing controlled environmental experimentation through an interactive dashboard.

---

## 🚀 Key Features

### 🔮 Forecast Mode (2026)
- 24-hour, 7-day, and 30-day recursive forecasting
- Autoregressive lag features (t-1, t-2, t-3)
- Confidence interval visualization
- Drift control for stability

### 📈 Historical Validation Mode
- Actual vs Predicted AQI comparison
- Hourly resampling
- One-step ahead validation

### 🧪 Research Mode (What-If)
- Modify pollution and weather variables
- Real-time AQI prediction
- SHAP feature contribution analysis

---

## 🧠 Model Details

**Model Used:** XGBoost Regressor  

**Feature Categories:**
- Temporal features (Year, Month, Day, Dayofweek, Hour_sin, Hour_cos)
- Meteorological features (Temperature, Humidity, Pressure, Wind Speed, Wind Direction)
- Pollutant features (PM2.5, PM10, NO2, SO2, O3, CO)
- Autoregressive lag features (aqi_lag1, aqi_lag2, aqi_lag3, rolling mean)

XGBoost was chosen for:
- Strong performance on structured time-series data
- Non-linear modeling capability
- Compatibility with SHAP explainability
- Faster and more interpretable deployment compared to deep LSTM models

---

## 📊 Model Performance

Performance metrics (RMSE, R²) are displayed within the application interface.

Confidence intervals are generated using model residual statistics.

---

## 🏗 Project Structure

AQI_prediction/
│
├── main.py  
├── models/  
│   ├── best_model.pkl  
│   ├── scaler.pkl  
│   ├── feature_list.pkl  
│   ├── labelencoder_location.pkl  
│   ├── labelencoder_city.pkl  
│   ├── metrics.json  
│   └── recent_tail.csv  
│
├── delhi_2000_2025_extended.csv  
└── requirements.txt  

---

## 🖥 Installation (Local)

### Create Virtual Environment
python -m venv myenv  

Activate (Windows):
myenv\Scripts\activate  

### Install Dependencies
pip install -r requirements.txt  

### Run Application
streamlit run main.py  

---

## ⚠️ Notes

- Historical validation works only where actual AQI data exists (2000–2025).
- Forecast mode provides predictions for 2026 without ground truth (real-world forecasting scenario).
- Recursive multi-step forecasting may accumulate prediction error over long horizons.

---

## 🔮 Future Improvements

- Real-time AQI API integration
- Automated retraining pipeline
- Probabilistic forecasting
- Drift detection and monitoring
- Location-specific calibration

---

## 👨‍💻 Author

Aryan  
AQI Research & Forecast Platform
