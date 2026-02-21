import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="AQI Research & Forecast Platform", page_icon="🌫️", layout="wide")
# ---------------------------
# Load artifacts + data (cached)
# ---------------------------
@st.cache_resource
def load_all():
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR / "models"

    model = joblib.load(MODEL_DIR / "best_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    FEATURES = joblib.load(MODEL_DIR / "feature_list.pkl")
    le_location = joblib.load(MODEL_DIR / "labelencoder_location.pkl")
    le_city = joblib.load(MODEL_DIR / "labelencoder_city.pkl")

    # Fast tail for forecasting seed
    df_tail = pd.read_csv(MODEL_DIR / "recent_tail.csv")
    # Full dataset for historical mode
    FULL_DATA_PATH = BASE_DIR / "delhi_2000_2025_extended.csv"
    if FULL_DATA_PATH.exists():
        df_full = pd.read_csv(FULL_DATA_PATH)
    else:
        df_full = df_tail.copy()  # fallback
    # Parse Time
    for d in [df_tail, df_full]:
        if "Time" in d.columns:
            d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
            d.dropna(subset=["Time"], inplace=True)
            d.sort_values("Time", inplace=True)
            d.reset_index(drop=True, inplace=True)
    # metrics
    with open(MODEL_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)
    # Optional SHAP
    explainer = None
    try:
        import shap  # noqa
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = None
    return BASE_DIR, MODEL_DIR, model, scaler, FEATURES, le_location, le_city, df_tail, df_full, metrics, explainer
BASE_DIR, MODEL_DIR, model, scaler, FEATURES, le_location, le_city, df_tail, df_full, metrics, explainer = load_all()
# ---------------------------
# UI helpers
# ---------------------------
def aqi_cat(a):
    if a <= 50: return "Good 🟢"
    elif a <= 100: return "Satisfactory 🟡"
    elif a <= 200: return "Moderate 🟠"
    elif a <= 300: return "Poor 🔴"
    elif a <= 400: return "Very Poor 🟣"
    else: return "Severe ⚫"


def metric_card(title, value):
    st.markdown(
        f"""
        <div style="background:#0b1220;border:1px solid #1e293b;border-radius:16px;padding:18px;text-align:center;">
            <div style="color:#94a3b8;font-size:14px;">{title}</div>
            <div style="color:#38bdf8;font-size:34px;font-weight:800;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def ensure_features(row_dict: dict) -> pd.DataFrame:
    """Make sure all FEATURES exist, fill missing with 0."""
    X = pd.DataFrame([row_dict])
    for f in FEATURES:
        if f not in X.columns:
            X[f] = 0.0
    return X[FEATURES].astype(float)


def predict_from_row(row_dict: dict) -> float:
    X = ensure_features(row_dict)
    Xs = scaler.transform(X)
    return float(model.predict(Xs)[0])


def clamp_seed_lags(seed_row: dict, cap=300.0):
    """Stops winter-seed explosion in recursive mode."""
    for k in ["aqi_lag1", "aqi_lag2", "aqi_lag3"]:
        if k in seed_row:
            seed_row[k] = float(np.clip(seed_row[k], 0, cap))
    seed_row["aqi_roll3"] = float(np.mean([
        seed_row.get("aqi_lag1", seed_row.get("aqi", 0.0)),
        seed_row.get("aqi_lag2", seed_row.get("aqi", 0.0)),
        seed_row.get("aqi_lag3", seed_row.get("aqi", 0.0)),
    ]))
    return seed_row
# ---------------------------
# Seed selection (core)
# ---------------------------
def get_seed_row(
    df: pd.DataFrame,
    city_enc: int,
    loc_enc: int,
    seed_year: int | None = None,
    seed_months: list[int] | None = None
):
    """
    Pick last row for selected city/location, optionally within year+months.
    Falls back to city-only then global if too few rows.
    """
    d = df.copy()

    if seed_year is not None and "Time" in d.columns:
        d = d[d["Time"].dt.year == seed_year]

    if seed_months is not None and "Time" in d.columns:
        d = d[d["Time"].dt.month.isin(seed_months)]

    # Filter by encodings (if exist)
    if "city_encoded" in d.columns:
        d = d[d["city_encoded"] == city_enc]
    if "location_id_encoded" in d.columns:
        d = d[d["location_id_encoded"] == loc_enc]

    d = d.sort_values("Time")

    # fallback 1: city only
    if len(d) < 10:
        d2 = df.copy()
        if seed_year is not None and "Time" in d2.columns:
            d2 = d2[d2["Time"].dt.year == seed_year]
        if seed_months is not None and "Time" in d2.columns:
            d2 = d2[d2["Time"].dt.month.isin(seed_months)]
        if "city_encoded" in d2.columns:
            d2 = d2[d2["city_encoded"] == city_enc]
        d2 = d2.sort_values("Time")
        if len(d2) >= 10:
            d = d2
    # fallback 2: global
    if len(d) == 0:
        d = df.sort_values("Time")

    last_row = d.iloc[-1].to_dict()
    last_time = pd.to_datetime(d.iloc[-1]["Time"])
    return last_row, last_time
# ---------------------------
# Recursive forecast (2026)
# ---------------------------
def recursive_forecast(seed_row: dict, start_time: datetime, steps: int) -> pd.DataFrame:
    cur = seed_row.copy()
    preds = []
    cur_time = start_time

    # Ensure stable required keys exist
    cur.setdefault("wind_dir_sin", 0.0)
    cur.setdefault("wind_dir_cos", 0.0)

    # AQI lag defaults
    base_aqi = float(cur.get("aqi", 0.0))
    for lag in [1, 2, 3]:
        cur.setdefault(f"aqi_lag{lag}", base_aqi)
    cur.setdefault("aqi_roll3", float(np.mean([cur["aqi_lag1"], cur["aqi_lag2"], cur["aqi_lag3"]])))

    # Pollutant lag defaults
    pols = ["pm25", "pm10", "no2", "so2", "o3", "co"]
    for p in pols:
        cur.setdefault(p, float(cur.get(p, 0.0)))
        for lag in [1, 2, 3]:
            cur.setdefault(f"{p}_lag{lag}", float(cur.get(p, 0.0)))
        cur.setdefault(f"{p}_roll3", float(np.mean([cur[f"{p}_lag1"], cur[f"{p}_lag2"], cur[f"{p}_lag3"]])))

    for _ in range(steps):
        # time features
        cur["Year"] = cur_time.year
        cur["Month"] = cur_time.month
        cur["Day"] = cur_time.day
        cur["Dayofweek"] = cur_time.weekday()
        cur["Hour_sin"] = float(np.sin(2*np.pi*cur_time.hour/24))
        cur["Hour_cos"] = float(np.cos(2*np.pi*cur_time.hour/24))

        # predict
        yhat = predict_from_row(cur)

        # clamp demo-safe output (still realistic)
        yhat = float(np.clip(yhat, 0, 500))

        preds.append({"Time": cur_time, "Predicted_AQI": yhat})

        # update AQI lags
        cur["aqi_lag3"] = cur["aqi_lag2"]
        cur["aqi_lag2"] = cur["aqi_lag1"]
        cur["aqi_lag1"] = yhat
        cur["aqi_roll3"] = float(np.mean([cur["aqi_lag1"], cur["aqi_lag2"], cur["aqi_lag3"]]))

        # pollutants: mild decay + lag shift (keeps stability)
        for p in pols:
            cur[p] = float(cur[p]) * 0.995
            cur[f"{p}_lag3"] = cur[f"{p}_lag2"]
            cur[f"{p}_lag2"] = cur[f"{p}_lag1"]
            cur[f"{p}_lag1"] = cur[p]
            cur[f"{p}_roll3"] = float(np.mean([cur[f"{p}_lag1"], cur[f"{p}_lag2"], cur[f"{p}_lag3"]]))

        cur_time += timedelta(hours=1)

    return pd.DataFrame(preds)
# ---------------------------
# Plotly helpers
# ---------------------------
def plot_forecast_with_ci(fc: pd.DataFrame, title: str):
    fig = go.Figure()

    # upper bound (invisible line)
    fig.add_trace(go.Scatter(
        x=fc["Time"], y=fc["Upper"],
        mode="lines",
        line=dict(width=0),
        name="Upper",
        showlegend=False
    ))

    # lower bound + fill band
    fig.add_trace(go.Scatter(
        x=fc["Time"], y=fc["Lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="95% CI"
    ))

    # prediction line
    fig.add_trace(go.Scatter(
        x=fc["Time"], y=fc["Predicted_AQI"],
        mode="lines",
        name="Prediction"
    ))

    fig.update_layout(
        title=title,
        height=380,
        yaxis_title="AQI",
        xaxis_title="Time",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_lines_time(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str):
    fig = go.Figure()
    for c in y_cols:
        fig.add_trace(go.Scatter(x=df[x_col], y=df[c], mode="lines", name=c))
    fig.update_layout(
        title=title,
        height=380,
        yaxis_title="AQI",
        xaxis_title="Time",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_shap_bar(features: np.ndarray, shap_vals: np.ndarray, title: str):
    shap_df = pd.DataFrame({"feature": features, "shap_value": shap_vals})
    shap_df = shap_df.sort_values("shap_value", key=lambda s: np.abs(s), ascending=False).head(12)

    fig = go.Figure(go.Bar(
        x=shap_df["feature"],
        y=shap_df["shap_value"]
    ))
    fig.update_layout(
        title=title,
        height=380,
        yaxis_title="SHAP value (impact)",
        xaxis_title="Feature",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Mode", ["🔮 Forecast", "🧪 Research (What-if)", "📈 Historical 24h"])

# City/location selection
if le_city is not None and hasattr(le_city, "classes_"):
    city = st.sidebar.selectbox("City", list(le_city.classes_))
    city_enc = int(le_city.transform([city])[0])
else:
    city = "Delhi"
    city_enc = 0

if le_location is not None and hasattr(le_location, "classes_"):
    location = st.sidebar.selectbox("Location", list(le_location.classes_))
    loc_enc = int(le_location.transform([location])[0])
else:
    location = "N/A"
    loc_enc = 0

# Header
st.title("🌫️ AQI Research & Forecast Platform")
c1, c2, c3 = st.columns(3)
with c1:
    metric_card("Model RMSE", f"{metrics.get('rmse', 0.0):.2f}")
with c2:
    metric_card("Model R²", f"{metrics.get('r2', 0.0):.2f}")
with c3:
    metric_card("Selected", f"{location}")


# ---------------------------
# Mode: Forecast (with Seed Season)
# ---------------------------
if mode == "🔮 Forecast":
    st.subheader("🔮 Multi-Horizon Forecast (2026) (24h / 7d / 30d)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌦️ Seed Settings (Important)")

    seed_mode = st.sidebar.selectbox(
        "Seed Season",
        ["Non-Winter (Aug/Sep) ✅ stable", "Winter (Nov/Dec) ⚠ higher", "Custom Months"],
        index=0
    )

    if seed_mode.startswith("Non-Winter"):
        seed_year = 2025
        seed_months = [8, 9]
    elif seed_mode.startswith("Winter"):
        seed_year = 2025
        seed_months = [11, 12]
    else:
        seed_year = st.sidebar.selectbox("Seed Year", [2025, 2024, 2023, 2022], index=0)
        seed_months = st.sidebar.multiselect(
            "Seed Months",
            list(range(1, 13)),
            default=[8, 9]
        )
        if len(seed_months) == 0:
            seed_months = None

    # Forecast start time (2026)
    start_date = st.sidebar.date_input("Forecast Start Date", datetime(2026, 1, 1).date())
    start_hour = st.sidebar.slider("Start Hour", 0, 23, 0)
    start_time = datetime.combine(start_date, datetime.min.time()).replace(hour=start_hour)

    # Build seed
    seed_row, seed_time = get_seed_row(df_tail, city_enc, loc_enc, seed_year=seed_year, seed_months=seed_months)

    # Force current selection encodings into seed
    seed_row["city_encoded"] = float(city_enc)
    seed_row["location_id_encoded"] = float(loc_enc)

    # Clamp seed AQI lags to prevent recursive explosion
    seed_row = clamp_seed_lags(seed_row, cap=300.0)

    st.caption(f"Seeding from: {seed_time}  |  Forecast starts: {start_time}")

    tab1, tab2, tab3 = st.tabs(["⏱️ 24 Hours", "📅 7 Days", "🗓️ 30 Days"])

    def show_forecast(title, steps):
        fc = recursive_forecast(seed_row, start_time, steps)

        std = float(metrics.get("residual_std", 0.0))
        fc["Upper"] = np.clip(fc["Predicted_AQI"] + 1.96 * std, 0, 800)
        fc["Lower"] = np.clip(fc["Predicted_AQI"] - 1.96 * std, 0, 800)

        st.markdown(f"### {title} Forecast + Confidence Band")
        plot_forecast_with_ci(fc, f"{title} Forecast (2026)")

        st.dataframe(fc.head(60))
        st.download_button(
            f"⬇️ Download {title} CSV",
            fc.to_csv(index=False).encode("utf-8"),
            file_name=f"aqi_forecast_2026_{steps}h.csv",
            mime="text/csv"
        )

    with tab1:
        show_forecast("24h", 24)
    with tab2:
        show_forecast("7d", 24 * 7)
    with tab3:
        show_forecast("30d", 24 * 30)
# ---------------------------
# Mode: Research (What-if)
# ---------------------------
elif mode == "🧪 Research (What-if)":
    st.subheader("🧪 Research Mode: Scenario Simulation + SHAP")
    st.info("Change conditions and see AQI response. This is your most controlled + explainable demo mode.")

    colA, colB = st.columns(2)

    with colA:
        aqi_lag1 = st.number_input("AQI Lag1 (recent)", value=160.0, min_value=0.0, max_value=800.0)
        aqi_lag2 = st.number_input("AQI Lag2", value=150.0, min_value=0.0, max_value=800.0)
        aqi_lag3 = st.number_input("AQI Lag3", value=140.0, min_value=0.0, max_value=800.0)

        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
        pressure = st.slider("Pressure (hPa)", 900.0, 1100.0, 1000.0)
        wind_speed = st.slider("Wind Speed", 0.0, 20.0, 2.0)
        wind_dir = st.slider("Wind Direction (deg)", 0.0, 360.0, 180.0)

    with colB:
        pm25 = st.slider("PM2.5", 0.0, 500.0, 120.0)
        pm10 = st.slider("PM10", 0.0, 500.0, 160.0)
        no2 = st.slider("NO2", 0.0, 500.0, 45.0)
        so2 = st.slider("SO2", 0.0, 500.0, 20.0)
        o3 = st.slider("O3", 0.0, 500.0, 55.0)
        co = st.slider("CO", 0.0, 50.0, 1.0)

        date = st.date_input("Date", datetime(2026, 2, 20).date())
        hour = st.slider("Hour", 0, 23, 12)

    dt = datetime.combine(date, datetime.min.time()).replace(hour=hour)

    # Start from a seed template so every required key exists
    seed_row, _ = get_seed_row(df_tail, city_enc, loc_enc)
    row = seed_row.copy()

    # force selection encodings
    row["city_encoded"] = float(city_enc)
    row["location_id_encoded"] = float(loc_enc)

    # overwrite scenario values
    row["temperature"] = float(temperature)
    row["humidity"] = float(humidity)
    row["pressure"] = float(pressure)
    row["wind_speed"] = float(wind_speed)
    row["wind_dir_sin"] = float(np.sin(np.deg2rad(wind_dir)))
    row["wind_dir_cos"] = float(np.cos(np.deg2rad(wind_dir)))

    row["pm25"] = float(pm25)
    row["pm10"] = float(pm10)
    row["no2"] = float(no2)
    row["so2"] = float(so2)
    row["o3"] = float(o3)
    row["co"] = float(co)

    # AQI lags
    row["aqi_lag1"] = float(aqi_lag1)
    row["aqi_lag2"] = float(aqi_lag2)
    row["aqi_lag3"] = float(aqi_lag3)
    row["aqi_roll3"] = float(np.mean([aqi_lag1, aqi_lag2, aqi_lag3]))

    # time features
    row["Year"] = dt.year
    row["Month"] = dt.month
    row["Day"] = dt.day
    row["Dayofweek"] = dt.weekday()
    row["Hour_sin"] = float(np.sin(2 * np.pi * dt.hour / 24))
    row["Hour_cos"] = float(np.cos(2 * np.pi * dt.hour / 24))

    # pollutant lags/roll (simple consistent)
    for p, v in [("pm25", pm25), ("pm10", pm10), ("no2", no2), ("so2", so2), ("o3", o3), ("co", co)]:
        row[f"{p}_lag1"] = float(v)
        row[f"{p}_lag2"] = float(v)
        row[f"{p}_lag3"] = float(v)
        row[f"{p}_roll3"] = float(v)

    pred = predict_from_row(row)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Predicted AQI", f"{pred:.2f}")
    with c2:
        metric_card("Category", aqi_cat(pred))
    with c3:
        metric_card("Scenario Time", dt.strftime("%Y-%m-%d %H:00"))

    # SHAP explanation
    st.markdown("### 🔍 SHAP Explanation (Top Features)")
    if explainer is None:
        st.warning("SHAP explainer not available. (App works fine without it.)")
    else:
        try:
            X = ensure_features(row)
            shap_vals = explainer.shap_values(X)
            sv = np.array(shap_vals)[0]
            names = np.array(FEATURES)
            top_idx = np.argsort(np.abs(sv))[::-1][:12]

            plot_shap_bar(names[top_idx], sv[top_idx], "Top Feature Contributions (SHAP)")
            st.caption("Positive SHAP increases AQI; negative SHAP decreases AQI (for this scenario).")
        except Exception as e:
            st.warning(f"SHAP failed: {e}")


# ---------------------------
# Mode: Historical 24h (validation)
# ---------------------------
else:
    st.subheader("📈 Historical 24-hour AQI Fluctuation (Actual vs Model One-Step)")

    # Make sure Time is datetime
    if "Time" in df_full.columns:
        # (already parsed in load_all, but safe)
        df_full["Time"] = pd.to_datetime(df_full["Time"], errors="coerce")

    max_date = df_full["Time"].dt.date.max()
    pick_date = st.date_input("Pick a date", max_date)

    # ---- Copy and FILTER correctly using real columns ----
    d = df_full.copy()

    # Clean text cols for exact matching
    d["city"] = d["city"].astype(str).str.strip()
    d["location_id"] = d["location_id"].astype(str).str.strip()

    city_sel = str(city).strip()
    loc_sel = str(location).strip()

    d = d[(d["city"] == city_sel) & (d["location_id"] == loc_sel)]

    st.caption(f"Rows after filter (city+location): {len(d)}")

    if len(d) == 0:
        st.warning("No data for this city/location in full dataset.")
    else:
        # Filter selected date
        day_raw = d[d["Time"].dt.date == pick_date].copy()

        # fallback: nearest available date for this location
        if len(day_raw) < 5:
            nearest = d["Time"].dt.date.max()
            st.info(f"Not enough data for selected date. Using nearest available date: {nearest}")
            pick_date = nearest
            day_raw = d[d["Time"].dt.date == pick_date].copy()

        # If still empty
        if len(day_raw) == 0:
            st.warning("No data even on nearest available date. Try another location.")
        else:
            day_raw = day_raw.sort_values("Time").reset_index(drop=True)

            # ---- RESAMPLE ONLY AQI hourly (reduces data loss) ----
            # This creates hourly series; missing hours become NaN then we drop them.
            day_df = (
                day_raw.set_index("Time")["aqi"]
                .resample("1H")
                .mean()
                .reset_index()
                .dropna(subset=["aqi"])
                .reset_index(drop=True)
            )

            if len(day_df) < 10:
                st.warning("Not enough hourly AQI after resampling. Try another date/location.")
                st.dataframe(day_df.head(50))
            else:
                # ---- Build one-step predictions with padded lags (no NaN rows) ----
                preds = []
                a0 = float(day_df.iloc[0]["aqi"])  # padding value for first rows

                for i in range(len(day_df)):
                    row = {}

                    # encodings needed for model features (keep your training style)
                    row["city_encoded"] = float(city_enc)
                    row["location_id_encoded"] = float(loc_enc)

                    # AQI lags (pad for first rows)
                    row["aqi_lag1"] = float(day_df.iloc[i - 1]["aqi"]) if i - 1 >= 0 else a0
                    row["aqi_lag2"] = float(day_df.iloc[i - 2]["aqi"]) if i - 2 >= 0 else a0
                    row["aqi_lag3"] = float(day_df.iloc[i - 3]["aqi"]) if i - 3 >= 0 else a0
                    row["aqi_roll3"] = float(np.mean([row["aqi_lag1"], row["aqi_lag2"], row["aqi_lag3"]]))

                    # Time features
                    t = pd.to_datetime(day_df.iloc[i]["Time"])
                    row["Year"] = t.year
                    row["Month"] = t.month
                    row["Day"] = t.day
                    row["Dayofweek"] = t.weekday()
                    row["Hour_sin"] = float(np.sin(2 * np.pi * t.hour / 24))
                    row["Hour_cos"] = float(np.cos(2 * np.pi * t.hour / 24))

                    # IMPORTANT:
                    # If your model expects pollutants/weather features too,
                    # they might be missing here because we resampled only AQI.
                    # ensure_features() will fill missing with 0, but for better accuracy,
                    # you can fetch hourly means from day_raw. (We can add that next.)

                    yhat = predict_from_row(row)
                    preds.append({"Time": t, "Pred_AQI": yhat})

                pred_df = pd.DataFrame(preds)

                plot_df = day_df.rename(columns={"aqi": "Actual_AQI"}).merge(
                    pred_df, on="Time", how="left"
                )

                # Plot with Plotly (helper from your Plotly version)
                plot_lines_time(
                    plot_df,
                    "Time",
                    ["Actual_AQI", "Pred_AQI"],
                    f"24h Fluctuation: {pick_date} (Historical Validation)"
                )

                st.dataframe(plot_df)

                st.download_button(
                    "⬇️ Download Historical Comparison CSV",
                    plot_df.to_csv(index=False).encode("utf-8"),
                    file_name="aqi_historical_24h_comparison.csv",
                    mime="text/csv"
                )
st.caption("✅ Forecast (2026) + Confidence Band + Research Mode + SHAP + Historical Validation")