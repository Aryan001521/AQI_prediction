# ============================
# AQI Research & Forecast Platform (FINAL FIXED)
# ✅ Fixes:
# 1) City dropdown empty / no effect  -> if model has NO city onehot cols, we disable city (expected)
# 2) Location change not affecting prediction -> now we SET the correct loc_ onehot column in input row
# 3) Historical 24h "aqi column nahi mila" + buggy break/c loop -> fixed robust AQI column detection
# 4) CSV load time -> uses usecols + caching; can also use parquet later
# ============================

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
def load_artifacts():
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR / "models"

    model = joblib.load(MODEL_DIR / "best_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    FEATURES = joblib.load(MODEL_DIR / "feature_list.pkl")

    # metrics
    metrics = {"rmse": 0.0, "r2": 0.0, "mae": 0.0, "residual_std": 0.0}
    mpath = MODEL_DIR / "metrics.json"
    if mpath.exists():
        try:
            with open(mpath, "r") as f:
                metrics.update(json.load(f))
        except Exception:
            pass

    # Optional SHAP
    explainer = None
    try:
        import shap  # noqa
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = None

    return BASE_DIR, MODEL_DIR, model, scaler, FEATURES, metrics, explainer


BASE_DIR, MODEL_DIR, model, scaler, FEATURES, metrics, explainer = load_artifacts()

# ---------------------------
# OneHot columns from feature_list.pkl
# ---------------------------
CITY_OHE_COLS = [c for c in FEATURES if str(c).lower().startswith("city_")]
LOC_OHE_COLS = [c for c in FEATURES if str(c).lower().startswith("loc_")]

def _label_from_ohe(col: str) -> str:
    return col.split("_", 1)[1] if "_" in col else col


# ---------------------------
# Data loaders (cached)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_tail():
    df_tail = pd.read_csv(MODEL_DIR / "recent_tail.csv")
    if "Time" in df_tail.columns:
        df_tail["Time"] = pd.to_datetime(df_tail["Time"], errors="coerce")
        df_tail = df_tail.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return df_tail


@st.cache_data(show_spinner=False)
def load_full(needed_cols: list[str]):
    """
    Loads full data with ONLY required cols for speed.
    If delhi_2000_2025_extended.csv not found -> fallback to tail.
    """
    FULL_DATA_PATH = BASE_DIR / "delhi_2000_2025_extended.csv"
    if not FULL_DATA_PATH.exists():
        return load_tail()

    # try to read only needed columns
    try:
        df_full = pd.read_csv(FULL_DATA_PATH, usecols=needed_cols)
    except Exception:
        # fallback read all
        df_full = pd.read_csv(FULL_DATA_PATH)

    if "Time" in df_full.columns:
        df_full["Time"] = pd.to_datetime(df_full["Time"], errors="coerce")
        df_full = df_full.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return df_full


df_tail = load_tail()

# minimal columns for historical (fast load)
hist_needed_cols = ["Time", "aqi", "AQI", "pm25", "pm10", "no2", "so2", "o3", "co", "wind_speed"]
df_full = load_full(hist_needed_cols)


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
    X = pd.DataFrame([row_dict])
    for f in FEATURES:
        if f not in X.columns:
            X[f] = 0.0
    # keep exact order
    return X[FEATURES].astype(float, errors="ignore").fillna(0.0)


def predict_from_row(row_dict: dict) -> float:
    X = ensure_features(row_dict)
    Xs = scaler.transform(X)
    return float(model.predict(Xs)[0])


def apply_onehot(row: dict, city_col: str | None, loc_col: str | None):
    # zero all onehot cols
    for c in CITY_OHE_COLS:
        row[c] = 0.0
    for c in LOC_OHE_COLS:
        row[c] = 0.0

    # set selected
    if city_col is not None and city_col in FEATURES:
        row[city_col] = 1.0
    if loc_col is not None and loc_col in FEATURES:
        row[loc_col] = 1.0
    return row


def clamp_seed_lags(seed_row: dict, cap=300.0):
    for k in ["aqi_lag1", "aqi_lag2", "aqi_lag3"]:
        if k in seed_row:
            seed_row[k] = float(np.clip(seed_row[k], 0, cap))
    seed_row["aqi_roll3"] = float(np.mean([
        seed_row.get("aqi_lag1", seed_row.get("aqi", 0.0)),
        seed_row.get("aqi_lag2", seed_row.get("aqi", 0.0)),
        seed_row.get("aqi_lag3", seed_row.get("aqi", 0.0)),
    ]))
    return seed_row


def get_seed_row_simple(df: pd.DataFrame, seed_year: int | None = None, seed_months: list[int] | None = None):
    """
    Simple: pick last row (optionally year/month filtered). Works even if df has NO city/location cols.
    """
    d = df.copy()
    if "Time" in d.columns:
        d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
        d = d.dropna(subset=["Time"]).sort_values("Time")
        if seed_year is not None:
            d = d[d["Time"].dt.year == seed_year]
        if seed_months is not None:
            d = d[d["Time"].dt.month.isin(seed_months)]
        if len(d) == 0:
            d = df.copy()
            d["Time"] = pd.to_datetime(d["Time"], errors="coerce")
            d = d.dropna(subset=["Time"]).sort_values("Time")

        last_row = d.iloc[-1].to_dict()
        last_time = pd.to_datetime(d.iloc[-1]["Time"])
        return last_row, last_time

    # if no Time at all
    last_row = d.iloc[-1].to_dict()
    return last_row, datetime(2025, 12, 31, 23, 0, 0)


def recursive_forecast(seed_row: dict, start_time: datetime, steps: int, city_col: str | None, loc_col: str | None) -> pd.DataFrame:
    cur = seed_row.copy()
    preds = []
    cur_time = start_time

    # apply correct onehot every time (important)
    cur = apply_onehot(cur, city_col, loc_col)

    # defaults
    cur.setdefault("wind_dir_sin", 0.0)
    cur.setdefault("wind_dir_cos", 1.0)
    cur.setdefault("wind_speed", float(cur.get("wind_speed", 0.0)))

    # AQI lag defaults
    base_aqi = float(cur.get("aqi", cur.get("aqi_lag1", 0.0)))
    for lag in [1, 2, 3]:
        cur.setdefault(f"aqi_lag{lag}", base_aqi)
    cur.setdefault("aqi_roll3", float(np.mean([cur["aqi_lag1"], cur["aqi_lag2"], cur["aqi_lag3"]])))

    # pollutants
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

        # always enforce onehot (safe)
        cur = apply_onehot(cur, city_col, loc_col)

        yhat = predict_from_row(cur)
        yhat = float(np.clip(yhat, 0, 500))
        preds.append({"Time": cur_time, "Predicted_AQI": yhat})

        # update AQI lags
        cur["aqi_lag3"] = cur["aqi_lag2"]
        cur["aqi_lag2"] = cur["aqi_lag1"]
        cur["aqi_lag1"] = yhat
        cur["aqi_roll3"] = float(np.mean([cur["aqi_lag1"], cur["aqi_lag2"], cur["aqi_lag3"]]))

        # mild decay for pollutants (demo stability)
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

    fig.add_trace(go.Scatter(
        x=fc["Time"], y=fc["Upper"],
        mode="lines", line=dict(width=0),
        name="Upper", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=fc["Time"], y=fc["Lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty", name="95% CI"
    ))
    fig.add_trace(go.Scatter(
        x=fc["Time"], y=fc["Predicted_AQI"],
        mode="lines", name="Prediction"
    ))

    fig.update_layout(
        title=title, height=380,
        yaxis_title="AQI", xaxis_title="Time",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_lines_time(df_: pd.DataFrame, x_col: str, y_cols: list[str], title: str):
    fig = go.Figure()
    for c in y_cols:
        fig.add_trace(go.Scatter(x=df_[x_col], y=df_[c], mode="lines", name=c))
    fig.update_layout(
        title=title, height=380,
        yaxis_title="AQI", xaxis_title="Time",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_shap_bar(features: np.ndarray, shap_vals: np.ndarray, title: str):
    shap_df = pd.DataFrame({"feature": features, "shap_value": shap_vals})
    shap_df = shap_df.sort_values("shap_value", key=lambda s: np.abs(s), ascending=False).head(12)

    fig = go.Figure(go.Bar(x=shap_df["feature"], y=shap_df["shap_value"]))
    fig.update_layout(
        title=title, height=380,
        yaxis_title="SHAP value (impact)", xaxis_title="Feature",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Mode", ["🔮 Forecast", "🧪 Research (What-if)", "📈 Historical 24h"])

# City selection (only if model has city onehot cols)
if len(CITY_OHE_COLS) > 0:
    city_labels = [_label_from_ohe(c) for c in CITY_OHE_COLS]
    city_idx = st.sidebar.selectbox("City", list(range(len(city_labels))), format_func=lambda i: city_labels[i])
    selected_city_col = CITY_OHE_COLS[city_idx]
else:
    st.sidebar.selectbox("City", ["Delhi (No City OneHot in model)"], index=0, disabled=True)
    selected_city_col = None

# Location selection (from model onehot cols)
if len(LOC_OHE_COLS) > 0:
    loc_labels = [_label_from_ohe(c) for c in LOC_OHE_COLS]
    loc_idx = st.sidebar.selectbox("Location", list(range(len(loc_labels))), format_func=lambda i: loc_labels[i])
    selected_loc_col = LOC_OHE_COLS[loc_idx]
else:
    st.sidebar.selectbox("Location", ["N/A (No loc_ OneHot in model)"], index=0, disabled=True)
    selected_loc_col = None


# ---------------------------
# Header + Metrics
# ---------------------------
st.title("🌫️ AQI Research & Forecast Platform")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Model RMSE", f"{metrics.get('rmse', 0.0):.2f}")
with c2:
    metric_card("Model MAE", f"{metrics.get('mae', 0.0):.2f}")
with c3:
    metric_card("Model R²", f"{metrics.get('r2', 0.0):.2f}")
with c4:
    metric_card("Selected", _label_from_ohe(selected_loc_col) if selected_loc_col else "N/A")


# Debug OneHot matching
with st.expander("🔎 Debug: OneHot matching", expanded=True):
    st.write("Matched city col:", selected_city_col if selected_city_col else "None")
    st.write("Matched loc col:", selected_loc_col if selected_loc_col else "None")
    st.write("Total city onehot cols:", len(CITY_OHE_COLS))
    st.write("Total loc onehot cols:", len(LOC_OHE_COLS))


# ---------------------------
# Mode: Forecast
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
        seed_months = st.sidebar.multiselect("Seed Months", list(range(1, 13)), default=[8, 9])
        if len(seed_months) == 0:
            seed_months = None

    start_date = st.sidebar.date_input("Forecast Start Date", datetime(2026, 1, 1).date())
    start_hour = st.sidebar.slider("Start Hour", 0, 23, 0)
    start_time = datetime.combine(start_date, datetime.min.time()).replace(hour=start_hour)

    # Seed row from tail
    seed_row, seed_time = get_seed_row_simple(df_tail, seed_year=seed_year, seed_months=seed_months)
    seed_row = clamp_seed_lags(seed_row, cap=300.0)
    seed_row = apply_onehot(seed_row, selected_city_col, selected_loc_col)

    st.caption(f"Seeding from: {seed_time}  |  Forecast starts: {start_time}")

    tab1, tab2, tab3 = st.tabs(["⏱️ 24 Hours", "📅 7 Days", "🗓️ 30 Days"])

    def show_forecast(title, steps):
        if selected_loc_col is None and len(LOC_OHE_COLS) > 0:
            st.warning("Location one-hot not set. Select a location.")
            return

        fc = recursive_forecast(seed_row, start_time, steps, selected_city_col, selected_loc_col)

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
    st.info("Change conditions and see AQI response. Now location one-hot is applied correctly.")

    colA, colB = st.columns(2)

    with colA:
        aqi_lag1 = st.number_input("AQI Lag1 (recent)", value=160.0, min_value=0.0, max_value=800.0)
        aqi_lag2 = st.number_input("AQI Lag2", value=150.0, min_value=0.0, max_value=800.0)
        aqi_lag3 = st.number_input("AQI Lag3", value=140.0, min_value=0.0, max_value=800.0)

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

    # Start from tail last row (stable base)
    seed_row, _ = get_seed_row_simple(df_tail)
    row = seed_row.copy()

    # scenario overwrite
    row["wind_speed"] = float(wind_speed)
    row["wind_dir_sin"] = float(np.sin(np.deg2rad(wind_dir)))
    row["wind_dir_cos"] = float(np.cos(np.deg2rad(wind_dir)))

    row["pm25"] = float(pm25)
    row["pm10"] = float(pm10)
    row["no2"] = float(no2)
    row["so2"] = float(so2)
    row["o3"] = float(o3)
    row["co"] = float(co)

    row["aqi_lag1"] = float(aqi_lag1)
    row["aqi_lag2"] = float(aqi_lag2)
    row["aqi_lag3"] = float(aqi_lag3)
    row["aqi_roll3"] = float(np.mean([aqi_lag1, aqi_lag2, aqi_lag3]))

    # time
    row["Year"] = dt.year
    row["Month"] = dt.month
    row["Day"] = dt.day
    row["Dayofweek"] = dt.weekday()
    row["Hour_sin"] = float(np.sin(2 * np.pi * dt.hour / 24))
    row["Hour_cos"] = float(np.cos(2 * np.pi * dt.hour / 24))

    # pollutant lags/roll
    for p, v in [("pm25", pm25), ("pm10", pm10), ("no2", no2), ("so2", so2), ("o3", o3), ("co", co)]:
        row[f"{p}_lag1"] = float(v)
        row[f"{p}_lag2"] = float(v)
        row[f"{p}_lag3"] = float(v)
        row[f"{p}_roll3"] = float(v)

    # ✅ apply onehot (THIS was missing earlier)
    row = apply_onehot(row, selected_city_col, selected_loc_col)

    pred = predict_from_row(row)

    cA, cB, cC = st.columns(3)
    with cA:
        metric_card("Predicted AQI", f"{pred:.2f}")
    with cB:
        metric_card("Category", aqi_cat(pred))
    with cC:
        metric_card("Scenario Time", dt.strftime("%Y-%m-%d %H:00"))

    st.markdown("### 🔍 SHAP Explanation (Top Features)")
    if explainer is None:
        st.warning("SHAP explainer not available. (App works fine without it.)")
    else:
        try:
            X_ = ensure_features(row)
            shap_vals = explainer.shap_values(X_)
            sv = np.array(shap_vals)[0]
            names = np.array(FEATURES)
            top_idx = np.argsort(np.abs(sv))[::-1][:12]
            plot_shap_bar(names[top_idx], sv[top_idx], "Top Feature Contributions (SHAP)")
        except Exception as e:
            st.warning(f"SHAP failed: {e}")

    with st.expander("🧾 Debug: first 20 features sent to model", expanded=False):
        Xdbg = ensure_features(row)
        st.write(Xdbg.iloc[0, :20])


# ---------------------------
# Mode: Historical 24h
# ---------------------------
else:
    st.subheader("📈 Historical 24-hour AQI Fluctuation (Actual vs Model One-Step)")

    if "Time" not in df_full.columns:
        st.warning("Full dataset missing 'Time' column. Historical mode can't run.")
    else:
        df_full["Time"] = pd.to_datetime(df_full["Time"], errors="coerce")
        df_full = df_full.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

        max_date = df_full["Time"].dt.date.max()
        pick_date = st.date_input("Pick a date", max_date)

        # robust AQI column detection
        aqi_col = None
        for c in df_full.columns:
            if str(c).strip().lower() == "aqi":
                aqi_col = c
                break

        if aqi_col is None:
            st.warning("Dataset me 'aqi' column nahi mila.")
        else:
            day_raw = df_full[df_full["Time"].dt.date == pick_date].copy()

            if len(day_raw) < 5:
                nearest = df_full["Time"].dt.date.max()
                st.info(f"Not enough data for selected date. Using nearest available date: {nearest}")
                pick_date = nearest
                day_raw = df_full[df_full["Time"].dt.date == pick_date].copy()

            if len(day_raw) == 0:
                st.warning("No data for selected/nearest date.")
            else:
                day_raw = day_raw.sort_values("Time").reset_index(drop=True)

                # hourly actual AQI
                day_df = (
                    day_raw.set_index("Time")[aqi_col]
                    .resample("1H").mean()
                    .reset_index()
                    .rename(columns={aqi_col: "Actual_AQI"})
                    .dropna(subset=["Actual_AQI"])
                    .reset_index(drop=True)
                )

                if len(day_df) < 10:
                    st.warning("Not enough hourly AQI after resampling. Try another date.")
                    st.dataframe(day_df.head(50))
                else:
                    preds = []
                    a0 = float(day_df.iloc[0]["Actual_AQI"])

                    # If pollutants exist in day_raw, we use nearest hour value
                    # else they will be 0 by ensure_features.
                    pols = ["pm25", "pm10", "no2", "so2", "o3", "co", "wind_speed"]
                    day_raw_hour = day_raw.copy()
                    day_raw_hour["Time_hr"] = day_raw_hour["Time"].dt.floor("H")
                    agg_map = {p: "mean" for p in pols if p in day_raw_hour.columns}
                    if len(agg_map) > 0:
                        aux = day_raw_hour.groupby("Time_hr", as_index=False).agg(agg_map)
                        aux = aux.rename(columns={"Time_hr": "Time"})
                        day_df = day_df.merge(aux, on="Time", how="left")

                    for i in range(len(day_df)):
                        row = {}

                        # AQI lags
                        row["aqi_lag1"] = float(day_df.iloc[i - 1]["Actual_AQI"]) if i - 1 >= 0 else a0
                        row["aqi_lag2"] = float(day_df.iloc[i - 2]["Actual_AQI"]) if i - 2 >= 0 else a0
                        row["aqi_lag3"] = float(day_df.iloc[i - 3]["Actual_AQI"]) if i - 3 >= 0 else a0
                        row["aqi_roll3"] = float(np.mean([row["aqi_lag1"], row["aqi_lag2"], row["aqi_lag3"]]))

                        t = pd.to_datetime(day_df.iloc[i]["Time"])
                        row["Year"] = t.year
                        row["Month"] = t.month
                        row["Day"] = t.day
                        row["Dayofweek"] = t.weekday()
                        row["Hour_sin"] = float(np.sin(2 * np.pi * t.hour / 24))
                        row["Hour_cos"] = float(np.cos(2 * np.pi * t.hour / 24))

                        # pollutants (if present)
                        for p in ["pm25", "pm10", "no2", "so2", "o3", "co"]:
                            if p in day_df.columns and pd.notna(day_df.iloc[i].get(p, np.nan)):
                                v = float(day_df.iloc[i][p])
                                row[p] = v
                                row[f"{p}_lag1"] = v
                                row[f"{p}_lag2"] = v
                                row[f"{p}_lag3"] = v
                                row[f"{p}_roll3"] = v

                        if "wind_speed" in day_df.columns and pd.notna(day_df.iloc[i].get("wind_speed", np.nan)):
                            row["wind_speed"] = float(day_df.iloc[i]["wind_speed"])

                        # wind dir defaults
                        row.setdefault("wind_dir_sin", 0.0)
                        row.setdefault("wind_dir_cos", 1.0)

                        # ✅ apply onehot
                        row = apply_onehot(row, selected_city_col, selected_loc_col)

                        yhat = predict_from_row(row)
                        preds.append({"Time": t, "Pred_AQI": float(yhat)})

                    pred_df = pd.DataFrame(preds)
                    plot_df = day_df.merge(pred_df, on="Time", how="left")

                    plot_lines_time(plot_df, "Time", ["Actual_AQI", "Pred_AQI"], f"24h Fluctuation: {pick_date}")
                    st.dataframe(plot_df)

                    st.download_button(
                        "⬇️ Download Historical Comparison CSV",
                        plot_df.to_csv(index=False).encode("utf-8"),
                        file_name="aqi_historical_24h_comparison.csv",
                        mime="text/csv"
                    )

st.caption("✅ Forecast (2026) + Confidence Band + Research Mode + SHAP + Historical Validation")

with st.expander("📌 Debug: feature_list first 60", expanded=False):
    st.write(FEATURES[:60])
