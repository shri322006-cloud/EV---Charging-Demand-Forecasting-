"""EV Charging Demand Forecasting (Synthetic Implementation)
Generates synthetic EV charging session data, trains SARIMAX model with weather/time features,
and forecasts next 7 days hourly demand. Saves outputs (CSV, PNG, PDF).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# Reproducibility
rng = np.random.default_rng(42)

# -----------------------------
# 1) Generate synthetic data
# -----------------------------
start = pd.Timestamp("2025-05-01 00:00:00")
end = pd.Timestamp("2025-07-31 23:00:00")
dt_index = pd.date_range(start, end, freq="H")

n_hours = len(dt_index)
n_stations = 5
stations = [f"STN_{i+1}" for i in range(n_stations)]

# Weather features (daily)
days = pd.date_range(start.normalize(), end.normalize(), freq="D")
n_days = len(days)
day_of_year = days.day_of_year.values
temp_daily = 30 + 6*np.sin(2*np.pi*(day_of_year/365.0)) + rng.normal(0, 1.5, size=n_days)
rain_prob = 0.3 + 0.2*np.sin(2*np.pi*((day_of_year+60)/365.0))
is_rain_day = rng.random(n_days) < np.clip(rain_prob, 0, 0.7)
rain_mm_daily = np.where(is_rain_day, rng.gamma(shape=2.0, scale=5.0, size=n_days), 0.0)

weather_daily = pd.DataFrame({
    "date": days,
    "temp_c": temp_daily,
    "rain_mm": rain_mm_daily
})

# Map to hourly
weather_hourly = (
    weather_daily
    .reindex(weather_daily.index.repeat(24))
    .reset_index(drop=True)
)
weather_hourly["timestamp"] = pd.date_range(days.min(), days.max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq="H")
hour = weather_hourly["timestamp"].dt.hour
weather_hourly["temp_c"] = weather_hourly["temp_c"] + (hour - 12) * 0.1 + rng.normal(0, 0.3, size=len(weather_hourly))
weather_hourly["rain_mm"] = weather_hourly["rain_mm"] / 24.0
weather_hourly = weather_hourly.loc[weather_hourly["timestamp"].between(start, end)].reset_index(drop=True)

# Demand per station per hour
records = []
for st in stations:
    base = rng.uniform(8, 12)
    for ts, temp, rain in zip(dt_index, weather_hourly["temp_c"].values, weather_hourly["rain_mm"].values):
        hod = ts.hour
        dow = ts.dayofweek
        hod_effect = (
            1.4 * np.exp(-((hod-9)/3.0)**2) +
            1.6 * np.exp(-((hod-18)/3.0)**2)
        )
        weekend_effect = 1.15 if dow >= 5 else 1.0
        temp_effect = 1.0 - 0.005 * (temp - 28)**2 / 10.0
        rain_effect = 1.0 - 0.15 * (rain > 0)
        st_noise = rng.normal(0, 0.6)
        demand = base * (1 + hod_effect) * weekend_effect * temp_effect * rain_effect
        demand = demand + rng.normal(0, 1.5) + st_noise
        demand = max(0.0, demand)
        records.append((ts, st, demand))

df = pd.DataFrame(records, columns=["timestamp", "station_id", "kwh"])
df = df.merge(weather_hourly[["timestamp","temp_c","rain_mm"]], on="timestamp", how="left")

# Aggregate total demand
hourly_total = (
    df.groupby("timestamp", as_index=False)
      .agg(total_kwh=("kwh", "sum"),
           temp_c=("temp_c", "mean"),
           rain_mm=("rain_mm", "sum"))
)
hourly_total["hour"] = hourly_total["timestamp"].dt.hour
hourly_total["dow"] = hourly_total["timestamp"].dt.dayofweek
hourly_total["is_weekend"] = (hourly_total["dow"] >= 5).astype(int)

# -----------------------------
# 2) Train/Test split
# -----------------------------
val_horizon = 24*7
train_df = hourly_total.iloc[:-val_horizon].copy()
val_df = hourly_total.iloc[-val_horizon:].copy()
exog_cols = ["temp_c", "rain_mm", "hour", "is_weekend"]
X_train = train_df[exog_cols]
X_val = val_df[exog_cols]

# -----------------------------
# 3) SARIMAX model
# -----------------------------
model = SARIMAX(train_df["total_kwh"], order=(2,0,2), exog=X_train, enforce_stationarity=False, enforce_invertibility=False)
res = model.fit(disp=False)
val_pred = res.get_forecast(steps=len(X_val), exog=X_val).predicted_mean

# -----------------------------
# 4) Refit on full data and forecast next 7 days
# -----------------------------
full_model = SARIMAX(hourly_total["total_kwh"], order=(2,0,2), exog=hourly_total[exog_cols], enforce_stationarity=False, enforce_invertibility=False)
full_res = full_model.fit(disp=False)

future_index = pd.date_range(hourly_total["timestamp"].iloc[-1] + pd.Timedelta(hours=1), periods=24*7, freq="H")
last_day = weather_hourly.iloc[-24:].copy().reset_index(drop=True)
future_weather = []
for i in range(len(future_index)):
    base_row = last_day.iloc[i % 24].copy()
    base_row["timestamp"] = future_index[i]
    base_row["temp_c"] = base_row["temp_c"] + rng.normal(0, 0.5)
    base_row["rain_mm"] = max(0.0, base_row["rain_mm"] + rng.normal(0, 0.05))
    future_weather.append(base_row)
future_weather = pd.DataFrame(future_weather)

future_exog = pd.DataFrame({
    "timestamp": future_index,
    "temp_c": future_weather["temp_c"].values,
    "rain_mm": future_weather["rain_mm"].values,
})
future_exog["hour"] = future_exog["timestamp"].dt.hour
future_exog["dow"] = future_exog["timestamp"].dt.dayofweek
future_exog["is_weekend"] = (future_exog["dow"] >= 5).astype(int)

future_pred = full_res.get_forecast(steps=len(future_index), exog=future_exog[exog_cols])
forecast_mean = future_pred.predicted_mean
forecast_ci = future_pred.conf_int(alpha=0.2)

forecast_df = pd.DataFrame({
    "timestamp": future_index,
    "forecast_kwh": forecast_mean.values,
    "lower_kwh": forecast_ci.iloc[:,0].values,
    "upper_kwh": forecast_ci.iloc[:,1].values
})

forecast_df.to_csv("ev_forecast_next_7_days_hourly.csv", index=False)
hourly_total.to_csv("ev_historical_hourly_kwh.csv", index=False)

print("Forecast saved to ev_forecast_next_7_days_hourly.csv")
print("Historical data saved to ev_historical_hourly_kwh.csv")
