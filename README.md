# EV---Charging-Demand-Forecasting-
 Forecast the demand at EV charging stations based on weather, time, and traffic.
⚡ Electric Vehicle Charging Demand Forecasting
📌 Project Overview
This project forecasts hourly EV charging demand using synthetic but realistic data (weather, time, and traffic).
The goal is to support staffing, pricing, and energy planning decisions at EV charging stations.
🗂 Dataset

Synthetic dataset generated hourly:

**Features:**
temp_c → temperature
precip → precipitation indicator
traffic_idx → traffic intensity
hour, dow, month → time-based features

**Target:**
sessions → EV charging demand (sessions/hour)
📊 Forecast dataset includes forecast_sessions for the next 14 days

**⚙️ Methodology :**

1)**Data Generation & Cleaning** – create realistic hourly demand with seasonality.
2)**Feature Engineering**– weather, time-of-day, day-of-week, traffic.
3)**Modeling** – Random Forest Regressor.
4)**Evaluation** – last 14 days held out for testing.
5)**Forecasting**– next 14 days predicted hourly.
6)**Visualization** – charts, heatmaps, and dashboard.

**📈 Results**

Model Used: Random Forest Regressor
Evaluation (last 14 days):
MAE = 11.51
RMSE = 12.07
MAPE = 29.22%
**Key insights**:

- Evening peaks dominate charging demand.
- Commute hours strongly influence usage.
- Precipitation reduces demand.
- Traffic index + hour-of-day are most influential drivers.
📊 **Dashboard & Reports**

📑 EV Charging Demand Forecasting Report
📊 Tableau-style Dashboard

📌 Future Work

//Use real-world station datasets instead of synthetic.
//Add holiday & event effects.
//Extend to multi-station forecasting.
//Deploy as an interactive dashboard (Tableau/Power BI).
sessions → EV charging demand (sessions/hour)

📊 Forecast dataset includes forecast_sessions for the next 14 days.
