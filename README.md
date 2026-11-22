# 🌱 Smart Greenhouse AIoT Dashboard

A production-ready Streamlit monitoring dashboard and microclimate forecasting system for smart greenhouse management. Connected to an ESP32 microcontroller (simulated in Wokwi) via ThingSpeak cloud telemetry and powered by two dedicated machine learning models.

---

## 📌 Architecture & Features

```
+--------------------------+       WiFi / HTTP        +--------------------------+
|  ESP32 Microcontroller   | -----------------------> |   ThingSpeak IoT Cloud   |
|  - Wokwi Simulation      |       (Every 20 s)       |  - Feeds API (JSON)      |
|  - DHT22 (Reads every 5s)|                          |  - Field 1: Temp (°C)    |
|  - Field 3: Alert Level  |                          |  - Field 2: Humidity (%) |
+--------------------------+                          +--------------------------+
                                                                    │
                                                                    │ REST GET (feeds.json)
                                                                    ▼
+--------------------------------------------------------------------------------+
|                           Streamlit AIoT Application                           |
|  ┌─────────────────────────────────────┐   ┌────────────────────────────────┐  |
|  │        1-Hour Forecast Model        │   │ Isolation Forest Anomaly Model │  |
|  │  - Instant Model: T, RH, Hour Sin/Cos │ - Pointwise Scoring [T, RH, VPD] │  |
|  │  - Lag Model: 10m, 30m, 60m Dynamics│   │ - Unusual Feature Breakdown    │  |
|  └─────────────────────────────────────┘   └────────────────────────────────┘  |
+--------------------------------------------------------------------------------+
```

### Key Highlights
1. **Live Microclimate Dashboard (Tab 1)**:
   - **Unified Top Status Banner**: Combines device alert level, 1-hour projected forecast alert, and isolation forest anomaly status into a single clear indicator (Green = All Normal, Orange = Warning, Red = Critical/Anomaly) with plain-language diagnostics.
   - **Real-Time KPIs**: Current Temperature, Relative Humidity, Vapor Pressure Deficit (VPD), Device Alert Level, +1h Forecasted Temperature with delta, Forecast Alert Level, and Anomaly Score.
   - **Interactive Plotly Visualizations**:
     - Temperature chart with 30 °C / 35 °C high limits, 5 °C / 2 °C low limits, and a distinct 1-hour forecast projection marker connected via dashed line.
     - Relative Humidity chart with 40 % and 90 % safety limits.
   - **Isolation Forest Anomaly Card**: Visual score gauge vs. threshold (0.6315), severity level (MEDIUM / HIGH), and dynamic detection of the 2 most unusual features.
   - **Stale Data Protection**: Automatic warning badge ("⚠️ Device offline / simulation not running") if latest feed reading is older than 2 minutes.
   - **Dynamic Auto-Refresh**: Uses modern `st.fragment(run_every=...)` for seamless non-blocking background refreshes without full-page reloads.
   - **CSV Export**: Instant download of recent readings.

2. **Offline Manual Simulator (Tab 2)**:
   - Complete interactive sandbox to demo the machine learning models without needing an active ThingSpeak channel or Wokwi simulation.
   - One-click presets: *Normal day (24 °C / 65 %)*, *Heat wave (36 °C / 30 %)*, *Frost (1 °C / 80 %)*, and *Humid (28 °C / 95 %)*.
   - Continuous sliders for Temperature (0–45 °C), Humidity (10–100 %), and Hour of Day (0–23.5 h).

3. **Models & About Specifications (Tab 3)**:
   - Full documentation of the 1-hour forecast model (Ridge/Linear instant and lag models, 10-minute grid resampling, MAE/RMSE/R² metrics).
   - Anomaly model evaluation details (precision, recall, F1, event recall across synthetic fault scenarios like spikes, drops, drifts, and shifts).
   - Honest methodological notes clarifying that metrics originate from controlled synthetic fault-injection stress testing rather than real-world ground truth.

---

## 🚀 Setup & Execution

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure ThingSpeak Credentials
Credentials can be provided in two ways (never hardcoded):
1. **Via `.streamlit/secrets.toml`** (recommended):
   ```toml
   THINGSPEAK_CHANNEL_ID = "3580105"
   THINGSPEAK_READ_KEY = "YOUR_READ_KEY_HERE"
   ```
2. **Via Environment Variables**:
   ```bash
   # Windows PowerShell
   $env:THINGSPEAK_CHANNEL_ID="3580105"
   $env:THINGSPEAK_READ_KEY="YOUR_READ_KEY_HERE"
   ```
3. **Directly in the App Sidebar**: You can override or enter Channel ID and Read Key at any time inside the app sidebar.

### 5. Launch the Dashboard
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
├── .streamlit/
│   └── secrets.toml                     # ThingSpeak credentials (optional local configuration)
├── models/
│   ├── greenhouse_forecast_bundle.joblib # 1-hour temperature forecast bundle
│   ├── climate_inference_bundle.joblib  # Isolation Forest anomaly detection bundle
│   ├── greenhouse_predict.py            # Forecast model prediction logic
│   ├── predict.py                       # Forwarding module for forecast logic
│   └── model_metadata_and_metrics.json  # Model evaluation metrics and warnings
├── anomaly.py                           # Isolation Forest scoring & VPD calculation
├── thingspeak.py                        # ThingSpeak REST client & stale data check
├── app.py                              # Main Streamlit dashboard application
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Documentation and run instructions
```

---

## 🛡️ Machine Learning Models

### 1. Temperature Forecast Model
- **Target**: Forecast greenhouse temperature 1 hour ahead (+60 min).
- **Instant Model**: Activated when continuous history is < 70 min. Uses `T`, `RH`, and sinusoidal diurnal encoding `hour_sin` / `hour_cos`.
- **Lag Model**: Automatically engaged when >= 70 min of continuous resampled history is available. Incorporates lag features (`T_lag1`, `T_lag3`, `T_lag6`, `RH_lag3`, `RH_lag6`, `dT_30`, `dT_60`).
- **Alert Levels**:
  - `Critical (Level 2)`: T <= 2.0 °C or T >= 35.0 °C
  - `Warning (Level 1)`: T <= 5.0 °C or T >= 30.0 °C
  - `Normal (Level 0)`: 5.0 °C < T < 30.0 °C

### 2. Climate Anomaly Detector
- **Algorithm**: Point-wise Isolation Forest trained on standardized features: `[Temperature, Humidity, VPD]`.
- **VPD Formula**: $\text{VPD} = 0.6108 \times \exp\left(\frac{17.27 \times T}{T + 237.3}\right) \times \left(1 - \frac{\text{RH}}{100}\right)$ kPa
- **Decision Boundary**: Anomaly if anomaly score > 0.6315.
- **Severity**: `HIGH` if score > 0.6315, else `MEDIUM`.
