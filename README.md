# Smart Greenhouse AIoT Dashboard

A production-ready Streamlit monitoring dashboard and microclimate forecasting system for smart greenhouse management. Connected to an ESP32 microcontroller (simulated in Wokwi) via ThingSpeak cloud telemetry and powered by two dedicated machine learning models.

---

## Architecture & Features

```
+--------------------------+       WiFi / HTTPS       +--------------------------+
|  ESP32 Microcontroller   | -----------------------> |   ThingSpeak IoT Cloud   |
|  - Wokwi Simulation      |       (Every 20 s)       |  - Feeds API (JSON)      |
|  - DHT22 (Reads every 5s)|                          |  - Field 1: Temp (°C)    |
|  - Field 3: Alert Level  |                          |  - Field 2: Humidity (%) |
+--------------------------+                          +--------------------------+
                                                                    |
                                                                    | HTTPS GET (feeds.json)
                                                                    v
+--------------------------------------------------------------------------------+
|                           Streamlit AIoT Application                           |
|  +-------------------------------------+   +--------------------------------+  |
|  |        1-Hour Forecast Model        |   | Isolation Forest Anomaly Model |  |
|  |  - Instant Model: T, RH, Hour Sin/Cos | - Pointwise Scoring [T, RH, VPD] |  |
|  |  - Lag Model: 10m, 30m, 60m Dynamics|   | - Unusual Feature Breakdown    |  |
|  +-------------------------------------+   +--------------------------------+  |
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
   - **Stale Data Protection**: Automatic indicator if latest feed reading exceeds the configured threshold (5 minutes), showing last known readings labeled as delayed/stale.
   - **Dynamic Auto-Refresh**: Uses modern `st.fragment(run_every=20)` for non-blocking background refreshes with a header Refresh button.
   - **CSV Export**: Instant download of recent readings.

2. **Offline Manual Simulator (Tab 2)**:
   - Complete interactive sandbox to demo the machine learning models without needing an active ThingSpeak channel or Wokwi simulation.
   - One-click presets: *Normal day (24 °C / 65 %)*, *Heat wave (36 °C / 30 %)*, *Frost hazard (1 °C / 80 %)*, and *High humidity (28 °C / 95 %)*.
   - Continuous sliders for Temperature (0-45 °C), Humidity (10-100 %), and Hour of Day (0-23.5 h).

3. **Models & About Specifications (Tab 3)**:
   - Full documentation of the 1-hour forecast model (Ridge/Linear instant and lag models, 10-minute grid resampling, MAE/RMSE/R² metrics).
   - Anomaly model evaluation details (precision, recall, F1, event recall across synthetic fault scenarios like spikes, drops, drifts, and shifts).
   - Methodological notes clarifying that metrics originate from controlled synthetic fault-injection stress testing rather than real-world ground truth.

---

## Setup & Execution

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

### 4. Configure Credentials
Credentials must be provided via `.streamlit/secrets.toml` or environment variables (never hardcoded in source code or UI):

1. **Via `.streamlit/secrets.toml`** (recommended):
   ```toml
   # ThingSpeak Telemetry
   THINGSPEAK_CHANNEL_ID = "YOUR_CHANNEL_ID"
   THINGSPEAK_READ_KEY = "YOUR_READ_KEY"

   # Telegram Risk Notifications (Optional)
   TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
   TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
   ```
   An example template is provided in `.streamlit/secrets.toml.example`.

2. **Via Environment Variables**:
   ```bash
   # Windows PowerShell
   $env:THINGSPEAK_CHANNEL_ID="YOUR_CHANNEL_ID"
   $env:THINGSPEAK_READ_KEY="YOUR_READ_KEY"
   $env:TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
   $env:TELEGRAM_CHAT_ID="YOUR_CHAT_ID"
   ```

### 5. Launch the Dashboard
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## Security

This application implements comprehensive defense-in-depth security principles suitable for cloud deployment:

### 1. Device Authentication
- The ESP32 edge microcontroller authenticates to ThingSpeak using channel write API keys over encrypted WiFi.
- The Streamlit dashboard authenticates against private channels using API read keys.
- Write and Read keys are segregated: the dashboard holds strictly read-only capabilities, preventing unauthorized state modification on edge nodes or channels.

### 2. Secure Protocols
- Telemetry ingestion and external alerting strictly require TLS/HTTPS (`https://api.thingspeak.com` and `https://api.telegram.org`). Plain HTTP is disabled.
- Standard certificate verification is strictly enforced (no `verify=False` exemptions).
- Network requests include hard timeouts (10 seconds) to prevent Denial of Service (DoS) through slow-loris or stalled connection attacks.

### 3. Protection of Sensitive Data
- **Zero Credentials in UI / Source**: All sidebar input widgets for API keys and channel IDs have been removed. Telemetry and Telegram credentials are read solely through `get_secret()`, which checks `st.secrets` then environment variables, never raising and never reflecting keys back to the client.
- **Log and Error Sanitization**: Logged errors catch `Exception` and log only exception type names (`type(exc).__name__`). URLs, query parameters, API keys, and headers are never written to standard output or error streams. The `redact()` filter sanitizes any potential leaked key strings with `***`.
- **Streamlit Hardening**: Configured `[client] showErrorDetails = "none"` to suppress raw Python tracebacks in the browser, `[server] enableXsrfProtection = true` to protect against cross-site request forgery, and `[browser] gatherUsageStats = false` to eliminate third-party telemetry.
- **Git Hygiene**: `.gitignore` strictly excludes `.streamlit/secrets.toml`, `.env`, and virtual environment directories.

### 4. Responsible Data Management
- Telemetry values are strictly coerced to numeric types (`pd.to_numeric(..., errors="coerce")`). All `NaN` and `Inf` entities are sanitized before ingestion into ML pipelines.
- Machine learning models are loaded exclusively from fixed, trusted local file paths (`models/*.joblib`). Network data is never evaluated with `eval()`, `exec()`, or `pickle.loads()`.
- HTML rendering applies strict input escaping (`html.escape`) and whitespace normalization to prevent Cross-Site Scripting (XSS) and markdown layout corruption.

---

## Known Caveats

1. **Public Channel Visibility**:
   - If a ThingSpeak channel is configured as "Public" in the ThingSpeak console, its telemetry feeds can be read by anyone possessing the numeric Channel ID without requiring a Read Key. To ensure complete privacy, commercial deployments must designate channels as "Private".

2. **Wokwi Simulation Clock Jitter**:
   - Microcontroller execution in Wokwi runs within a browser-based WebAssembly/JavaScript environment. Under CPU load or browser tab throttling, the simulated ESP32 clock may drift slower than wall-clock real time. The stale data threshold is defaulted to 5 minutes to tolerate simulation jitter without falsely flagging the device as offline.

3. **Stateless Model Architecture**:
   - The Isolation Forest anomaly detector is evaluated pointwise on instantaneous microclimate triplets `[Temperature, Humidity, VPD]`. Temporal trend shifts over long horizons rely on the multi-step autoregressive lag model in the 1-hour forecast engine.

---

## Project Structure

```
├── .streamlit/
│   ├── config.toml                      # Theme and server security hardening
│   ├── secrets.toml                     # Private local credentials (gitignored)
│   └── secrets.toml.example             # Credential configuration template
├── models/
│   ├── greenhouse_forecast_bundle.joblib # 1-hour temperature forecast bundle
│   ├── climate_inference_bundle.joblib  # Isolation Forest anomaly detection bundle
│   ├── greenhouse_predict.py            # Forecast model prediction logic
│   ├── predict.py                       # Forwarding module for forecast logic
│   └── model_metadata_and_metrics.json  # Model evaluation metrics and warnings
├── config.py                            # Central constants and security helpers
├── anomaly.py                           # Isolation Forest scoring & VPD calculation
├── thingspeak.py                        # Hardened ThingSpeak client with HTTPS & regex check
├── notifier.py                          # Telegram risk notifications with global cooldown
├── ui.py                                # Component library, HTML sanitizer, Plotly charts
├── styles.css                           # Modern dark greenhouse theme
├── app.py                              # Main Streamlit dashboard application
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # System documentation and security analysis
```
