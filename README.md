# ✈️ Flight Delay Predictor

Predicts whether a flight will be delayed >15 minutes using historical delay data, airline, airport, weather, and time-of-day patterns.

## 🧠 How It Works
```
Historical Flight Data (2M+ flights)
          ↓
Feature Engineering:
  • Airline delay history (avg, max, std)
  • Airport congestion (arrivals/hr, gate utilization)
  • Weather (wind speed, visibility, precipitation)
  • Temporal (hour, day-of-week, season, holidays)
  • Route (distance, aircraft type)
          ↓
Handle class imbalance (75% on-time, 25% delayed)
          ↓
Train XGBoost + LightGBM ensemble
          ↓
Score prediction + feature importance (SHAP)
```

## 🛠️ Tech Stack
- **Pandas, NumPy** — feature engineering
- **Scikit-learn** — preprocessing, train-test split
- **XGBoost, LightGBM** — gradient boosting ensembles
- **SHAP** — feature importance + explainability
- **Streamlit** — web interface

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/flight-delay-predictor
cd flight-delay-predictor
pip install -r requirements.txt
streamlit run app.py
```

## 💼 Business Impact
- Airlines use to rebook passengers proactively (save rebooking costs)
- Travelers predict delays 24 hours in advance (plan connections)
- Gate scheduling optimizes for congestion (reduce cascading delays)

## 🎤 Interview Talking Points

**1. Feature engineering for time-series domains is non-obvious.**
Not just: airline, airport, weather. Better: rolling averages ("this airline delayed 40% of flights last week"), volatility ("Hartsfield has 2x more delays Mon than Fri"), interaction features ("American Airlines + ATL + rain = high delay").

**2. Class imbalance (75% on-time) destroys naive models.**
If you train on raw data, model learns to always predict "on-time." SMOTE, class weights, or threshold tuning fixes it. Industry standard: collect more delay samples, balance to 50-50.

**3. SHAP + business explanation drive adoption.**
Prediction alone: "85% delay risk." With SHAP: "Weather (wind +12%), airport congestion (+8%), aircraft swap (+5%)." Airlines now see actionable drivers: "Book an earlier departure to avoid peak times."
