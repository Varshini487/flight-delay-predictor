import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="✈️ Flight Delay Predictor", layout="wide")
st.title("✈️ Flight Delay Predictor")
st.markdown("Predict flight delays 24 hours in advance using ML")

# Generate synthetic flight data
@st.cache_data
def generate_flight_data(n=5000):
    np.random.seed(42)
    airlines = np.random.choice(["American", "Delta", "United", "Southwest", "JetBlue"], n)
    airports = np.random.choice(["ATL", "LAX", "ORD", "DFW", "JFK"], n)
    distance = np.random.randint(200, 2500, n)
    wind_speed = np.random.gamma(2, 4, n)
    visibility = np.random.gamma(5, 1, n)
    hour = np.random.randint(0, 24, n)
    day_of_week = np.random.randint(0, 7, n)
    
    # Create delay based on features
    base_delay_risk = 0.15
    delay_risk = base_delay_risk
    delay_risk += (wind_speed > 15).astype(float) * 0.15
    delay_risk += (visibility < 3).astype(float) * 0.20
    delay_risk += ((hour >= 17) & (hour <= 19)).astype(float) * 0.10
    delay_risk += (day_of_week == 4).astype(float) * 0.08
    delay_risk += (distance > 2000).astype(float) * 0.05
    
    delay = (np.random.random(n) < delay_risk).astype(int)
    
    df = pd.DataFrame({
        "airline": airlines,
        "airport": airports,
        "distance": distance,
        "wind_speed": wind_speed,
        "visibility": visibility,
        "hour": hour,
        "day_of_week": day_of_week,
        "delay": delay
    })
    return df

df = generate_flight_data()

tab1, tab2, tab3 = st.tabs(["📊 Data", "🤖 Model", "🔮 Predict"])

with tab1:
    st.subheader("Flight Data Overview")
    st.dataframe(df.head(20), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flights", len(df))
    col2.metric("Delayed", df["delay"].sum())
    col3.metric("Delay Rate", f"{df['delay'].mean():.1%}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    df[df.delay==0]["wind_speed"].hist(bins=30, ax=axes[0,0], alpha=0.7, label="On-time", color="green")
    df[df.delay==1]["wind_speed"].hist(bins=30, ax=axes[0,0], alpha=0.7, label="Delayed", color="red")
    axes[0,0].set_title("Wind Speed vs Delay")
    axes[0,0].legend()
    
    df.groupby("hour")["delay"].mean().plot(ax=axes[0,1], color="orange")
    axes[0,1].set_title("Delay Rate by Hour of Day")
    
    df.groupby("airport")["delay"].mean().plot(kind="barh", ax=axes[1,0], color="steelblue")
    axes[1,0].set_title("Delay Rate by Airport")
    
    df.groupby("airline")["delay"].mean().plot(kind="barh", ax=axes[1,1], color="purple")
    axes[1,1].set_title("Delay Rate by Airline")
    
    st.pyplot(fig)

with tab2:
    if st.button("🚀 Train Model"):
        # Feature engineering
        df_model = df.copy()
        df_model["is_evening"] = ((df_model["hour"] >= 17) & (df_model["hour"] <= 19)).astype(int)
        df_model["high_wind"] = (df_model["wind_speed"] > 15).astype(int)
        df_model["low_visibility"] = (df_model["visibility"] < 3).astype(int)
        df_model["friday"] = (df_model["day_of_week"] == 4).astype(int)
        df_model["long_flight"] = (df_model["distance"] > 2000).astype(int)
        
        # Encode categorical
        airline_enc = pd.get_dummies(df_model["airline"], prefix="airline")
        airport_enc = pd.get_dummies(df_model["airport"], prefix="airport")
        
        X = pd.concat([
            df_model[["distance", "wind_speed", "visibility", "hour", "day_of_week",
                     "is_evening", "high_wind", "low_visibility", "friday", "long_flight"]],
            airline_enc, airport_enc
        ], axis=1)
        y = df_model["delay"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        model.fit(X_train_scaled, y_train)
        
        accuracy = model.score(X_test_scaled, y_test)
        st.success(f"✅ Model trained! Accuracy: **{accuracy:.1%}**")
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["feature_names"] = X.columns
        
        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(10)
        
        fig, ax = plt.subplots()
        ax.barh(feature_importance["feature"], feature_importance["importance"], color="skyblue")
        ax.set_title("Top 10 Feature Importance")
        st.pyplot(fig)

with tab3:
    st.subheader("Predict Flight Delay")
    
    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox("Airline", ["American", "Delta", "United", "Southwest", "JetBlue"])
        airport = st.selectbox("Destination Airport", ["ATL", "LAX", "ORD", "DFW", "JFK"])
        distance = st.slider("Flight Distance (miles)", 200, 2500, 1000)
    
    with col2:
        wind = st.slider("Wind Speed (mph)", 0, 30, 8)
        visibility = st.slider("Visibility (miles)", 0, 10, 7)
        hour = st.slider("Departure Hour", 0, 23, 14)
    
    if st.button("🔮 Predict"):
        if "model" in st.session_state:
            day_of_week = datetime.now().weekday()
            is_evening = 1 if (hour >= 17 and hour <= 19) else 0
            high_wind = 1 if wind > 15 else 0
            low_vis = 1 if visibility < 3 else 0
            friday = 1 if day_of_week == 4 else 0
            long_flight = 1 if distance > 2000 else 0
            
            input_data = pd.DataFrame({
                "distance": [distance],
                "wind_speed": [wind],
                "visibility": [visibility],
                "hour": [hour],
                "day_of_week": [day_of_week],
                "is_evening": [is_evening],
                "high_wind": [high_wind],
                "low_visibility": [low_vis],
                "friday": [friday],
                "long_flight": [long_flight],
            })
            
            for col in ["airline_American", "airline_Delta", "airline_United", "airline_Southwest", "airline_JetBlue"]:
                input_data[col] = 1 if f"airline_{airline}" == col else 0
            for col in ["airport_ATL", "airport_LAX", "airport_ORD", "airport_DFW", "airport_JFK"]:
                input_data[col] = 1 if f"airport_{airport}" == col else 0
            
            input_scaled = st.session_state["scaler"].transform(input_data)
            prob = st.session_state["model"].predict_proba(input_scaled)[0][1]
            
            if prob > 0.6:
                st.error(f"🚨 HIGH DELAY RISK: **{prob:.0%}**")
                st.write("💡 Recommendation: Book earlier flight or expect delays")
            elif prob > 0.35:
                st.warning(f"⚠️ MODERATE RISK: **{prob:.0%}**")
            else:
                st.success(f"✅ LOW RISK: **{prob:.0%}**")
        else:
            st.error("Train model first!")

st.markdown("---")
st.caption("Stack: Pandas · Scikit-learn · RandomForest · Streamlit")
