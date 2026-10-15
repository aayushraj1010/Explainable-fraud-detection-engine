import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Explainable Fraud Engine", layout="wide")
st.title("🛡️ Real-Time Explainable Fraud Detection Engine")
st.caption("Powered by LightGBM + SHAP TreeExplainer + FastAPI")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Simulate Incoming Card Swipe")
    amt_ratio = st.slider("Amount vs. User's 30-Day Avg (x)", 0.1, 10.0, 4.8, 0.1)
    velocity = st.slider("Card Swipes in Last 1 Hour", 0, 12, 5)
    distance = st.slider("Distance from Previous Swipe (km)", 0.0, 500.0, 180.0, 5.0)
    late_night = st.selectbox(
        "Time of Transaction", 
        [0, 1], 
        format_func=lambda x: "Late Night (1 AM - 5 AM)" if x == 1 else "Normal Hours"
    )
    analyze_btn = st.button("Analyze Transaction", type="primary", use_container_width=True)

with col2:
    st.subheader("Live AI Decision & SHAP Explainability")
    if analyze_btn:
        payload = {
            "amt_ratio_30d": amt_ratio,
            "tx_velocity_1hr": velocity,
            "distance_km": distance,
            "is_late_night": late_night,
        }
        
        try:
            res = requests.post("http://127.0.0.1:8000/predict", json=payload).json()

            m1, m2, m3 = st.columns(3)
            m1.metric("Decision", res["status"])
            m2.metric("Fraud Probability", f"{res['fraud_probability'] * 100:.1f}%")
            m3.metric("Inference Latency", f"{res['latency_ms']} ms")

            st.write("### Why did the AI make this decision? (SHAP Impact)")
            st.caption("Positive bars push the transaction toward FRAUD. Negative bars push toward SAFE.")
            shap_df = pd.DataFrame.from_dict(
                res["shap_explanations"], orient="index", columns=["SHAP Risk Impact"]
            )
            st.bar_chart(shap_df)
            
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI server! Make sure your first terminal is running `uvicorn api:app --reload`.")