import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Explainable Fraud Engine",
    page_icon="🛡️",
    layout="wide",
)

# Load Model and SHAP Explainer directly
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_model.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    return model, explainer

model, explainer = load_artifacts()

st.title("🛡️ Real-Time Explainable Fraud Detection Engine")
st.caption("Powered by LightGBM + SHAP TreeExplainer (Cloud Edition)")

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
        # Construct feature array
        input_data = pd.DataFrame([[amt_ratio, velocity, distance, late_night]], 
                                  columns=["amt_ratio_30d", "tx_velocity_1hr", "distance_km", "is_late_night"])
        
        # Predict probability
        fraud_prob = float(model.predict_proba(input_data)[0][1])
        status = "🚨 BLOCKED (FRAUD)" if fraud_prob > 0.5 else "✅ APPROVED"
        
        # Calculate SHAP values
        shap_values = explainer(input_data)
        # Extract shap values for the positive class
        if len(shap_values.values.shape) == 3:
            s_vals = shap_values.values[0, :, 1]
        else:
            s_vals = shap_values.values[0]

        m1, m2 = st.columns(2)
        m1.metric("Decision", status)
        m2.metric("Fraud Probability", f"{fraud_prob * 100:.1f}%")

        st.write("### Why did the AI make this decision? (SHAP Impact)")
        st.caption("Positive bars push the transaction toward FRAUD. Negative bars push toward SAFE.")
        
        shap_df = pd.DataFrame({
            "Feature": input_data.columns,
            "SHAP Risk Impact": s_vals
        }).set_index("Feature")
        
        st.bar_chart(shap_df)