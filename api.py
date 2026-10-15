import time
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Explainable Fraud Detection API")

model = joblib.load("fraud_model.pkl")
explainer = joblib.load("shap_explainer.pkl")

class Transaction(BaseModel):
    amt_ratio_30d: float
    tx_velocity_1hr: int
    distance_km: float
    is_late_night: int

@app.post("/predict")
def predict_fraud(tx: Transaction):
    start_time = time.perf_counter()
    df_input = pd.DataFrame([tx.model_dump()])

    fraud_prob = float(model.predict_proba(df_input)[0][1])
    is_flagged = fraud_prob >= 0.50

    shap_values = explainer.shap_values(df_input)
    vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    reasons = {col: round(float(val), 4) for col, val in zip(df_input.columns, vals)}
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "fraud_probability": round(fraud_prob, 4),
        "status": "BLOCKED (FRAUD)" if is_flagged else "APPROVED",
        "latency_ms": latency_ms,
        "shap_explanations": reasons,
    }