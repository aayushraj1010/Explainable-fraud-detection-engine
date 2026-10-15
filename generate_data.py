import numpy as np
import pandas as pd

np.random.seed(42)
n_rows = 10000

data = {
    "amt_ratio_30d": np.random.exponential(scale=1.0, size=n_rows).round(2),
    "tx_velocity_1hr": np.random.poisson(lam=1.2, size=n_rows),
    "distance_km": np.random.exponential(scale=15.0, size=n_rows).round(1),
    "is_late_night": np.random.choice([0, 1], size=n_rows, p=[0.88, 0.12]),
}

df = pd.DataFrame(data)

fraud_score = (
    (df["amt_ratio_30d"] > 3.5).astype(int) * 2
    + (df["tx_velocity_1hr"] >= 4).astype(int) * 2
    + (df["distance_km"] > 100).astype(int) * 2
    + df["is_late_night"]
)
noise = np.random.normal(0, 0.8, size=n_rows)
df["is_fraud"] = ((fraud_score + noise) >= 4.0).astype(int)

df.to_csv("transactions.csv", index=False)
print(
    f"SUCCESS! Created transactions.csv with {len(df)} rows and"
    f" {df['is_fraud'].sum()} fraud cases."
)