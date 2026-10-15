import joblib
import lightgbm as lgb
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("transactions.csv")
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

imbalance_ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    scale_pos_weight=imbalance_ratio,
    random_state=42,
    verbose=-1,
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, probs)
print(f"Model Trained! PR-AUC Score: {pr_auc:.4f}\n")
print(classification_report(y_test, model.predict(X_test), zero_division=0))

explainer = shap.TreeExplainer(model)

joblib.dump(model, "fraud_model.pkl")
joblib.dump(explainer, "shap_explainer.pkl")
print("SUCCESS! Saved fraud_model.pkl and shap_explainer.pkl")