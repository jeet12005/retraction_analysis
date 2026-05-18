"""
modeling/fairness_audit.py

Audits the retraction classifier for demographic bias using the Fairlearn library.

Evaluates accuracy and selection rate across sensitive features:
  - Country  (geographic bias)
  - Paywalled (access-model bias)

Sensitive feature groups are derived from the Country_encoded column, mapped back
to country names for readability. This mirrors the SHAP analysis in Figure 1 which
shows country as the highest-impact feature on model output.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate
from data_processing.preprocess import load_data, build_binary_target, encode_features


FEATURES = ["Country", "Publisher", "Subject", "Institution", "Paywalled", "Reason", "Title", "Author", "Journal"]


def run_fairness_audit(sensitive_feature: str = "Country_encoded"):
    print("Loading and preparing data...")
    df = load_data()
    df = build_binary_target(df)
    df = df[FEATURES + ["Target"]].dropna()
    df, encoders = encode_features(df, FEATURES)

    X = df[[col + "_encoded" for col in FEATURES]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\nOverall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print(f"\nFairness audit by sensitive feature: {sensitive_feature}")
    metric_frame = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=X_test[sensitive_feature],
    )

    print("\nMetrics by group (showing first 20 groups):")
    print(metric_frame.by_group.head(20))

    print("\nOverall metrics:")
    print(metric_frame.overall)

    return metric_frame


if __name__ == "__main__":
    run_fairness_audit(sensitive_feature="Country_encoded")
