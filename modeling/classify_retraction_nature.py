"""
modeling/classify_retraction_nature.py

Trains an XGBoost classifier to predict whether a paper entry is a retraction
(vs. a correction or other notice). Handles class imbalance via scale_pos_weight.

Outputs:
  - Classification report (precision, recall, F1)
  - Feature importance pie chart
  - Retraction rate bar chart by country (top 15)
  - Retraction breakdown pie charts by publisher, subject, institution, and reason
  - Sample predictions table with decoded feature labels
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from data_processing.preprocess import load_data, build_binary_target, encode_features, explode_semicolon_column


FEATURES = ["Country", "Publisher", "Subject", "Institution", "Paywalled", "Reason"]


def train_model(df: pd.DataFrame):
    df = build_binary_target(df)
    df = df[FEATURES + ["Target"]].dropna()
    df, encoders = encode_features(df, FEATURES)

    X = df[[col + "_encoded" for col in FEATURES]]
    y = df["Target"]

    neg, pos = y.value_counts()
    scale_pos_weight = neg / pos

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost classifier...")
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, encoders, X_test, y_test, y_pred


def plot_feature_importance_pie(model, feature_names: list):
    importances = model.feature_importances_
    percentages = importances / importances.sum() * 100

    plt.figure(figsize=(8, 8))
    plt.pie(percentages, labels=feature_names, autopct="%1.1f%%", startangle=140)
    plt.title("Feature Importances (% of total)")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_retraction_rate_by_country(df: pd.DataFrame, top_n: int = 15):
    df = df[["Country", "RetractionNature"]].dropna()
    df = explode_semicolon_column(df[df["Country"].str.strip() != ""], "Country")

    total = df["Country"].value_counts()
    retracted = df[df["RetractionNature"] == "Retraction"]["Country"].value_counts()

    stats = pd.DataFrame({"Total": total, "Retracted": retracted}).fillna(0)
    stats["Retraction Rate (%)"] = (stats["Retracted"] / stats["Total"]) * 100
    top = stats.sort_values("Total", ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    plt.bar(top.index, top["Retraction Rate (%)"], color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Retraction Rate (%)")
    plt.title(f"Top {top_n} Countries by Retraction Rate")
    plt.tight_layout()
    plt.show()


def plot_retraction_breakdown_pie(df: pd.DataFrame, feature: str, top_n: int = 8):
    retracted = df[df["RetractionNature"] == "Retraction"]
    counts = retracted[feature].value_counts().dropna().head(top_n)

    if retracted[feature].nunique() > top_n:
        counts["Other"] = retracted[feature].value_counts().iloc[top_n:].sum()

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140)
    plt.title(f"Retractions by {feature} (Top {top_n} + Other)")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    model, encoders, X_test, y_test, y_pred = train_model(df)

    plot_feature_importance_pie(model, [col + "_encoded" for col in FEATURES])

    plot_retraction_rate_by_country(load_data())

    df_viz = load_data()
    for feature in ["Publisher", "Subject", "Institution", "Reason"]:
        plot_retraction_breakdown_pie(df_viz, feature, top_n=8)

    # Decode predictions for interpretability
    df_model = build_binary_target(load_data())
    df_model = df_model[FEATURES + ["Target"]].dropna()
    df_model, enc = encode_features(df_model, FEATURES)
    X = df_model[[col + "_encoded" for col in FEATURES]]
    y = df_model["Target"]
    _, X_test_raw, _, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

    result_df = X_test_raw.copy()
    result_df["Predicted"] = model.predict(X_test_raw)
    result_df["True"] = y_test_raw.values
    for col in FEATURES:
        result_df[col] = enc[col].inverse_transform(result_df[col + "_encoded"])

    print("\nSample Predictions:")
    print(result_df[FEATURES + ["Predicted", "True"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
