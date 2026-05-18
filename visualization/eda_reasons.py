"""
visualization/eda_reasons.py

Analyzes retraction reasons across the full dataset and broken down by country.

Outputs:
  - Bar chart: Top 10 most common retraction reasons (global)
  - CSV: Pivot table of top reasons × top 10 countries (outputs/Reason_By_Country_Table.csv)
"""

import matplotlib.pyplot as plt
from collections import Counter
from data_processing.preprocess import load_data, explode_semicolon_column


def get_reason_counts(df):
    df = explode_semicolon_column(df.dropna(subset=["Reason"]), "Reason")
    return Counter(df["Reason"])


def plot_top_reasons(data: list, title: str):
    reasons, counts = zip(*data)
    plt.figure(figsize=(10, 6))
    plt.bar(reasons, counts, color="lightgreen")
    plt.xlabel("Reason for Retraction")
    plt.ylabel("Number of Retractions")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def build_reason_by_country_table(df, top_n_countries: int = 10, output_path: str = "outputs/Reason_By_Country_Table.csv"):
    """
    Build and save a pivot table: retraction reasons (rows) × top N countries (columns).
    Rows are sorted by total reason frequency descending.
    """
    df = df.dropna(subset=["Country", "Reason"])
    df = explode_semicolon_column(df, "Country")
    df = explode_semicolon_column(df, "Reason")

    top_countries = df["Country"].value_counts().head(top_n_countries).index.tolist()
    df_top = df[df["Country"].isin(top_countries)]

    pivot = df_top.pivot_table(index="Reason", columns="Country", aggfunc="size", fill_value=0)
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False).drop(columns="Total")

    pivot.to_csv(output_path)
    print(f"Saved reason-by-country table to {output_path}")
    return pivot


def main():
    df = load_data()

    reason_counts = get_reason_counts(df)
    top_10 = reason_counts.most_common(10)
    plot_top_reasons(top_10, "Top 10 Most Common Reasons for Retractions")

    pivot = build_reason_by_country_table(df)
    print("\nTop Retraction Reasons by Country (Top 10 Countries):")
    print(pivot.head(20))


if __name__ == "__main__":
    main()
