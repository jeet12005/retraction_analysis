"""
visualization/eda_temporal.py

Analyzes retraction trends over time.

Outputs:
  - Line chart: Total retractions per year (global)
  - Bar chart: Retractions per year for a specified country and date range
"""

import matplotlib.pyplot as plt
from data_processing.preprocess import load_with_dates, explode_semicolon_column


def plot_global_retractions_over_time(df):
    yearly_counts = df["RetractionDate"].dt.year.value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    plt.plot(yearly_counts.index, yearly_counts.values, marker="o")
    plt.title("Total Retractions per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Retractions")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_retractions_by_country(df, country: str, start_year: int = 1940, end_year: int = 2024):
    """Bar chart of annual retraction counts for a single country within a date range."""
    df = df[df["RetractionDate"].dt.year.between(start_year, end_year)].copy()
    df = explode_semicolon_column(df.dropna(subset=["Country"]), "Country")

    country_df = df[df["Country"] == country]
    year_counts = country_df["RetractionDate"].dt.year.value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    plt.bar(year_counts.index, year_counts.values, color="skyblue")
    plt.title(f"Retractions in {country} ({start_year}–{end_year})")
    plt.xlabel("Year")
    plt.ylabel("Number of Retractions")
    plt.tight_layout()
    plt.show()


def main():
    df = load_with_dates()
    df = df.dropna(subset=["RetractionDate", "Country"])

    plot_global_retractions_over_time(df)
    plot_retractions_by_country(df, country="United States", start_year=1940, end_year=2024)


if __name__ == "__main__":
    main()
