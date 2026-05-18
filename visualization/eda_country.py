"""
visualization/eda_country.py

Analyzes retraction counts by country.
Produces bar charts for the top 10 and bottom 10 countries by retraction volume.
"""

import matplotlib.pyplot as plt
from collections import Counter
from data_processing.preprocess import load_data, explode_semicolon_column


def get_country_counts(df):
    df = explode_semicolon_column(df.dropna(subset=["Country"]), "Country")
    return Counter(df["Country"])


def plot_country_bar(data: list, title: str):
    countries, counts = zip(*data)
    plt.figure(figsize=(10, 5))
    plt.bar(countries, counts, color="skyblue")
    plt.xlabel("Country")
    plt.ylabel("Number of Retractions")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    country_counts = get_country_counts(df)

    top_10 = country_counts.most_common(10)
    bottom_10 = sorted(country_counts.items(), key=lambda x: x[1])[:10]

    plot_country_bar(top_10, "Top 10 Countries with Most Retractions")
    plot_country_bar(bottom_10, "10 Countries with Fewest Retractions")


if __name__ == "__main__":
    main()
