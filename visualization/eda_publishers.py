"""
visualization/eda_publishers.py

Analyzes retractions by publisher, broken down by country of the associated paper.
Produces bar charts for the top 10 and bottom 10 publisher–country pairs by retraction volume.
"""

import matplotlib.pyplot as plt
from collections import Counter
from data_processing.preprocess import load_data


def get_publisher_country_counts(df):
    """
    Build (Publisher, Country) pair counts by matching positional indices
    within semicolon-delimited cells.
    """
    df = df.dropna(subset=["Publisher", "Country"])
    pairs = []

    for pub, country in zip(df["Publisher"], df["Country"]):
        pubs = [p.strip() for p in pub.split(";") if p.strip()]
        countries = [c.strip() for c in country.split(";") if c.strip()]
        for i in range(min(len(pubs), len(countries))):
            pairs.append((pubs[i], countries[i]))

    return Counter(pairs)


def plot_publisher_country_bar(data: list, title: str):
    labels = [f"{pub} ({country})" for (pub, country), _ in data]
    counts = [count for _, count in data]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts, color="lightseagreen")
    plt.xlabel("Publisher (Country)")
    plt.ylabel("Number of Retractions")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    pub_country_counts = get_publisher_country_counts(df)

    top_10 = pub_country_counts.most_common(10)
    # Bottom 10 excludes pairs with only a single retraction (likely noise)
    bottom_10 = sorted(
        [(pair, count) for pair, count in pub_country_counts.items() if count > 1],
        key=lambda x: x[1]
    )[:10]

    plot_publisher_country_bar(top_10, "Top 10 Publisher–Country Pairs by Retraction Count")
    plot_publisher_country_bar(bottom_10, "Lowest-Volume Publisher–Country Pairs (min. 2 retractions)")


if __name__ == "__main__":
    main()
