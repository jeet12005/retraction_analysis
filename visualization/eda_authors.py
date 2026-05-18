"""
visualization/eda_authors.py

Analyzes retractions by author, paired with their associated country.
Produces bar charts for the top 10 and bottom 10 author–country pairs by retraction volume.
"""

import matplotlib.pyplot as plt
from collections import Counter
from data_processing.preprocess import load_data


def get_author_country_counts(df):
    """
    Build (Author, Country) pair counts by matching positional indices
    within semicolon-delimited cells.
    """
    df = df.dropna(subset=["Author", "Country"])
    pairs = []

    for authors, countries in zip(df["Author"], df["Country"]):
        author_list = [a.strip() for a in authors.split(";") if a.strip()]
        country_list = [c.strip() for c in countries.split(";") if c.strip()]
        for i in range(min(len(author_list), len(country_list))):
            pairs.append((author_list[i], country_list[i]))

    return Counter(pairs)


def plot_author_country_bar(data: list, title: str):
    labels = [f"{author} ({country})" for (author, country), _ in data]
    counts = [count for _, count in data]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts, color="mediumseagreen")
    plt.xlabel("Author (Country)")
    plt.ylabel("Number of Retractions")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    author_counts = get_author_country_counts(df)

    top_10 = author_counts.most_common(10)
    bottom_10 = sorted(author_counts.items(), key=lambda x: x[1])[:10]

    plot_author_country_bar(top_10, "Top 10 Authors by Retraction Count")
    plot_author_country_bar(bottom_10, "Authors with Fewest Retractions")


if __name__ == "__main__":
    main()
