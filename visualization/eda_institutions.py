"""
visualization/eda_institutions.py

Analyzes retractions by institution, paired with the institution's country.
Produces bar charts for the top 10 and bottom 10 institution–country pairs.
"""

import matplotlib.pyplot as plt
from collections import Counter
from data_processing.preprocess import load_data


def get_institution_country_counts(df):
    """
    Build (Institution, Country) pair counts by matching positional indices
    within semicolon-delimited cells.
    """
    df = df.dropna(subset=["Institution", "Country"])
    pairs = []

    for inst, country in zip(df["Institution"], df["Country"]):
        insts = [i.strip() for i in inst.split(";") if i.strip()]
        countries = [c.strip() for c in country.split(";") if c.strip()]
        for i in range(min(len(insts), len(countries))):
            pairs.append((insts[i], countries[i]))

    return Counter(pairs)


def plot_institution_country_bar(data: list, title: str):
    labels = [f"{inst} ({country})" for (inst, country), _ in data]
    counts = [count for _, count in data]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts, color="lightcoral")
    plt.xlabel("Institution (Country)")
    plt.ylabel("Number of Retractions")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    inst_counts = get_institution_country_counts(df)

    top_10 = inst_counts.most_common(10)
    bottom_10 = sorted(inst_counts.items(), key=lambda x: x[1])[:10]

    plot_institution_country_bar(top_10, "Top 10 Institutions by Retraction Count")
    plot_institution_country_bar(bottom_10, "Institutions with Fewest Retractions")


if __name__ == "__main__":
    main()
