"""
visualization/eda_subjects.py

Analyzes retractions by academic subject area.
Produces bar charts for the top 10 and bottom 10 subjects by retraction volume.
"""

import matplotlib.pyplot as plt
from collections import Counter
from data_processing.preprocess import load_data, explode_semicolon_column


def get_subject_counts(df):
    df = explode_semicolon_column(df.dropna(subset=["Subject"]), "Subject")
    return Counter(df["Subject"])


def plot_subject_bar(data: list, title: str):
    subjects, counts = zip(*data)
    plt.figure(figsize=(10, 5))
    plt.bar(subjects, counts, color="lightcoral")
    plt.xlabel("Subject")
    plt.ylabel("Number of Retractions")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    subject_counts = get_subject_counts(df)

    top_10 = subject_counts.most_common(10)
    bottom_10 = sorted(subject_counts.items(), key=lambda x: x[1])[:10]

    plot_subject_bar(top_10, "Top 10 Subjects with Most Retractions")
    plot_subject_bar(bottom_10, "10 Subjects with Fewest Retractions")


if __name__ == "__main__":
    main()
