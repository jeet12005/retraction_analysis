# retraction_analysis/run_all.py
"""
Entry point to run the full analysis pipeline sequentially.
Each module can also be run independently from its own directory.
"""

from visualization import eda_country, eda_reasons, eda_subjects, eda_publishers, eda_authors, eda_institutions, eda_temporal
from modeling import classify_retraction_nature, fairness_audit


def main():
    print("=" * 60)
    print("RETRACTION WATCH — FULL ANALYSIS PIPELINE")
    print("=" * 60)

    print("\n[1/8] Country EDA")
    eda_country.main()

    print("\n[2/8] Reason EDA + Country Pivot Table")
    eda_reasons.main()

    print("\n[3/8] Subject EDA")
    eda_subjects.main()

    print("\n[4/8] Publisher EDA")
    eda_publishers.main()

    print("\n[5/8] Author EDA")
    eda_authors.main()

    print("\n[6/8] Institution EDA")
    eda_institutions.main()

    print("\n[7/8] Temporal Trends")
    eda_temporal.main()

    print("\n[8/8] Classification + Fairness Audit")
    classify_retraction_nature.main()
    fairness_audit.run_fairness_audit()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
