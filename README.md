# Retraction Watch — Academic Paper Retraction Analysis

End-to-end ML pipeline for analyzing and classifying academic paper retractions across nearly 1 million records from the Retraction Watch database.

## Project Highlights
- classifying retraction reasons using Random Forest and XGBoost
- SHAP-based explainability showing country and publisher as top predictive features
- Fairness auditing with `fairlearn` across sensitive demographic features (country, paywalled status)
- Full EDA suite: temporal trends, subject areas, institutions, authors, and publishers

## Structure
```
retraction_analysis/
├── data_processing/
│   └── preprocess.py          # Cleaning, splitting multi-value fields, encoding
├── visualization/
│   ├── eda_country.py         # Country-level retraction counts & trends
│   ├── eda_reasons.py         # Reason frequency analysis
│   ├── eda_subjects.py        # Subject area breakdown
│   ├── eda_publishers.py      # Publisher-level analysis
│   ├── eda_authors.py         # Author-level analysis
│   ├── eda_institutions.py    # Institution-level analysis
│   └── eda_temporal.py        # Time-series retraction trends
├── modeling/
│   ├── classify_retraction_nature.py   # XGBoost classifier + SHAP explainability
│   └── fairness_audit.py              # Fairlearn bias/fairness metrics
├── outputs/
│   └── Reason_By_Country_Table.csv    # Pivot table: top reasons × top 10 countries
└── README.md
```

## Setup
```bash
pip install pandas scikit-learn xgboost shap fairlearn matplotlib
```

Place `retraction_watch.csv` in the project root, then run any module independently.

## Data
Source: [Retraction Watch Database](https://retractionwatch.com/retraction-watch-database-user-guide/)  
Key fields used: `Country`, `Publisher`, `Subject`, `Institution`, `Reason`, `RetractionNature`, `RetractionDate`, `Paywalled`
