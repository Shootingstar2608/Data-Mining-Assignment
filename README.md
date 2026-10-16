# Customer Segmentation: ETL, RFM and Modeling

## Setup

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## 1. Run the ETL pipeline

The script reads `Online Retail.xlsx`, removes invalid transactions, creates
`TotalPrice`, and aggregates customer-level RFM features.

```bash
python src/etl_rfm.py
```

Generated files:

- `data/online_retail_clean.csv`: valid transaction-level data.
- `data/rfm.csv`: one row per customer with `Recency`, `Frequency`, and `Monetary`.

## 2. Build customer features

Refines the clean data (drops duplicate lines and non-product codes such as
`POST`/`M`/`BANK CHARGES`, subtracts cancelled orders from the raw Excel file)
and builds net RFM plus profiling features (`Tenure`, `AvgOrderValue`,
`UniqueProducts`, `TotalItems`, `ReturnRate`, `Country`).

```bash
python src/features.py
```

Generated file: `data/customer_features.csv`.

## 3. Modeling

`notebooks/modeling.ipynb` compares four segmentation approaches on
log-transformed, standardized RFM: RFM scoring (rule-based baseline), K-Means
(final model, k = 4), Gaussian Mixture Model, and hierarchical clustering (Ward).
It then profiles the segments and proposes a marketing strategy for each.

```bash
jupyter lab notebooks/modeling.ipynb
```

Generated files:

- `outputs/customer_segments.csv`: segment labels from every model for each customer.
- `outputs/segment_profile.csv`: profile of the four K-Means segments.
- `outputs/model_comparison.csv`: silhouette, Davies-Bouldin, Calinski-Harabasz per model.
- `outputs/figures/*.png`: charts for the report.
