# Customer Segmentation: ETL and RFM

## Setup

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the ETL pipeline

The script reads `Online Retail.xlsx`, removes invalid transactions, creates
`TotalPrice`, and aggregates customer-level RFM features.

```bash
python src/etl_rfm.py
```

Generated files:

- `data/online_retail_clean.csv`: valid transaction-level data.
- `data/rfm.csv`: one row per customer with `Recency`, `Frequency`, and `Monetary`.