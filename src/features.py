"""Refine the cleaned transactions and build customer-level features for segmentation.

On top of `etl_rfm.py`, this step:
- drops exact duplicate transaction lines (they inflate Monetary),
- keeps only real product lines (drops POST, DOT, C2, M, BANK CHARGES, PADS...),
- subtracts cancelled value (from the raw Excel file) so Monetary is net spend,
- adds behavioural features used to profile the segments.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "Online Retail.xlsx"
CLEAN_PATH = PROJECT_ROOT / "data" / "online_retail_clean.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "customer_features.csv"

# Product StockCodes start with 5 digits; everything else is postage, fees or manual adjustments.
PRODUCT_CODE_PATTERN = r"^\d{5}"


def is_product(stock_codes: pd.Series) -> pd.Series:
    return stock_codes.astype(str).str.match(PRODUCT_CODE_PATTERN)


def refine_transactions(clean: pd.DataFrame) -> pd.DataFrame:
    refined = clean.drop_duplicates()
    refined = refined[is_product(refined["StockCode"])]
    return refined.copy()


def load_cancellations(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """Return cancelled product lines with a positive `CancelledValue` column."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_path}")
    raw = pd.read_excel(raw_path).dropna(subset=["CustomerID"])
    cancelled = raw[raw["InvoiceNo"].astype(str).str.startswith("C")]
    cancelled = cancelled[is_product(cancelled["StockCode"])].drop_duplicates()
    cancelled = cancelled.assign(
        CancelledValue=(cancelled["Quantity"] * cancelled["UnitPrice"]).abs()
    )
    return cancelled[["CustomerID", "InvoiceNo", "StockCode", "CancelledValue"]]


def build_customer_features(
    purchases: pd.DataFrame, cancellations: pd.DataFrame
) -> pd.DataFrame:
    reference_date = purchases["InvoiceDate"].max() + pd.Timedelta(days=1)
    features = purchases.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda dates: (reference_date - dates.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        GrossMonetary=("TotalPrice", "sum"),
        Tenure=("InvoiceDate", lambda dates: (reference_date - dates.min()).days),
        UniqueProducts=("StockCode", "nunique"),
        TotalItems=("Quantity", "sum"),
        Country=("Country", lambda countries: countries.mode().iat[0]),
    )

    cancelled_value = cancellations.groupby("CustomerID")["CancelledValue"].sum()
    features["CancelledValue"] = cancelled_value.reindex(features.index, fill_value=0.0)
    # Round to pennies so a fully cancelled order nets to exactly 0 instead of float noise.
    features["Monetary"] = (features["GrossMonetary"] - features["CancelledValue"]).round(2)

    # Customers who returned everything they bought carry no purchase behaviour to segment.
    features = features[features["Monetary"] > 0].copy()
    features["AvgOrderValue"] = features["Monetary"] / features["Frequency"]
    features["ReturnRate"] = features["CancelledValue"] / features["GrossMonetary"]

    columns = [
        "Recency",
        "Frequency",
        "Monetary",
        "Tenure",
        "AvgOrderValue",
        "UniqueProducts",
        "TotalItems",
        "ReturnRate",
        "Country",
    ]
    return features[columns].reset_index()


def main() -> None:
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["InvoiceDate"])
    purchases = refine_transactions(clean)
    cancellations = load_cancellations()
    features = build_customer_features(purchases, cancellations)
    features.to_csv(FEATURES_PATH, index=False)

    print(f"Clean rows: {len(clean):,}")
    print(f"Duplicate rows removed: {clean.duplicated().sum():,}")
    print(f"Product rows kept: {len(purchases):,}")
    print(f"Cancelled product lines: {len(cancellations):,}")
    print(f"Customers: {len(features):,}")
    print(f"Saved: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
