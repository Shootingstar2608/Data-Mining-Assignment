"""Clean the Online Retail dataset and build customer-level RFM data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "Online Retail.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data"
CLEAN_PATH = OUTPUT_DIR / "online_retail_clean.csv"
RFM_PATH = OUTPUT_DIR / "rfm.csv"


def load_data(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the original Excel file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    return pd.read_excel(input_path)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "InvoiceNo",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["CustomerID"])
    cleaned = cleaned[~cleaned["InvoiceNo"].astype(str).str.startswith("C")]
    cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned["UnitPrice"] > 0)]
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"])
    cleaned["TotalPrice"] = cleaned["Quantity"] * cleaned["UnitPrice"]
    return cleaned


def build_rfm(cleaned: pd.DataFrame) -> pd.DataFrame:
    reference_date = cleaned["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        cleaned.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda dates: (reference_date - dates.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )
    return rfm


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = load_data()
    cleaned = clean_transactions(raw)
    rfm = build_rfm(cleaned)
    cleaned.to_csv(CLEAN_PATH, index=False)
    rfm.to_csv(RFM_PATH, index=False)

    print(f"Raw rows: {len(raw):,}")
    print(f"Clean rows: {len(cleaned):,}")
    print(f"Customers: {len(rfm):,}")
    print(f"Saved: {CLEAN_PATH}")
    print(f"Saved: {RFM_PATH}")


if __name__ == "__main__":
    main()