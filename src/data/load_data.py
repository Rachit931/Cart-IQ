from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Online_Retail.xlsx"


def load_raw_data() -> pd.DataFrame:
    """
    Load raw e-commerce dataset from Excel file.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)

    if df.empty:
        raise ValueError("Loaded DataFrame is empty")

    expected_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]

    missing_cols = set(expected_columns) - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    return df


PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"


def load_processed_data() -> pd.DataFrame:
    """
    Load processed customer-level
    feature dataset.
    """

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at " f"{PROCESSED_DATA_PATH}"
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)

    if df.empty:
        raise ValueError("Processed dataset is empty")

    return df
