from pathlib import Path
import pandas as pd

DATA_PATH = Path("../data/raw/Online_Retail.xlsx")


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
