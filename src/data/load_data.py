import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw e-commerce datasert from Excel file.

    Agrs:
        path (str): Path to the excel file

    Returns:
        pd.DataFrame: Loaded DataFrame
    """

    df = pd.read_excel(path)

    if df.empty:
        raise ValueError("Loaded DataFrame is empty")

    expected_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]

    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    return df
