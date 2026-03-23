import pandas as pd


def clean_data(df):

    df = df.copy()

    # 1. Remove missing values
    df = df.dropna(subset=["CustomerID", "Description"])

    # 2. Convert datatypes
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # 3. Remove cancalled invoices
    cancelled_mask = df["InvoiceNo"].astype(str).str.startswith("C")
    df = df[~cancelled_mask]

    # 4. Remove invalid quantities
    df = df[df["Quantity"] > 0]

    # 5. Remove invalid prices
    df = df[df["UnitPrice"] > 0]

    # 6. Remove invalid country
    df = df[df["Country"] != "Unspecified"]

    # 7. Resetting index
    return df.reset_index(drop=True)
