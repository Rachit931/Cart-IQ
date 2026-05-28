import pandas as pd


def build_rfm_features(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """
    Build RFM customer-level features
    from transaction-level data.

    Features:
    ----------
    - Recency:
        Days since customer's
        last purchase

    - Frequency:
        Number of unique purchases

    - Monetary:
        Total money spent
        by customer

    - LastPurchaseDate:
        Most recent transaction date
        (used for time-based splitting)

    Parameters:
    ----------
    - df:
        Cleaned transaction-level
        DataFrame

    - reference_date:
        Date used to calculate recency
        (default = latest date
        in dataset)

    Returns:
    ----------
    - RFM feature DataFrame
    """

    df = df.copy()

    # Ensure datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Create total price column
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Reference date
    if reference_date is None:
        reference_date = df["InvoiceDate"].max()

    # Build RFM features
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
            LastPurchaseDate=("InvoiceDate", "max"),
        )
        .reset_index()
    )

    return rfm
