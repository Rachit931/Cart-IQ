import pandas as pd


def time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build time-based customer-level features
    from transaction-level data.

    Features:
    - CustomerTenure: Total active duration of customer (days)

    - RecencyNormalized: Recency scaled by tenure
      (captures inactivity relative to customer lifetime)

    Notes:
    - Raw recency is NOT returned (already in RFM)
    - No frequency-based features (avoid duplication wtih RFM)

    Parameters:
    - df: cleaned transactional dataframe

    Returns:
    - time_features DataFrame
    """

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    customer_dates = (
        df.groupby("CustomerID")
        .agg(
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max"),
        )
        .reset_index()
    )

    reference_date = df["InvoiceDate"].max()

    # Core Durations
    customer_dates["CustomerTenure"] = (
        customer_dates["LastPurchase"] - customer_dates["FirstPurchase"]
    ).dt.days

    customer_dates["Recency"] = (
        reference_date - customer_dates["LastPurchase"]
    ).dt.days

    # Avoid divide-by-zero
    customer_dates["CustomerTenure"] = customer_dates["CustomerTenure"].replace(0, 1)

    # Engineered signal
    customer_dates["RecencyNormalized"] = (
        customer_dates["Recency"] / customer_dates["CustomerTenure"]
    )

    return customer_dates[
        [
            "CustomerID",
            "CustomerTenure",
            "RecencyNormalized",
        ]
    ]
