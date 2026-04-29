import pandas as pd


def build_rfm_features(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """
    Build RFM customer level features
    from transaction level data.

    Features:
    - Recency: Time since customer's
    last purchase
    - Frequency: No. of times a customer
    has purchased
    - Monetary: Total money spent
    by the customer

    Parameters:
    - df: cleaned DataFrame
    - reference_date: date to calculate recency
    (default = max data in the dataset)

    Returns:
    - rfm DataFrame
    """

    df = df.copy()

    # Create Total Price column
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Set reference date
    if reference_date is None:
        reference_date = df["InvoiceDate"].max()

    # Group by customer
    rfm = df.groupby("CustomerID").agg(
        Recency=(
            "InvoiceDate",
            lambda x: (reference_date - x.max()).days,
        ),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )

    # Reset Index
    rfm = rfm.reset_index()

    return rfm
