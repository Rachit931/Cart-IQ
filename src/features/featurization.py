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


def build_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Building customer-level behavioral features from
    transcation-level data.

    Features:
    - AvgOrderValue: Average money spent per order(InvoiceNo)
    - AvgQuantity: Average number of items per order
    - UniqueProducts: Number of distinct products purchased
    - TotalItems: Total quantity of items purchased per customer

    Parameters:
    - df (pd.DataFrame): Cleaned transactional dataset

    Returns:
    - pd.DataFrame: Customer-level behavioral features
    """

    df = df.copy()

    # Create TOtalPrice (transaction-level helper)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Build ORDER-LEVEL Data

    order_level = (
        df.groupby(["CustomerID", "InvoiceNo"])
        .agg(
            OrderValue=("TotalPrice", "sum"),
            OrderQuantity=("Quantity", "sum"),
        )
        .reset_index()
    )

    # Aggregate to CUSTOMER

    behavior_orders = (
        order_level.groupby("CustomerID")
        .agg(
            AvgOrderValue=("Order_value", "sum"),
            AvgQuantity=("Order_quantity", "mean"),
        )
        .reset_index()
    )

    # Product-based Features

    behavior_products = (
        df.groupby("CustomerID")
        .agg(
            UniqueProducts=("StockCode", "nunique"),
            TotalItems=("Quantity", "sum"),
        )
        .reset_index()
    )

    # Merge Both Feature sets

    behavior = behavior_orders.merge(behavior_products, on="CustomerID")

    return behavior
