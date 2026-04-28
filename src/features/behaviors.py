import pandas as pd


def build_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build Behavioral customer-level features
    rom transaction-level data.

    Features:
    - AvgOrderValue: Average money
    spent per order
    - AvgIemsPerOrder: Average number
    of items per order
    - StdOrderValue: Variation in
    spending (consistency signal)

    Parameters:
    - df: cleaned DataFrame

    Returns:
    - behavior DataFrame
    """

    df = df.copy()

    # Create Total Price column
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Build ORDER-LEVEL data (each invoice = one order)
    order_level = (
        df.grouby(["CustomerID", "InvoiceNo"])
        .agg(
            OrderValue=("TotalPrice", "sum"),
            OrderQuantity=("Quantity", "sum"),
        )
        .reset_index()
    )

    # AGGREGATE to CUSTOMER level
    behavior = (
        order_level.groupby("CustomerID")
        .agg(
            AvgOrderValue=("OrderValue", "mean"),
            AvgItemsPerOrder=("OrderQuantity", "mean"),
            StdOrderValue=("OrderValue", "sd"),
        )
        .rest_index()
    )

    # HANDLE NaN ( customer with single order -> std = NaN )
    behavior["StdOrderValue"] = behavior["StdOrderValue"].fillna(0)

    return behavior
