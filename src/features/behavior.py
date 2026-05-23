# src/features/behavior.py

import pandas as pd


def build_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build behavioral customer-level features
    from transaction-level data.

    Features:
    ----------
    - AvgOrderValue:
        Average money spent per order

    - AvgItemsPerOrder:
        Average number of items
        purchased per order

    - StdOrderValue:
        Variation in order spending
        (consistency signal)

    Parameters:
    ----------
    - df:
        Cleaned transaction-level DataFrame

    Returns:
    ----------
    - behavior DataFrame
    """

    df = df.copy()

    # Create TotalPrice column
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # BUILD ORDER-LEVEL DATA

    order_level = (
        df.groupby(["CustomerID", "InvoiceNo"])[["TotalPrice", "Quantity"]]
        .sum()
        .reset_index()
    )

    order_level = order_level.rename(
        columns={
            "TotalPrice": "OrderValue",
            "Quantity": "OrderQuantity",
        }
    )

    # CUSTOMER-LEVEL AGGREGATION

    behavior = (
        order_level.groupby("CustomerID")
        .agg(
            {
                "OrderValue": ["mean", "std"],
                "OrderQuantity": "mean",
            }
        )
        .reset_index()
    )

    # Flatten multi-index columns
    behavior.columns = [
        "CustomerID",
        "AvgOrderValue",
        "StdOrderValue",
        "AvgItemsPerOrder",
    ]

    # Handle NaN std values
    behavior["StdOrderValue"] = behavior["StdOrderValue"].fillna(0)

    return behavior
