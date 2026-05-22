import pandas as pd


def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build product diversity-based
    customer-level features.

    Features:
    ----------
    - UniqueProducts:
        Number of distinct products
        purchased by customer

    - AvgUniqueProductsPerOrder:
        Customer product diversity
        normalized by order count

    Parameters:
    ----------
    - df:
        Cleaned transaction-level DataFrame

    Returns:
    ----------
    - product features DataFrame
    """

    df = df.copy()

    # Customer-level produtt diversity
    product_features = (
        df.groupby("CustomerID")
        .agg(
            UniqueProducts=("StockCode", "nunique"),
            UniqueInvoices=("InvoiceNo", "nunique"),
        )
        .reset_index()
    )

    # Average unique products per order
    product_features["AvgUniqueProductsPerOrder"] = (
        product_features["UniqueProducts"] / product_features["UniqueInvoices"]
    )

    return product_features[
        ["CustomerID", "UniqueProducts", "AvgUniqueProductsPerOrder"]
    ]
