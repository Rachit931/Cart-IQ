import pandas as pd


def build_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer value-oriented features.

    Features:
    ----------
    - TotalSpend:
        Total money spent by customer

    - AvgvalueSpend:
        Average spend per active month

    - MonthlySpendStd:
        value spending volatility
        (consistency signal)

    Parameters:
    ----------
    - df:
        Cleaned transaction-level DataFrame

    Returns:
    ----------
    - value features DataFrame
    """

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")

    monthly_spend = (
        df.groupby(["CustomerID", "YearMonth"])
        .agg(valueSpend=("TotalPrice", "sum"))
        .reset_index()
    )

    value_features = (
        monthly_spend.groupby("CustomerID")
        .agg(
            AvgvalueSpend=("valueSpend", "mean"),
            MonthlySpendStd=("valueSpend", "std"),
        )
        .reset_index()
    )

    value_features["MonthlySpendStd"] = value_features["MonthlySpendStd"].fillna(0)

    return value_features
