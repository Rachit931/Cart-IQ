import pandas as pd


def build_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer value-oriented features.

    Features:
    ----------
    - TotalSpend:
        Total money spent by customer

    - AvgMonthlySpend:
        Average spend per active month

    - MonthlySpendStd:
        Monthly spending volatility
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
        .agg(MonthlySpend=("TotalPrice", "sum"))
        .reset_index()
    )

    monthly_features = (
        monthly_spend.groupby("CustomerID")
        .agg(
            AvgMonthlySpend=("MonthlySpend", "mean"),
            MonthlySpendStd=("MonthlySpend", "std"),
        )
        .reset_index()
    )

    monthly_features["MonthlySpendStd"] = monthly_features["MonthlySpendStd"].fillna(0)

    total_spend = (
        df.groupby("CustomerID").agg(TotalSpend=("TotalPrice", "sum")).reset_index()
    )

    value_features = total_spend.merge(monthly_features, on="CustomerID", how="left")

    return value_features
