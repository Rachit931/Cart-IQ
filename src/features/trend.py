import numpy as np
import pandas as pd


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer spending trend features

    Features:
    ---------
    - RecentSpend:
        Spend during recent time window

    - OldSpend:
        Spend before recent window

    - SpendGrowthRate:
        Log-based spending growth signal

    Parameters:
    -----------
    - df:
        Cleaned transaction-level DataFrame

    Returns:
    --------
    - trend features DataFrame
    """

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    split_date = df["InvoiceDate"].max() - pd.Timedelta(days=90)

    recent_df = df[df["InvoiceDate"] >= split_date]

    old_df = df[df["InvoiceDate"] < split_date]

    recent_spend = (
        recent_df.groupby("CustomerID")["TotalPrice"]
        .sum()
        .reset_index(name="RecentSpend")
    )

    old_spend = (
        old_df.groupby("CustomerID")["TotalPrice"].sum().reset_index(name="OldSpend")
    )

    trend = recent_spend.merge(old_spend, on="CustomerID", how="outer").fillna(0)

    # Stable log-based growth signal
    trend["SpendGrowthRate"] = np.log1p(trend["RecentSpend"]) - np.log1p(
        trend["OldSpend"]
    )

    return trend
