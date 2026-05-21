import pandas as pd


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
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

    trend["SpendGrowthRate"] = (trend["RecentSpend"] - trend["OldSpend"]) / (
        trend["OldSpend"] + 1
    )

    return trend
