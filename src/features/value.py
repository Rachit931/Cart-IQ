import pandas as pd


def build_value_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Monthly period
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")

    monthly_spend = (
        df.groupby(["CustomerID", "YearMonth"])
        .agg(MonthlySpend=("TotalPrice", "sum"))
        .reset_index()
    )

    avg_monthly = (
        monthly_spend.groupby("CustomerID")
        .agg(AvgMonthlySpend=("MonthlySpend", "mean"))
        .reset_index()
    )

    customer_value = (
        df.groupby("CustomerID")
        .agg(
            TotalSpend=("TotalPrice", "sum"),
            AvgOrderValue=("TotalPrice", "mean"),
        )
        .reset_index()
    )

    value_features = customer_value.merge(
        avg_monthly,
        on="CustomerID",
        how="left",
    )

    return value_features[
        [
            "CustomerID",
            "TotalSpend",
            "AvgMonthlySpend",
            "AvgOrderValue",
        ]
    ]
