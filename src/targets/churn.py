import pandas as pd


def create_churn_table(
    customer_features: pd.DataFrame,
    churn_threshold: int = 60,
) -> pd.DataFrame:
    """ "
    Create churm labels using inactivity threshold.

    Parameters:
    -----------
    customer_features: pd.DataFrame
        Customer-level features dataset

    churtn_threshold: int
        Days of inactivity requred to mark
        a customer as churned

    Returns
    -------
    pd.DataFrame
    """

    df = customer_features.copy()

    df["Churn"] = (df["Recency"] > churn_threshold).astype(int)

    return df
