from functools import reduce

import pandas as pd

from src.features.behavior import build_behavior_features
from src.features.product import build_product_features
from src.features.rfm import build_rfm_features
from src.features.time import build_time_features
from src.features.trend import build_trend_features
from src.features.value import build_value_features


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build complete customer-level
    feature dataset.

    Feature Groups:
    ----------
    - RFM features
    - Value features
    - Behavioral features
    - Product diversity features
    - Temporal features
    - Trend features

    Parameters:
    ----------
    - df:
        Cleaned transaction-level DataFrame

    Returns:
    ----------
    - Final merged customer-level
      feature DataFrame
    """

    feature_dfs = [
        build_rfm_features(df),
        build_value_features(df),
        build_behavior_features(df),
        build_product_features(df),
        build_time_features(df),
        build_trend_features(df),
    ]

    customer_features = reduce(
        lambda left, right: pd.merge(left, right, on="CustomerID", how="outer"),
        feature_dfs,
    )

    customer_features = customer_features.fillna(0)

    return customer_features
