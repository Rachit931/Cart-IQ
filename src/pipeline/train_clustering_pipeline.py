import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_processed_data


def train_clustering_pipeline():
    """
    Prepare dataset for customer clustering.

    Steps:
    --------
    1. Load processed dataset
    2. Prepare clustering features
    3. Scale features

    Returns:
    --------
    X
    scaler
    """

    # LOAD DATA

    df = load_processed_data()

    print(f"Loaded dataset shape: {df.shape}")

    # PREPARE CLUSTERING FEATURES

    cluster_features = [
        "Recency",
        "Frequency",
        "Monetary",
        "CustomerTenure",
        "UniqueProducts",
        "SpendGrowthRate",
    ]

    missing_features = set(cluster_features) - set(df.columns)

    if missing_features:
        raise ValueError(f"Missing clustering features: {sorted(missing_features)}")

    X = df[cluster_features].copy()

    print("\nFeature Matrix Shape:")
    print(f"X shape: {X.shape}")

    # FEATURE SCALING

    scaler = StandardScaler()

    X = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )

    print("\nScaled Feature Matrix:")
    print(f"X shape: {X.shape}")

    return (
        X,
        scaler,
    )


if __name__ == "__main__":
    train_clustering_pipeline()
