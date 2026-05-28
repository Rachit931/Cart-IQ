import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_processed_data
from src.targets.churn import create_churn_table


def train_pipeline(
    test_size: float = 0.2,
):
    """
    Prepare train/test datasets
    for churn classification.

    Steps:
    ----------
    1. Load processed dataset
    2. Create churn labels
    3. Prepare features + target
    4. Stratified train/test split

    Parameters:
    ----------
    test_size: float
        Fraction of dataset used
        for testing

    Returns:
    ----------
    X_train
    X_test
    y_train
    y_test
    """

    # LOAD DATA

    df = load_processed_data()

    print(f"Loaded dataset shape: " f"{df.shape}")

    # CREATE CHURN LABELS

    df = create_churn_table(
        customer_features=df,
        churn_threshold=60,
    )

    print("\nChurn Distribution:")

    print(df["Churn"].value_counts(normalize=True))

    # PREPARE FEATURES + TARGET

    drop_columns = [
        "CustomerID",
        "Churn",
        "LastPurchaseDate",
        "Recency",
    ]

    X = df.drop(columns=drop_columns)

    y = df["Churn"]

    print("\nFeature Matrix Shape:")

    print(f"X shape: {X.shape}")

    print(f"y shape: {y.shape}")

    # STRATIFIED SPLIT

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=42,
    )

    # FEATURE SCALING

    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
    )

    print("\nTrain/Test Split:")

    print(f"X_train: {X_train.shape}")

    print(f"X_test : {X_test.shape}")

    print(f"y_train:\n" f"{y_train.value_counts(normalize=True)}")

    print(f"\ny_test:\n" f"{y_test.value_counts(normalize=True)}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
    )


if __name__ == "__main__":

    train_pipeline()
