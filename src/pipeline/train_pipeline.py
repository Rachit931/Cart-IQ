from src.data.load_data import load_processed_data
from src.targets.churn import create_churn_table


def train_pipeline(test_size: float = 0.2):
    """
    Build train/test datasets
    for churn classification.

    Steps:
    ----------
    1. Load processed dataset
    2. Create churn labels
    3. Sort customers chronologically
    4. Perform time-based split
    5. Prepare train/test sets

    Parameters:
    ----------
    test_size: float
        Fraction of data used
        for testing

    Returns:
    ----------
    X_train, X_test,
    y_train, y_test
    """

    # =================================
    # LOAD PROCESSED DATASET
    # =================================

    df = load_processed_data()

    print(f"Loaded dataset shape: " f"{df.shape}")

    # =================================
    # CREATE CHURN LABELS
    # =================================

    df = create_churn_table(
        customer_features=df,
        churn_threshold=60,
    )

    print("\nChurn Distribution:")

    print(df["Churn"].value_counts(normalize=True))

    # =================================
    # SORT CHRONOLOGICALLY
    # =================================

    df = df.sort_values(by="LastPurchaseDate")

    # =================================
    # TIME-BASED SPLIT
    # =================================

    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index]

    test_df = df.iloc[split_index:]

    print("\nTrain/Test Split:")

    print(f"Train size: {train_df.shape}")

    print(f"Test size : {test_df.shape}")

    # =================================
    # PREPARE FEATURES + TARGET
    # =================================

    drop_columns = [
        "CustomerID",
        "Churn",
        "LastPurchaseDate",
    ]

    X_train = train_df.drop(columns=drop_columns)

    y_train = train_df["Churn"]

    X_test = test_df.drop(columns=drop_columns)

    y_test = test_df["Churn"]

    print("\nFeature Shapes:")

    print(f"X_train: {X_train.shape}")

    print(f"X_test : {X_test.shape}")

    return (X_train, X_test, y_train, y_test)


if __name__ == "__main__":

    train_pipeline()
