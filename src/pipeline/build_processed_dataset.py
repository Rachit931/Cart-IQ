from pathlib import Path

from src.data.clean_data import clean_data
from src.data.load_data import load_raw_data
from src.pipeline.build_features import build_all_features

PROCESSED_DATA_PATH = Path("data/processed/customer_features.csv")


def build_processed_dataset():
    """
    Full ETL pipeline.

    Steps:
    ----------
    1. Load raw dataset
    2. Clean transaction data
    3. Build customer-level features
    4. Save processed dataset

    Returns:
    ----------
    pd.DataFrame
    """

    # Load raw data
    raw_df = load_raw_data()

    # Clean data
    clean_df = clean_data(raw_df)

    # Build features
    customer_features = build_all_features(clean_df)

    # Create processed directory if missing
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save dataset
    customer_features.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed dataset saved to:" f" {PROCESSED_DATA_PATH}")

    print(f"Dataset shape:" f" {customer_features.shape}")

    return customer_features


if __name__ == "__main__":
    build_processed_dataset()
