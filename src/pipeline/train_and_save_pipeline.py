from src.models.train_classifier import train_xgboost
from src.pipeline.train_pipeline import train_pipeline
from src.utils.helpers import save_artifact, save_metadata


def train_and_save_pipeline():
    """
    Train final classifier
    and save artifacts.
    """

    # LOAD TRAINING DATA

    X_train, X_test, y_train, y_test, scaler = train_pipeline()

    # TRAIN FINAL MODEL

    model = train_xgboost(
        X_train,
        y_train,
    )

    # CREATING METADATA
    metadata = {
        "model_name": "XGBoost",
        "target": "Churn",
        "feature_count": len(X_train.columns),
        "feature_names": (X_train.columns.tolist()),
        "scaler": "StandardScaler",
        "model_parameters": (model.get_params()),
    }

    # SAVEL MODEL

    save_artifact(
        model,
        "artifacts/churn/models/xgb_model.pkl",
    )

    # SAVE SCALER

    save_artifact(
        scaler,
        "artifacts/churn/scalers/scaler.pkl",
    )

    # SAVE FEATURE COLUMNS

    save_artifact(
        X_train.columns.to_list(),
        "artifacts/churn/features/feature_columns.pkl",
    )

    # SAVE METADATA

    save_metadata(
        metadata,
        "artifacts/churn/metrics/metadta.json",
    )

    print("\n Training and saving complete. ")


if __name__ == "__main__":
    train_and_save_pipeline()
