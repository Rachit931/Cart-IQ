import numpy as np
from sklearn.model_selection import cross_val_score


def perform_cross_validation(
    model,
    X_train,
    y_train,
    cv: int = 5,
    scoring: str = "f1",
):
    """
    Perform cross-validation
    on classification models.

    Parameters:
    -----------
    model:
        ML model

    X_train:
        training features

    y_train:
        training labels

    cv: int
        number of folds

    scoring: str
        evaluation metric

    Returns:
    ----------
    scores
    """

    scores = cross_val_score(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
    )

    print("\nCross Validation Scores: ")

    print(scores)

    print(f"\nMean {scoring}: " f"{np.mean(scores):.4f}")

    print(f"Std {scoring}: " f"{np.std(scores):.4f}")

    return scores
