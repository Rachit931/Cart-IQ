from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_logistic_regression(
    x_train,
    y_train,
):
    """
    Train Logistic Regression classifier.

    Parameters:
    ----------
    X_train:
        Training feature matrix

    y_train:
        Training labels

    Returns:
    ----------
    Trained LogisticRegression model
    """

    model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(
        x_train,
        y_train,
    )

    return model


def train_random_forest(x_train, y_train):
    """
    Train Random Forest Classifier.

    Parameters:
    -----------
    x_train:
        Training feature matrix

    y_train:
        Training labels

    Returns:
    --------
    Trained RandomForestClassifier
    """

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    model.fit(x_train, y_train)

    return model


def train_xgboost(
    x_train,
    y_train,
):
    """
    Train XGBoost classifier.

    Parameters:
    ----------
    X_train:
        Training feature matrix

    y_train:
        Training labels

    Returns:
    ----------
    Trained XGBClassifier
    """

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(x_train, y_train)

    return model
