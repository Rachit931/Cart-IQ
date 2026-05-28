from sklearn.model_selection import GridSearchCV


def tune_random_forest(X_train, y_train):
    """
    Hyperparameter tuning
    for Random Forest
    """

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [
            100,
            200,
        ],
        "max_depth": [
            5,
            10,
            None,
        ],
        "min_samples_split": [
            2,
            5,
        ],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    print("\n----- RANDOM FOREST TUNING -----")

    print("\nBest Parameters:")

    print(grid_search.best_params_)

    print(f"\nBest F1 Score: " f"{grid_search.best_score_:.4F}")

    return grid_search.best_estimator_


def tune_xgboost(
    X_train,
    y_train,
):
    """
    Hyperparameter Tuning
    for XGBoost
    """

    from xgboost import XGBClassifier

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
    )

    param_grid = {
        "n_estimators": [
            100,
            200,
        ],
        "max_depth": [
            3,
            5,
            7,
        ],
        "learning_rate": [
            0.01,
            0.1,
        ],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    print("\n----- XGBOOST TUNING -----")

    print("\nBest Parameters:")

    print(grid_search.best_params_)

    print(f"\nBest F1 Score: " f"{grid_search.best_score_:.4f}")

    return grid_search.best_estimator_
