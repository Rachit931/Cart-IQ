from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    model,
    X_test,
    y_test,
):
    """
    Evaluate classification model.

    Metrics:
    ----------
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - Confusion Matrix
    - Classification Report

    Parameters:
    ----------
    model:
        Trained classification model

    X_test:
        Test feature matrix

    y_test:
        Ground truth labels

    Returns:
    ----------
    None
    """

    # ==========================
    # PREDICTIONS
    # ==========================

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # ==========================
    # METRICS
    # ==========================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_prob)

    # ==========================
    # OUTPUT
    # ==========================

    print("\nMetrics:")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print(f"ROC-AUC  : {roc_auc:.4f}")

    # ==========================
    # CONFUSION MATRIX
    # ==========================

    print("\nConfusion Matrix:")

    print(confusion_matrix(y_test, y_pred))

    # ==========================
    # CLASSIFICATION REPORT
    # ==========================

    print("\nClassification Report:")

    print(classification_report(y_test, y_pred))
