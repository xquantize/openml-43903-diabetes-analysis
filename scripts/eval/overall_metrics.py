from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)


def compute_overall_metrics(model, X_test, y_test):
    """Compute and return overall model performance metrics.

    Args:
        model (Pipeline): Trained model pipeline.
        X_test (pd.DataFrame): Features to test.
        y_test (pd.Series): Ground truth test labels.

    Returns:
        list[str]: Lines of formatted metrics for the report.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Brier Score": brier_score_loss(y_test, y_proba)
    }

    lines = ["OVERALL METRICS"]
    for k, v in metrics.items():
        lines.append(f"{k}: {v:.4f}")
    lines.append("")

    return lines
