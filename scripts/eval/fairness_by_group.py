from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

import pandas as pd


def evaluate_fairness_by_group(model, X_test, y_test, sens_df, sensitive_features):
    """Evaluate model performance and fairness across sensitive groups.

    Args:
        model: Trained classifier.
        X_test: Feature data.
        y_test: Ground truth labels.
        sens_df: Sensitive attribute DataFrame aligned with X_test.
        sensitive_features: List of sensitive feature column names.

    Returns:
        List of string lines for the report.
    """
    lines = []
    y_pred = model.predict(X_test)

    metrics_dict = {
        "accuracy": accuracy_score,
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0)
    }

    for feature in sensitive_features:
        lines.append(f"METRIC BREAKDOWN BY {feature.upper()}")
        lines.append("-" * 40)

        group_counts = sens_df[feature].value_counts().to_dict()
        positive_rates = (
            pd.concat([sens_df[feature], y_test], axis=1)
            .groupby(feature)[y_test.name]
            .mean()
            .round(4)
            .to_dict()
        )

        lines.append(f"Sample counts by {feature}: {group_counts}")
        lines.append(f"Positive class rate by {feature}: {positive_rates}")

        for group, rate in positive_rates.items():
            if rate < 0.01 or rate > 0.99:
                lines.append(f"\u26a0\ufe0f  Extreme imbalance in group '{group}' for '{feature}': {rate:.2f}")

        metric_frame = MetricFrame(
            metrics=metrics_dict,
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sens_df[feature]
        )

        lines.append(metric_frame.by_group.to_string())

        accs = metric_frame.by_group["accuracy"]
        disparity = accs.min() / accs.max() if accs.max() > 0 else 0
        worst_group = accs.idxmin()
        best_group = accs.idxmax()
        lines.append(f"\nDisparity (min/max accuracy): {disparity:.4f}")
        lines.append(f"Worst-performing group (accuracy): {worst_group} = {accs[worst_group]:.4f}")
        lines.append(f"Best-performing group (accuracy): {best_group} = {accs[best_group]:.4f}")

        dp = demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sens_df[feature]
        )

        eo = equalized_odds_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sens_df[feature]
        )

        lines.append(f"Demographic Parity Difference ({feature}): {dp:.4f}")
        lines.append(f"Equalized Odds Difference ({feature}): {eo:.4f}\n")

    return lines
