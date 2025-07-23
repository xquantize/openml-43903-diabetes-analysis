import os
import sys

import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.calibration import calibration_curve

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix
)

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference
)

from fairlearn.postprocessing import ThresholdOptimizer

import matplotlib.pyplot as plt
import seaborn as sns


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_PATH = "data/raw/diabetes-130-hospitals_(fairlearn)_2025-07-23.csv"
REPORT_DIR = "journal"
REPORT_PATH = os.path.join(REPORT_DIR, f"fairness_model_eval_{datetime.today().strftime('%Y-%m-%d')}.txt")
PLOT_DIR = os.path.join(REPORT_DIR, "plots")

TARGET = "readmit_30_days"
SENSITIVE_FEATURES = ["race", "gender", "age", "medicare", "medicaid"]

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def load_data(path):
    """Load dataset from the specified path."""
    return pd.read_csv(path)


def preprocess_data(df, target):
    """Preprocess the dataset: encode categorical, separate features and labels."""
    df_clean = df.copy()
    y = df_clean[target].astype(int)
    X = df_clean.drop(columns=[target])
    sensitive = df_clean[SENSITIVE_FEATURES]

    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.difference(SENSITIVE_FEATURES)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    return X, y, sensitive, preprocessor


def train_model(X_train, y_train, preprocessor):
    """Train logistic regression with balanced class weights."""
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(solver="liblinear", class_weight="balanced"))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def plot_probability_calibration(y_true, y_prob, path):
    """Plot probability calibration curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Probability Calibration Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def evaluate_fairness(model, X_test, y_test, sensitive_df):
    """Evaluate fairness metrics, probability calibration, and disparity metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report_lines = []

    report_lines.append("FAIRNESS MODEL EVALUATION REPORT")
    report_lines.append("=" * 40)
    report_lines.append(f"Evaluated on {len(y_test)} samples\n")

    overall_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Brier Score": brier_score_loss(y_test, y_proba)
    }

    report_lines.append("OVERALL METRICS")
    for k, v in overall_metrics.items():
        report_lines.append(f"{k}: {v:.4f}")
    report_lines.append("")

    metrics_dict = {
        "accuracy": accuracy_score,
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0)
    }

    for feature in SENSITIVE_FEATURES:
        report_lines.append(f"METRIC BREAKDOWN BY {feature.upper()}")
        report_lines.append("-" * 40)

        metric_frame = MetricFrame(
            metrics=metrics_dict,
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_df[feature]
        )

        report_lines.append(metric_frame.by_group.to_string())
        accs = metric_frame.by_group["accuracy"]
        disparity = accs.min() / accs.max() if accs.max() > 0 else 0
        report_lines.append(f"\nDisparity (min/max accuracy): {disparity:.4f}\n")

        dp = demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_df[feature]
        )
        eo = equalized_odds_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_df[feature]
        )

        report_lines.append(f"Demographic Parity Difference ({feature}): {dp:.4f}")
        report_lines.append(f"Equalized Odds Difference ({feature}): {eo:.4f}\n")

    plot_probability_calibration(y_test, y_proba, os.path.join(PLOT_DIR, "calibration_curve.png"))

    return report_lines


def main():
    df = load_data(DATA_PATH)
    X, y, sensitive_df, preprocessor = preprocess_data(df, TARGET)
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_df, test_size=0.2, stratify=y, random_state=42
    )

    model = train_model(X_train, y_train, preprocessor)
    report = evaluate_fairness(model, X_test, y_test, sens_test)

    print("\n".join(report))
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report))

    print(f"\nReport saved to {REPORT_PATH}")
    print(f"Calibration curve saved to {os.path.join(PLOT_DIR, 'calibration_curve.png')}")


if __name__ == "__main__":
    main()
