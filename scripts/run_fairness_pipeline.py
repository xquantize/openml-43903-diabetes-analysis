import os
from datetime import datetime

from data.load_and_preprocess import load_and_preprocess
from model.train import train_model
from eval.overall_metrics import compute_overall_metrics
from eval.calibration import evaluate_calibration
from eval.fairness_by_group import evaluate_fairness_by_group
from reports.writer import write_report, save_calibration_plot


DATA_PATH = "data/raw/diabetes-130-hospitals_(fairlearn)_2025-07-23.csv"
TARGET = "readmit_30_days"
SENSITIVE_FEATURES = ["race", "gender", "age", "medicare", "medicaid"]
REPORT_DIR = "journal"
PLOT_DIR = os.path.join(REPORT_DIR, "plots")
REPORT_PATH = os.path.join(REPORT_DIR, f"fairness_model_eval_{datetime.today().strftime('%Y-%m-%d')}.txt")
CALIBRATION_PLOT_PATH = os.path.join(PLOT_DIR, "calibration_curve.png")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def main():
    df, X_train, X_test, y_train, y_test, sens_train, sens_test, preprocessor = load_and_preprocess(
        DATA_PATH, TARGET, SENSITIVE_FEATURES
    )

    model = train_model(X_train, y_train, preprocessor)

    report_lines = []
    report_lines += compute_overall_metrics(model, X_test, y_test)
    report_lines += evaluate_fairness_by_group(model, X_test, y_test, sens_test, SENSITIVE_FEATURES)

    y_proba = model.predict_proba(X_test)[:, 1]

    report_lines += evaluate_calibration(y_test, y_proba, CALIBRATION_PLOT_PATH)

    save_calibration_plot(y_test, y_proba, CALIBRATION_PLOT_PATH)

    write_report(report_lines, REPORT_PATH)

    print(f"Report written to {REPORT_PATH}")
    print(f"Calibration plot saved to {CALIBRATION_PLOT_PATH}")


if __name__ == "__main__":
    main()
