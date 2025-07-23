import os
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def write_report(report_lines, report_path):
    """Writes the list of report strings to a file.

    Parameters:
    - report_lines (list of str): Lines to write.
    - report_path (str): Path to save the report file.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))


def save_calibration_plot(y_true, y_prob, plot_path):
    """Generates and saves a probability calibration curve.

    Parameters:
    - y_true (array-like): Ground truth binary labels.
    - y_prob (array-like): Predicted probabilities.
    - plot_path (str): File path to save the calibration plot.
    """
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Probability Calibration Curve')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
