import os
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def evaluate_calibration(y_true, y_proba, save_path):
    """Evaluate and plot a probability calibration curve.

    Parameters:
    - y_true (array-like): Ground truth binary labels.
    - y_proba (array-like): Predicted probabilities for the positive class.
    - save_path (str): Path to save the calibration plot (e.g., PNG file).

    Returns:
    - lines (list of str): Human-readable report lines describing the output.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Probability Calibration Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    return [f"Calibration curve saved to: {save_path}"]
