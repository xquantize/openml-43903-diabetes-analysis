import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_PATH = "data/raw/diabetes-130-hospitals_(fairlearn)_2025-07-23.csv"
TARGET = "readmit_30_days"
SENSITIVE_FEATURES = ['race', 'gender', 'age', 'medicare', 'medicaid']
LAB_FEATURES = ['A1Cresult', 'max_glu_serum']
REPORT_DIR = "journal/fairness"
JOURNAL_PATH = f"journal/fairness_summary_{datetime.today().strftime('%Y-%m-%d')}.txt"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs("journal", exist_ok=True)

df = pd.read_csv(DATA_PATH)
report_lines = []


def add_section(title: str):
    """Adds a section title and underline to the report.

    Parameters
    ----------
    title : str
        The title of the section to be added to the report.
    """
    report_lines.append("\n" + title)
    report_lines.append("=" * len(title))


def print_and_log(text: str):
    """Prints and appends a line of text to the report.

    Parameters
    ----------
    text : str
        The string to be printed and logged.
    """
    print(text)
    report_lines.append(text)


def plot_readmission_rate_by_race(df: pd.DataFrame, target_col: str, save_path: str):
    """Creates a dual-axis horizontal bar plot showing readmission rates and sample counts by race.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    target_col : str
        The name of the binary target column.
    save_path : str
        The path to save the resulting plot image.
    """
    group = df.groupby("race")[target_col].agg(['mean', 'count']).sort_values("mean")
    colors = sns.color_palette("viridis", len(group))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twiny()

    bars = ax1.barh(group.index, group['mean'], color=colors, height=0.4)
    ax1.set_xlabel("Readmission Rate (<30 Days)")
    ax1.set_xlim(0, max(group['mean']) + 0.02)
    ax1.set_title("Readmission Rate by Race (with Sample Count Overlay)")

    ax2.plot(group['count'], group.index, "o-", color="grey")
    ax2.set_xlabel("Sample Count")
    ax2.grid(True, axis='x', linestyle='--', alpha=0.3)

    for i, (rate, count) in enumerate(zip(group['mean'], group['count'])):
        ax1.text(rate + 0.003, i, f"{rate:.3f}", va='center', fontsize=9)
        ax2.text(count + 200, i, str(count), va='center', fontsize=8, color="grey")

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_missing_a1c_by_race(df: pd.DataFrame, save_path: str):
    """Plots the proportion of missing A1Cresult values by racial group.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    save_path : str
        Path to save the generated bar plot.
    """
    miss = df.groupby("race")["A1Cresult"].apply(lambda x: x.isna().mean()).sort_values()
    colors = sns.color_palette("crest", len(miss))

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=miss.values, y=miss.index, palette=colors)
    ax.set_title("Missing A1Cresult by Race", fontsize=13)
    ax.set_xlabel("Fraction Missing")
    ax.set_xlim(0, max(miss.values) + 0.05)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for i, v in enumerate(miss.values):
        ax.text(v + 0.01, i, f"{v:.2%}", color="black", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


add_section("FAIRNESS PRE-CHECK REPORT")
print_and_log(f"Dataset shape: {df.shape}")
print_and_log(f"Target column: {TARGET}")
print_and_log(f"Sensitive features: {SENSITIVE_FEATURES}\n")

overall_rate = df[TARGET].mean()
print_and_log(f"Overall readmission rate (<30 days): {overall_rate:.4f}")

add_section("READMISSION RATES BY GROUP")
for feature in SENSITIVE_FEATURES:
    print_and_log(f"\n--- {feature.upper()} ---")
    grouped = df.groupby(feature)[TARGET].agg(['count', 'mean']).rename(columns={"mean": "readmit_rate"})
    print_and_log(str(grouped))

plot_readmission_rate_by_race(df, TARGET, os.path.join(REPORT_DIR, "readmission_rate_by_race.png"))

add_section("MISSINGNESS BY GROUP")
for lab in LAB_FEATURES:
    print_and_log(f"\n--- {lab.upper()} ---")
    for feature in SENSITIVE_FEATURES:
        grouped_missing = df.groupby(feature)[lab].apply(lambda x: x.isna().mean()).sort_index()
        print_and_log(f"\nMissing {lab} by {feature}:")
        print_and_log(str(grouped_missing))

plot_missing_a1c_by_race(df, os.path.join(REPORT_DIR, "missing_a1c_by_race.png"))

add_section("DISPARITY RATIOS (min / max readmit rate)")
for feature in SENSITIVE_FEATURES:
    rates = df.groupby(feature)[TARGET].mean()
    ratio = rates.min() / rates.max() if rates.max() > 0 else 0
    print_and_log(f"{feature}: {ratio:.4f}")

add_section("SMALL GROUPS (count < 100)")
for feature in SENSITIVE_FEATURES:
    counts = df[feature].value_counts()
    small = counts[counts < 100]
    if not small.empty:
        print_and_log(f"\n{feature}:")
        print_and_log(str(small))
    else:
        print_and_log(f"\n{feature}: No small groups")

with open(JOURNAL_PATH, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nFairness plots saved to: {os.path.abspath(REPORT_DIR)}")
print(f"Text summary saved to: {JOURNAL_PATH}")
