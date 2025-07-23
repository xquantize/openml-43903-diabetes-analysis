import os
import re
from datetime import datetime

import numpy as np
import pandas as pd


def generate_data_quality_report(
    df: pd.DataFrame,
    target_column: str = None,
    output_path: str = "journal"
):
    """Generates a detailed data quality report for the given DataFrame.
    Outputs a plain text file with diagnostics such as missing values,
    outliers, data types, duplicates, and categorical summaries.

    Args:
        df (pd.DataFrame): The input dataset.
        target_column (str, optional): Name of the target column to evaluate.
        output_path (str): Directory where the report will be saved.
    """
    report_lines = []

    report_lines.append("DATA QUALITY REPORT")
    report_lines.append(f"Generated on: {datetime.today().strftime('%Y-%m-%d')}")
    report_lines.append("")

    report_lines.append("BASIC STRUCTURE")
    report_lines.append(f"Shape: {df.shape}")
    report_lines.append(f"Columns: {list(df.columns)}")
    report_lines.append("")

    report_lines.append("DATA TYPES")
    report_lines.append(str(df.dtypes))
    report_lines.append("")

    missing = df.isnull().sum()
    total_missing = missing.sum()
    report_lines.append("MISSING VALUES")
    if total_missing > 0:
        missing_percent = (missing / len(df)) * 100
        missing_report = pd.DataFrame({
            'missing_count': missing,
            'missing_percent': missing_percent
        })
        missing_report = missing_report[missing_report['missing_count'] > 0]
        report_lines.append(missing_report.to_string())
    else:
        report_lines.append("No missing values found.")
    report_lines.append("")

    dup_count = df.duplicated().sum()
    report_lines.append("DUPLICATE ROWS")
    report_lines.append(f"{dup_count} duplicate rows found.")
    report_lines.append("")

    report_lines.append("UNIQUE VALUES PER COLUMN")
    report_lines.append(str(df.nunique()))
    report_lines.append("")

    report_lines.append("NUMERICAL DESCRIPTIVE STATISTICS")
    numeric_stats = df.select_dtypes(include=[np.number]).describe().T
    report_lines.append(numeric_stats.to_string())
    report_lines.append("")

    report_lines.append("HIGH CARDINALITY CATEGORICAL FEATURES")
    high_cardinality = df.select_dtypes(include=["object"]).nunique()
    high_cardinality = high_cardinality[high_cardinality > 50]
    if not high_cardinality.empty:
        report_lines.append(high_cardinality.to_string())
    else:
        report_lines.append("No high cardinality object features found.")
    report_lines.append("")

    if target_column and target_column in df.columns:
        report_lines.append(f"TARGET DISTRIBUTION: {target_column}")
        target_counts = df[target_column].value_counts()
        report_lines.append(str(target_counts))
        imbalance_ratio = target_counts.min() / target_counts.max()
        report_lines.append(f"Imbalance ratio (minority/majority): {imbalance_ratio:.4f}")
        report_lines.append("")

    constant_features = df.columns[df.nunique() <= 1]
    report_lines.append("CONSTANT FEATURES")
    if len(constant_features) > 0:
        report_lines.append(f"{len(constant_features)} constant features found: {list(constant_features)}")
    else:
        report_lines.append("No constant features found.")
    report_lines.append("")

    report_lines.append("POTENTIAL OUTLIERS (NUMERICAL RANGES)")
    for col in df.select_dtypes(include=[np.number]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        report_lines.append(f"{col}: {outliers} outliers")
    report_lines.append("")

    object_cols = df.select_dtypes(include="object")
    for col in object_cols.columns:
        report_lines.append(f"CATEGORICAL SUMMARY: {col}")
        report_lines.append(str(df[col].value_counts(dropna=False).head(10)))
        report_lines.append("")

    bool_cols = df.select_dtypes(include='bool')
    for col in bool_cols.columns:
        report_lines.append(f"BOOLEAN SUMMARY: {col}")
        report_lines.append(str(df[col].value_counts()))
        report_lines.append("")

    for col in df.columns:
        unique_types = df[col].dropna().map(type).unique()
        if len(unique_types) > 1:
            report_lines.append(f"MIXED TYPES DETECTED IN: {col} → {unique_types}")
    report_lines.append("")

    for col in object_cols.columns:
        dirty = df[col].astype(str).str.contains(r"\?|unknown|missing", flags=re.IGNORECASE, na=False)
        if dirty.any():
            report_lines.append(f"POSSIBLE DIRTY VALUES DETECTED IN: {col}")
    report_lines.append("")

    numeric_corr = df.select_dtypes(include=np.number).corr()
    corr_pairs = numeric_corr.unstack().dropna()
    strong_corr = corr_pairs[(abs(corr_pairs) < 1.0)].sort_values(ascending=False)
    report_lines.append("CORRELATION MATRIX (Top 5 strongest non-1.0 pairs)")
    report_lines.append(str(strong_corr.head(5)))
    report_lines.append("")

    os.makedirs(output_path, exist_ok=True)
    filename = f"data_profile_{datetime.today().strftime('%Y-%m-%d')}.txt"
    full_path = os.path.join(output_path, filename)

    with open(full_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Data quality report written to {full_path}")
