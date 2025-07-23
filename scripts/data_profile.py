import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_quality import generate_data_quality_report


if __name__ == "__main__":
    DATA_PATH = "data/raw/diabetes-130-hospitals_(fairlearn)_2025-07-23.csv"
    TARGET = "readmit_30_days"

    df = pd.read_csv(DATA_PATH)
    generate_data_quality_report(df, target_column=TARGET)
