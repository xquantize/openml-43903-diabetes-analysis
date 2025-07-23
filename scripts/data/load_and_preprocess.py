import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = "data/raw/diabetes-130-hospitals_(fairlearn)_2025-07-23.csv"
TARGET = "readmit_30_days"
SENSITIVE_FEATURES = ["race", "gender", "age", "medicare", "medicaid"]


def load_and_preprocess(data_path, target_col, sensitive_features):
    """Loads the dataset, processes categorical and numerical features,
    splits the data into train/test sets, and builds a preprocessor.

    Args:
        data_path (str): Path to the dataset CSV.
        target_col (str): Name of the target column.
        sensitive_features (list): List of sensitive feature column names.

    Returns:
        df (pd.DataFrame): Full original dataframe.
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Testing feature set.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        sens_train (pd.DataFrame): Sensitive features for training set.
        sens_test (pd.DataFrame): Sensitive features for test set.
        preprocessor (ColumnTransformer): Preprocessing transformer for pipeline.
    """
    df = pd.read_csv(data_path)

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    sensitive = df[sensitive_features]

    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.difference(sensitive_features)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.2, stratify=y, random_state=42
    )

    return df, X_train, X_test, y_train, y_test, sens_train, sens_test, preprocessor
