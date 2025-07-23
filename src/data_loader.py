import openml
import pandas as pd
import os
from datetime import datetime


def load_openml_dataset(dataset_id: int = 43903, save_path: str = "data/raw") -> pd.DataFrame:
    """Download and save an OpenML dataset, with metadata printout and versioning.

    Args:
        dataset_id (int): The OpenML dataset ID.
        save_path (str): Local directory to save the dataset CSV.

    Returns:
        pd.DataFrame: The full dataset as a DataFrame.
    """
    print(f"Connecting to OpenML and downloading dataset ID: {dataset_id}")
    dataset = openml.datasets.get_dataset(dataset_id)

    print(f"Dataset name: {dataset.name}")
    print(f"Dataset ID: {dataset.dataset_id}")
    print(f"Version: {dataset.version}")
    print(f"Default target attribute: {dataset.default_target_attribute}")
    print(f"Number of features: {len(dataset.features)}")

    X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format='dataframe'
    )

    df = X.copy()
    df[dataset.default_target_attribute] = y

    print(f"Dataset loaded with shape: {df.shape}")
    print("Preview:")
    print(df.head(3))

    os.makedirs(save_path, exist_ok=True)
    filename = f"{dataset.name.replace(' ', '_').lower()}_{datetime.today().strftime('%Y-%m-%d')}.csv"
    filepath = os.path.join(save_path, filename)
    df.to_csv(filepath, index=False)

    print(f"Dataset saved to: {filepath}")

    return df
