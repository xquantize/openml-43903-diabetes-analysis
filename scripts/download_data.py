import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_openml_dataset

if __name__ == "__main__":
    load_openml_dataset(dataset_id=43903)
