import os
import zipfile
import numpy as np
import pandas as pd

DATASET_ARCHIVE = "dataset.zip"
DATASET_DIRECTORY = "dataset"
TARGET_COLUMN = "Close"
SPLIT_RATIO = 0.8

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATASET_ARCHIVE_PATH = os.path.join(SCRIPT_DIRECTORY, DATASET_ARCHIVE)
DATASET_EXTRACT_PATH = os.path.join(SCRIPT_DIRECTORY, DATASET_DIRECTORY)

INFY_STOCK_CSV_PATH = os.path.join(DATASET_EXTRACT_PATH, "infy_stock.csv")
NIFTY_STOCK_CSV_PATH = os.path.join(DATASET_EXTRACT_PATH, "nifty_it_index.csv")
TCS_STOCK_CSV_PATH = os.path.join(DATASET_EXTRACT_PATH, "tcs_stock.csv")


def extract_dataset(archive_path, extract_path):
    """Extract dataset from zip archive to the specified directory."""
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"Dataset extracted to: {extract_path}")


def load_dataset(csv_path, target_column):
    """Load dataset from CSV and separate features and target column."""
    data = pd.read_csv(csv_path)
    x_data = data.drop(columns=[target_column]).values
    y_data = data[target_column].values
    return x_data, y_data


def get_dataset_b(csv_path=INFY_STOCK_CSV_PATH, target_column=TARGET_COLUMN):
    """Load dataset B."""
    extract_dataset(DATASET_ARCHIVE_PATH, DATASET_EXTRACT_PATH)
    return load_dataset(csv_path, target_column)


def get_dataset_a():
    """Generate dataset A with a polynomial relation and some noise."""
    x_data = np.linspace(-1, 1, 101)
    num_coef = 3
    coef = [-10, 2, 3]
    y_data = 0

    for i in range(num_coef):
        y_data += coef[i] * np.power(x_data, i)

    y_data += np.random.randn(*x_data.shape) * 1.5

    return x_data, y_data


def get_dataset(variant="a"):
    """Return dataset based on the variant selected."""
    if variant == "a":
        return get_dataset_a()
    elif variant == "b":
        return get_dataset_b()
    else:
        raise ValueError(f"Invalid dataset variant: {variant}")


def split_dataset(x_data, y_data):
    """Split dataset into train and test based on the split ratio."""
    split_index = int(len(x_data) * SPLIT_RATIO)

    x_train_data = x_data[:split_index]
    y_train_data = y_data[:split_index]
    x_test_data = x_data[split_index:]
    y_test_data = y_data[split_index:]

    print("x_train:", len(x_train_data))
    print("x_test:", len(x_test_data))

    return x_train_data, y_train_data, x_test_data, y_test_data


def shuffle_dataset(x_data, y_data):
    """Shuffle dataset by examples, keeping labels and targets in sync."""
    indices = np.random.permutation(len(x_data))
    return x_data[indices], y_data[indices]


def prepare_dataset(x_data, y_data):
    """Shuffle and split dataset."""
    x_data, y_data = shuffle_dataset(x_data, y_data)
    return split_dataset(x_data, y_data)


if __name__ == "__main__":
    x, y = get_dataset(variant="b")
    x_train, y_train, x_test, y_test = prepare_dataset(x, y)
