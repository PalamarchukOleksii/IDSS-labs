import os
import zipfile
import numpy as np
import pandas as pd

DATASET_ARCHIVE = "dataset.zip"
DATASET_DIRECTORY = "dataset"

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATASET_ARCHIVE_PATH = os.path.join(SCRIPT_DIRECTORY, DATASET_ARCHIVE)
DATASET_EXTRACT_PATH = os.path.join(SCRIPT_DIRECTORY, DATASET_DIRECTORY)

TARGET_COLUMN = "Close"
INFY_STOCK_CSV_PATH = os.path.join(DATASET_EXTRACT_PATH, "infy_stock.csv")
NIFTY_STOCK_CSV_PATH = os.path.join(DATASET_EXTRACT_PATH, "nifty_it_index.csv")
TCS_STOCK_CSV_PATH = os.path.join(DATASET_EXTRACT_PATH, "tcs_stock.csv")

DATASET_VARIANT = "a"


def extract_dataset(archive_path, extract_path):
    print("Starting to unpack the archive...")
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    with zipfile.ZipFile(archive_path) as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"The archive has been successfully unpacked to {extract_path}")


def load_dataset(csv_path, target_column):
    data = pd.read_csv(csv_path)
    x_data = data.drop(columns=[target_column]).values
    y_data = data[target_column].values

    return x_data, y_data


def get_dataset_b(
    csv_path=INFY_STOCK_CSV_PATH,
    target_column=TARGET_COLUMN,
    archive_path=DATASET_ARCHIVE_PATH,
    extract_path=DATASET_EXTRACT_PATH,
):
    extract_dataset(archive_path, extract_path)
    x_data, y_data = load_dataset(csv_path, target_column)

    return x_data, y_data


def get_dataset_a():
    x_data = np.linspace(-1, 1, 101)
    num_coef = 3
    coef = [-10, 2, 3]
    y_data = 0

    for i in range(num_coef):
        y_data += coef[i] * np.power(x_data, i)

    y_data += np.random.randn(*x_data.shape) * 1.5

    return x_data, y_data


def get_dataset(
    variant="a",
    csv_path=INFY_STOCK_CSV_PATH,
    target_column=TARGET_COLUMN,
    archive_path=DATASET_ARCHIVE_PATH,
    extract_path=DATASET_EXTRACT_PATH,
):
    if variant == "a":
        return get_dataset_a()
    elif variant == "b":
        return get_dataset_b(csv_path, target_column, archive_path, extract_path)
    else:
        print(f"No dataset for varinat: {variant}")


if __name__ == "__main__":
    csv_data_path = os.path.join(DATASET_EXTRACT_PATH, INFY_STOCK_CSV_PATH)

    x, y = get_dataset_b(csv_data_path)

    print(x, y)
