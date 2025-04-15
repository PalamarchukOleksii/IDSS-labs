import os
import pandas as pd
import pickle
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleDataset:
    def __init__(self, dataset_name: str, download_dir: str = "dataset"):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.__dataset_name = dataset_name
        self.__download_path = os.path.join(script_dir, download_dir, dataset_name)
        self.__api = KaggleApi()

        self.__api.authenticate()

    def download(self, unzip: bool = True):
        if (
            os.listdir(self.__download_path)
            if os.path.exists(self.__download_path)
            else []
        ):
            print(f"Dataset already exists at: {self.__download_path}")
            return

        if not os.path.exists(self.__download_path):
            os.makedirs(self.__download_path)

        print(f"Downloading dataset: {self.__dataset_name}")
        print(f"Saving to: {self.__download_path}")
        self.__api.dataset_download_files(
            dataset=self.__dataset_name,
            path=self.__download_path,
            unzip=unzip,
        )
        print("Download complete.")

    def list_files(self):
        files = []
        for root, _, filenames in os.walk(self.__download_path):
            for f in filenames:
                files.append(os.path.join(root, f))
        return files

    def get_data(
        self, filename: str, data_type: str = "train", labels_column: str = "labels"
    ) -> tuple[np.ndarray, np.ndarray]:
        if not filename:
            file_extensions = [".pickle", ".csv"]
            for ext in file_extensions:
                potential_filename = f"{data_type}{ext}"
                file_path = os.path.join(self.__download_path, potential_filename)
                if os.path.exists(file_path):
                    filename = potential_filename
                    break
            else:
                raise FileNotFoundError(
                    f"No file found with name {data_type} in {self.__download_path}."
                )

        file_path = os.path.join(self.__download_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File {filename} not found in {self.__download_path}"
            )

        file_extension = filename.split(".")[-1]

        if file_extension == "csv":
            return self.__get_csv_data(file_path, labels_column)
        elif file_extension == "pickle":
            return self.__get_pickle_data(file_path, data_type)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def __get_csv_data(
        self, file_path: str, labels_column: str = "labels"
    ) -> tuple[np.ndarray, np.ndarray]:
        print(f"Loading CSV data from {file_path}")

        df = pd.read_csv(file_path)

        if labels_column in df.columns:
            x_data = df.drop(columns=[labels_column])
            y_data = df[[labels_column]]

            if len(x_data.shape) == 2:
                height = int(np.sqrt(x_data.shape[1]))
                x_data = x_data.values.reshape(x_data.shape[0], height, height, 1)
            else:
                raise ValueError("Unsupported data shape for CNN.")

            return x_data, y_data
        else:
            raise KeyError(f"Label column '{labels_column}' not found in the CSV file.")

    def __get_pickle_data(
        self, file_path: str, data_type: str = "train"
    ) -> tuple[np.ndarray, np.ndarray]:
        print(f"Loading Pickle data from {file_path}")
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            x_key = f"x_{data_type}"
            y_key = f"y_{data_type}"

            if x_key in data and y_key in data:
                x_data = data[x_key]
                y_data = data[y_key]

                if len(x_data.shape) == 4:
                    pass
                elif len(x_data.shape) == 3:
                    height = int(np.sqrt(x_data.shape[1]))
                    x_data = x_data.values.reshape(x_data.shape[0], height, height, 1)
                else:
                    raise ValueError("Unsupported data shape.")

                return x_data, y_data
            else:
                raise KeyError(f"Data type '{data_type}' not found in the pickle file.")
        else:
            raise ValueError(
                "Pickle file doesn't contain expected dictionary structure."
            )


if __name__ == "__main__":
    NON_COLORED_DATASET = {
        "name": "zalando-research/fashionmnist",
        "test_filename": "fashion-mnist_test.csv",
        "train_filename": "fashion-mnist_train.csv",
        "labels_column": "label",
    }

    COLORED_DATASET = {
        "name": "valentynsichkar/traffic-signs-preprocessed",
        "filename": "data0.pickle",
    }

    dataset_config = NON_COLORED_DATASET

    downloader = KaggleDataset(dataset_config["name"])
    downloader.download()

    x_df, y_df = downloader.get_data(
        dataset_config["test_filename"], labels_column=dataset_config["labels_column"]
    )
    print(x_df)
    print(y_df)
