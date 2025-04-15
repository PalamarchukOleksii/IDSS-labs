import os
import pandas as pd
import pickle
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from typing import Tuple, Optional, List
import tensorflow as tf
from tensorflow.keras import layers, models


class DatasetConfig:
    def __init__(
        self,
        name: str,
        file_type: str,
        train_filename: Optional[str] = None,
        test_filename: Optional[str] = None,
        combined_filename: Optional[str] = None,
        labels_column: str = "label",
    ):
        self.name = name
        self.file_type = file_type
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.combined_filename = combined_filename
        self.labels_column = labels_column

    @classmethod
    def fashion_mnist(cls) -> "DatasetConfig":
        return cls(
            name="zalando-research/fashionmnist",
            file_type="csv",
            train_filename="fashion-mnist_train.csv",
            test_filename="fashion-mnist_test.csv",
        )

    @classmethod
    def traffic_signs(cls) -> "DatasetConfig":
        return cls(
            name="valentynsichkar/traffic-signs-preprocessed",
            file_type="pickle",
            combined_filename="data0.pickle",
        )


class KaggleDataset:
    def __init__(
        self,
        config: DatasetConfig,
        download_dir: str = "dataset",
        auto_load: bool = False,
        normalize: bool = True,
        shuffle: bool = True,
    ):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.config = config
        self.__download_path = os.path.join(script_dir, download_dir, config.name)
        self.__api = KaggleApi()

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.need_normalize = normalize
        self.need_shuffle = shuffle
        self.is_normalized = False
        self.is_shuffled = False

        self.__api.authenticate()

        if auto_load:
            self.download()
            self.load_all_data()

    def download(self, unzip: bool = True) -> None:
        if (
            os.listdir(self.__download_path)
            if os.path.exists(self.__download_path)
            else []
        ):
            print(f"Dataset already exists at: {self.__download_path}")
            return

        if not os.path.exists(self.__download_path):
            os.makedirs(self.__download_path)

        print(f"Downloading dataset: {self.config.name}")
        print(f"Saving to: {self.__download_path}")
        self.__api.dataset_download_files(
            dataset=self.config.name,
            path=self.__download_path,
            unzip=unzip,
        )
        print("Download complete.")

    def list_files(self) -> List[str]:
        files = []
        for root, _, filenames in os.walk(self.__download_path):
            for f in filenames:
                files.append(os.path.join(root, f))
        return files

    def load_all_data(self) -> None:
        if self.config.file_type == "csv":
            self.x_train, self.y_train = self.__load_csv_file("train")
            self.x_test, self.y_test = self.__load_csv_file("test")
        elif self.config.file_type == "pickle":
            self.x_train, self.y_train, self.x_test, self.y_test = (
                self.__load_pickle_file()
            )
        else:
            raise ValueError(f"Unsupported file type: {self.config.file_type}")

        if self.need_normalize:
            self.normalize()

        if self.need_shuffle:
            self.shuffle()

        print(f"Train data loaded: {self.x_train.shape}, {self.y_train.shape}")
        print(f"Test data loaded: {self.x_test.shape}, {self.y_test.shape}")

    def normalize(self) -> None:
        if self.x_train is not None:
            self.x_train = self.x_train / 255.0
        if self.x_test is not None:
            self.x_test = self.x_test / 255.0
        self.is_normalized = True
        print("Data normalized")

    def shuffle(self) -> None:
        if self.x_train is not None and self.y_train is not None:
            indices = np.random.permutation(len(self.x_train))
            self.x_train = self.x_train[indices]
            self.y_train = self.y_train[indices]

        if self.x_test is not None and self.y_test is not None:
            indices = np.random.permutation(len(self.x_test))
            self.x_test = self.x_test[indices]
            self.y_test = self.y_test[indices]

        self.is_shuffled = True
        print("Data shuffled")

    def __load_csv_file(self, data_type: str) -> tuple[np.ndarray, np.ndarray]:
        if data_type == "train":
            filename = self.config.train_filename
        elif data_type == "test":
            filename = self.config.test_filename
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        if not filename:
            raise ValueError(f"No {data_type} filename specified in config")

        file_path = os.path.join(self.__download_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File {filename} not found in {self.__download_path}"
            )

        print(f"Loading CSV data from {file_path}")
        df = pd.read_csv(file_path)

        if self.config.labels_column in df.columns:
            x_data = df.drop(columns=[self.config.labels_column])
            y_data = df[[self.config.labels_column]]

            if len(x_data.shape) == 2:
                height = int(np.sqrt(x_data.shape[1]))
                x_data = x_data.values.reshape(x_data.shape[0], height, height, 1)
            else:
                raise ValueError("Unsupported data shape for CNN.")

            return x_data, y_data.to_numpy().flatten()
        else:
            raise KeyError(
                f"Label column '{self.config.labels_column}' not found in the CSV file."
            )

    def __load_pickle_file(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.config.combined_filename:
            raise ValueError("No combined filename specified in config")

        file_path = os.path.join(self.__download_path, self.config.combined_filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File {self.config.combined_filename} not found in {self.__download_path}"
            )

        print(f"Loading pickle data from {file_path}")
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                "Pickle file doesn't contain expected dictionary structure."
            )

        required_keys = ["x_train", "y_train", "x_test", "y_test"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Required key '{key}' not found in pickle file.")

        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

        for x_data_name, x_data in [("x_train", x_train), ("x_test", x_test)]:
            if len(x_data.shape) == 4:
                if x_data.shape[-1] > 3:
                    if x_data_name == "x_train":
                        x_train = np.transpose(x_data, (0, 2, 3, 1))
                    else:
                        x_test = np.transpose(x_data, (0, 2, 3, 1))
            elif len(x_data.shape) == 3:
                height = int(np.sqrt(x_data.shape[1]))
                if x_data_name == "x_train":
                    x_train = x_data.reshape(x_data.shape[0], height, height, 1)
                else:
                    x_test = x_data.reshape(x_data.shape[0], height, height, 1)
            else:
                raise ValueError(f"Unsupported shape for {x_data_name}")

        return x_train, y_train, x_test, y_test

    def get_sample_shape(self) -> Tuple[int, int, int]:
        return self.x_train.shape[1:]

    def get_num_of_classes(self) -> int:
        return len(np.unique(self.y_train))


class CNNModel:
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        conv_filters: int = 32,
        kernel_size: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        padding: str = "same",
        dense_units: int = 64,
        num_conv_layers: int = 1,
        use_batch_norm: bool = False,
        use_dropout: bool = False,
        dropout_rate: float = 0.5,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dense_units = dense_units
        self.num_conv_layers = num_conv_layers
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.model = None

    def build(self) -> None:
        inputs = layers.Input(shape=self.input_shape)
        x = inputs

        for i in range(self.num_conv_layers):
            x = layers.Conv2D(
                self.conv_filters * (2**i),
                self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                activation="relu",
            )(x)

            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D((2, 2))(x)

            if self.use_dropout:
                x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Flatten()(x)

        if self.dense_units > 0:
            x = layers.Dense(self.dense_units, activation="relu")(x)
            if self.use_dropout:
                x = layers.Dropout(self.dropout_rate)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = models.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model.summary()

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 32,
        epochs: int = 5,
    ) -> None:
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
        )
        return history

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        loss, accuracy = self.model.evaluate(x_test, y_test)
        return loss, accuracy


class Utils:
    @staticmethod
    def set_np_tf_seed(seed: int = 42) -> None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    @staticmethod
    def set_tf_gpu() -> None:
        physical_devices = tf.config.list_physical_devices("GPU")
        if not physical_devices:
            print("No GPU found. Using CPU instead.")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    Utils.set_np_tf_seed()
    Utils.set_tf_gpu()

    dataset_config = DatasetConfig.traffic_signs()
    dataset = KaggleDataset(dataset_config, auto_load=True)

    cnn_model = CNNModel(
        input_shape=dataset.get_sample_shape(),
        num_classes=dataset.get_num_of_classes(),
    )

    cnn_model.build()
    cnn_model.train(
        dataset.x_train,
        dataset.y_train,
    )

    train_loss, train_acc = cnn_model.evaluate(dataset.x_train, dataset.y_train)
    test_loss, test_acc = cnn_model.evaluate(dataset.x_test, dataset.y_test)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
