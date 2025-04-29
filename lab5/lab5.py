import os
import sys
import pickle
import datetime
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, List
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    ):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.config = config
        self.__download_path = os.path.join(script_dir, download_dir, config.name)
        self.__api = KaggleApi()

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

        self.__api.authenticate()

        self.__download()
        self.__load_all_data()

        self.__augment()
        self.__normalize()
        self.__shuffle()

    def __download(self, unzip: bool = True) -> None:
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
        print("Download complete")

    def list_files(self) -> List[str]:
        files = []
        for root, _, filenames in os.walk(self.__download_path):
            for f in filenames:
                files.append(os.path.join(root, f))
        return files

    def __load_all_data(self) -> None:
        if self.config.file_type == "pickle":
            (
                self.x_train,
                self.y_train,
                self.x_test,
                self.y_test,
                self.x_val,
                self.y_val,
            ) = self.__load_pickle_file()
        else:
            raise ValueError(f"Unsupported file type: {self.config.file_type}")

        if self.x_train is None or self.y_train is None:
            raise ValueError("Training data is not loaded properly.")
        if self.x_test is None or self.y_test is None:
            raise ValueError("Test data is not loaded properly.")
        if self.x_val is None or self.y_val is None:
            raise ValueError("Validation data is not loaded properly.")

        print(f"Train data loaded: {self.x_train.shape}, {self.y_train.shape}")
        print(f"Validation data loaded: {self.x_val.shape}, {self.y_val.shape}")
        print(f"Test data loaded: {self.x_test.shape}, {self.y_test.shape}")

    def __normalize(self) -> None:
        print("Starting normalization...")
        if self.x_train is None or self.x_test is None or self.x_val is None:
            raise RuntimeError("Data not loaded. Cannot normalize.")

        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0
        self.x_val = self.x_val / 255.0

        print("Data normalized")

    def __shuffle(self) -> None:
        print("Starting data shuffle...")
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Training data not loaded. Cannot shuffle.")

        indices = np.random.permutation(len(self.x_train))
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]

        indices = np.random.permutation(len(self.x_val))
        self.x_val = self.x_val[indices]
        self.y_val = self.y_val[indices]

        print("Train and validation data shuffled")

    def __augment(self) -> None:
        print("Starting data augmentation...")
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Training data not loaded. Cannot augment.")

        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            horizontal_flip=False,
        )

        augment_size = len(self.x_train)
        batch_size = 32
        y_train_cat = tf.keras.utils.to_categorical(
            self.y_train, self.get_num_of_classes()
        )

        generator = datagen.flow(
            self.x_train, y_train_cat, batch_size=batch_size, shuffle=True
        )

        augmented_images = np.empty(
            (augment_size, *self.x_train.shape[1:]), dtype=self.x_train.dtype
        )
        augmented_labels = np.empty((augment_size,), dtype=self.y_train.dtype)

        generated = 0
        while generated < augment_size:
            x_batch, y_batch = next(generator)
            n = min(batch_size, augment_size - generated)
            augmented_images[generated : generated + n] = x_batch[:n]
            augmented_labels[generated : generated + n] = np.argmax(y_batch[:n], axis=1)
            generated += n

        self.x_train = np.concatenate([self.x_train, augmented_images], axis=0)
        self.y_train = np.concatenate([self.y_train, augmented_labels], axis=0)

        indices = np.random.permutation(len(self.x_train))
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]

        print(f"Train data augmented. New shape: {self.x_train.shape}")

    def __load_pickle_file(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        required_keys = [
            "x_train",
            "y_train",
            "x_test",
            "y_test",
            "x_validation",
            "y_validation",
        ]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Required key '{key}' not found in pickle file.")

        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
        x_val = data["x_validation"]
        y_val = data["y_validation"]

        for name, x_data in [
            ("x_train", x_train),
            ("x_test", x_test),
            ("x_val", x_val),
        ]:
            if len(x_data.shape) == 4:
                if x_data.shape[1] not in [1, 3]:
                    raise ValueError(f"Unsupported number of channels for {name}")
                x_data = np.transpose(x_data, (0, 2, 3, 1))
            elif len(x_data.shape) == 3:
                height = int(np.sqrt(x_data.shape[1]))
                x_data = x_data.reshape(x_data.shape[0], height, height, 1)
            else:
                raise ValueError(f"Unsupported shape for {name}")

            if name == "x_train":
                x_train = x_data
            elif name == "x_test":
                x_test = x_data
            else:
                x_val = x_data

        return x_train, y_train, x_test, y_test, x_val, y_val

    def get_sample_shape(self) -> Tuple[int, int, int]:
        if self.x_train is None:
            raise RuntimeError(
                "Training data not loaded. Cannot retrieve sample shape."
            )
        return self.x_train.shape[1:]

    def get_num_of_classes(self) -> int:
        if self.y_train is None:
            raise RuntimeError(
                "Training data not loaded. Cannot retrieve number of classes."
            )
        return len(np.unique(self.y_train))


class OutputLogger:
    def __init__(
        self,
        log_to_file_flag: bool,
        output_directory: str = "outputs",
        log_filename: str = "output.txt",
    ):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self._log_to_file = log_to_file_flag
        self._output_dir = output_directory
        self._log_dir_path = os.path.join(script_dir, output_directory)
        self._log_file_path = os.path.join(self._log_dir_path, log_filename)

        self._original_stdout = sys.stdout
        self._log_file = None

    def start(self) -> None:
        if not self._log_to_file:
            print("Logging to file is disabled.")
            return

        if self._log_file:
            print("Logging has already started.")
            return

        os.makedirs(self._log_dir_path, exist_ok=True)
        self._log_file = open(self._log_file_path, "w", encoding="utf-8")
        print(f"Logging output to {self._log_file_path}...")
        sys.stdout = self._log_file

    def stop(self) -> None:
        if not self._log_to_file:
            return

        if self._log_file:
            sys.stdout = self._original_stdout
            self._log_file.close()
            print(f"All output was logged to {self._log_file_path}")
            self._log_file = None

    def __del__(self):
        self.stop()


class Utils:
    @staticmethod
    def set_np_tf_seed(seed: int = 42) -> None:
        np.random.seed(seed)
        print(f"NumPy seed set to {seed}")

        tf.random.set_seed(seed)
        print(f"TensorFlow seed set to {seed}")

    @staticmethod
    def set_tf_gpu() -> None:
        physical_devices = tf.config.list_physical_devices("GPU")
        if not physical_devices:
            print("No GPU found, using CPU instead")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    @staticmethod
    def get_tf_log_verbosity(log_to_file_flag: bool) -> int:
        return 2 if log_to_file_flag else 1

    @staticmethod
    def get_tensorboard_callback(log_dir_prefix="logs/fit") -> TensorBoard:
        base_log_dir = os.path.normpath(log_dir_prefix)
        os.makedirs(base_log_dir, exist_ok=True)

        log_dir = os.path.join(
            base_log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(log_dir, exist_ok=True)

        print(f"TensorBoard logs will be saved to: {log_dir}")
        return TensorBoard(log_dir=log_dir, histogram_freq=1)


if __name__ == "__main__":
    LOGGING_ENABLED = False
    TF_LOG_VERBOSITY = Utils.get_tf_log_verbosity(LOGGING_ENABLED)

    logger = OutputLogger(LOGGING_ENABLED)
    logger.start()

    Utils.set_np_tf_seed()
    Utils.set_tf_gpu()

    dataset_config = DatasetConfig.traffic_signs()
    dataset = KaggleDataset(dataset_config)
