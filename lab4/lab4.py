import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import History
from typing import Tuple, Optional, List, Dict
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.utils import to_categorical
import random
from sklearn.metrics import ConfusionMatrixDisplay
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
        auto_load: bool = True,
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

        self.is_normalized = not normalize
        self.is_shuffled = not shuffle

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

        if not self.is_normalized:
            self.normalize()

        if not self.is_shuffled:
            self.shuffle()

        print(f"Train data loaded: {self.x_train.shape}, {self.y_train.shape}")
        print(f"Test data loaded: {self.x_test.shape}, {self.y_test.shape}")

    def normalize(self) -> None:
        if self.x_train is not None and not self.is_normalized:
            self.x_train = self.x_train / 255.0
        if self.x_test is not None and not self.is_normalized:
            self.x_test = self.x_test / 255.0
        self.is_normalized = True
        print("Data normalized")

    def shuffle(self) -> None:
        if (
            self.x_train is not None
            and self.y_train is not None
            and not self.is_shuffled
        ):
            indices = np.random.permutation(len(self.x_train))
            self.x_train = self.x_train[indices]
            self.y_train = self.y_train[indices]

        if self.x_test is not None and self.y_test is not None and not self.is_shuffled:
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
        kernel_initializer: str = "glorot_uniform",
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
        self.kernel_initializer = kernel_initializer
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
                kernel_initializer=self.kernel_initializer,
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
            verbose: int = 1,
            use_early_stopping: bool = False,
            callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> History:
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )
        return history

    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        verbose: int = 1,
    ) -> Tuple[float, float]:
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=verbose)
        return loss, accuracy


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
    def get_dataset_config(is_dataset_colored: bool) -> Dict:
        if is_dataset_colored:
            print("Dataset set to: traffic-signs-preprocessed")
            return DatasetConfig.traffic_signs()

        print("Dataset set to: fashionmnist")
        return DatasetConfig.fashion_mnist()

    @staticmethod
    def get_tf_log_verbosity(log_to_file_flag: bool) -> int:
        return 2 if log_to_file_flag else 1

    @staticmethod
    def get_tensorboard_callback(log_dir_prefix="logs/fit") -> TensorBoard:
        base_log_dir = os.path.normpath(log_dir_prefix)
        os.makedirs(base_log_dir, exist_ok=True)  # ← гарантуємо наявність базової папки

        log_dir = os.path.join(base_log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)  # ← гарантуємо створення конкретного підкаталогу

        print(f"TensorBoard logs will be saved to: {log_dir}")
        return TensorBoard(log_dir=log_dir, histogram_freq=1)


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


def analyze_conv_parameters(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    verbosity: int = 1,
) -> Dict:
    """Analysis of convolution parameters: kernel_size, strides, padding."""
    from itertools import product

    param_combinations = list(product(
        [(3, 3), (5, 5)],           # kernel_size
        [(1, 1), (2, 2)],           # strides
        ["same", "valid"]           # padding
    ))

    best_config = None
    best_accuracy = -1.0
    results = []

    print("\n=== Analysis of convolution parameters ===")
    for kernel_size, strides, padding in param_combinations:
        print(f"\nConfiguration: kernel={kernel_size}, strides={strides}, padding={padding}")

        model = CNNModel(
            input_shape=input_shape,
            num_classes=num_classes,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
        )
        model.build()

        history = model.train(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=3,
            batch_size=64,
            verbose=verbosity
        )

        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=verbosity)
        print(f"Validation accuracy = {val_acc:.4f}")
        results.append(((kernel_size, strides, padding), val_acc))

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_config = (kernel_size, strides, padding)

    print("\nThe best configuration:")
    print(f"Kernel: {best_config[0]}, Strides: {best_config[1]}, Padding: {best_config[2]}")
    print(f"Validation Accuracy: {best_accuracy:.4f}")

    return {
        "best_kernel_size": best_config[0],
        "best_strides": best_config[1],
        "best_padding": best_config[2],
        "best_accuracy": best_accuracy,
        "all_results": results
    }


def evaluate_architectures_detailed(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    x_val: np.ndarray,
    y_val: np.ndarray,
    architectures: List[Dict],
    verbosity: int = 1
) -> None:
    """Evaluate CNN architectures using multiple classification metrics on validation set."""
    print("\n=== Detailed Evaluation of Architectures ===")

    results = []

    for arch in architectures:
        print(f"\nEvaluating: {arch['name']}")

        model = CNNModel(
            input_shape=input_shape,
            num_classes=num_classes,
            num_conv_layers=arch["num_conv"],
            use_batch_norm=arch["batch_norm"],
            use_dropout=arch["dropout"],
        )
        model.build()

        # Train model (can reuse train data from earlier)
        model.train(
            x_train_part, y_train_part,
            validation_data=(x_val, y_val),
            epochs=5,
            batch_size=64,
            verbose=verbosity
        )

        # Predictions
        y_pred_probs = model.model.predict(x_val)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        try:
            auc = roc_auc_score(y_val, y_pred_probs, multi_class="ovo", average="macro")
        except ValueError:
            auc = float("nan")  # For binary or degenerate cases

        results.append({
            "name": arch["name"],
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auc": auc,
        })

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")

    print("\n=== Summary Table ===")
    for r in results:
        print(f"{r['name']:<20} | Acc: {r['accuracy']:.4f} | Prec: {r['precision']:.4f} | Recall: {r['recall']:.4f} | F1: {r['f1_score']:.4f} | AUC: {r['auc']:.4f}")

    # Identify best model for each metric
    best_by_accuracy = max(results, key=lambda r: r["accuracy"])
    best_by_precision = max(results, key=lambda r: r["precision"])
    best_by_recall = max(results, key=lambda r: r["recall"])
    best_by_f1 = max(results, key=lambda r: r["f1_score"])
    best_by_auc = max(results, key=lambda r: r["auc"] if not np.isnan(r["auc"]) else -1)

    print("\n=== Best models per metric ===")
    print(f"Accuracy:  {best_by_accuracy['name']} ({best_by_accuracy['accuracy']:.4f})")
    print(f"Precision: {best_by_precision['name']} ({best_by_precision['precision']:.4f})")
    print(f"Recall:    {best_by_recall['name']} ({best_by_recall['recall']:.4f})")
    print(f"F1-score:  {best_by_f1['name']} ({best_by_f1['f1_score']:.4f})")
    print(f"AUC:       {best_by_auc['name']} ({best_by_auc['auc']:.4f})")

    # Calculate average of all 5 metrics
    for r in results:
        r["mean_score"] = np.mean([
            r["accuracy"],
            r["precision"],
            r["recall"],
            r["f1_score"],
            r["auc"] if not np.isnan(r["auc"]) else 0.0
        ])

    best_overall = max(results, key=lambda r: r["mean_score"])

    print("\n=== Best overall model (by mean of all metrics) ===")
    print(f"Name: {best_overall['name']}")
    print(f"Mean Score: {best_overall['mean_score']:.4f}")
    print(f"Accuracy:  {best_overall['accuracy']:.4f}")
    print(f"Precision: {best_overall['precision']:.4f}")
    print(f"Recall:    {best_overall['recall']:.4f}")
    print(f"F1-Score:  {best_overall['f1_score']:.4f}")
    print(f"AUC:       {best_overall['auc']:.4f}")

from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model(model, x_test, y_test, class_names=None):
    """
    Завдання 9: Розрахунок оцінок якості моделі на тестовій множині
    """
    print("\n=== Оцінка якості моделі на тестовій множині ===")
    
    # Оцінка точності та втрат
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Передбачення класів
    y_pred = np.argmax(model.model.predict(x_test), axis=1)
    
    # Розрахунок метрик
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Візуалізація confusion matrix
    if class_names:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def load_and_predict_random_images(model, x_test, y_test, class_names=None, num_images=5):
    print("\n=== Розпізнавання тестових зображень ===")
    
    # Вибір випадкових зображень
    indices = random.sample(range(len(x_test)), num_images)
    sample_images = x_test[indices]
    sample_labels = y_test[indices]
    
    # Передбачення класів
    predictions = model.model.predict(sample_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Візуалізація результатів
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        if sample_images[i].shape[-1] == 1:  # Чорно-білі зображення
            plt.imshow(sample_images[i].squeeze(), cmap='gray')
        else:  # Кольорові зображення
            plt.imshow(sample_images[i])
        
        true_label = class_names[sample_labels[i]] if class_names else sample_labels[i]
        pred_label = class_names[predicted_classes[i]] if class_names else predicted_classes[i]
        
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=title_color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class MLPModel:
    """
    Клас для багатошарового персептрона (MLP)
    """
    def __init__(self, input_shape, num_classes, hidden_layer_sizes=(128,), max_iter=100):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.model = None
    
    def build(self):
        input_size = np.prod(self.input_shape)
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=42
        )
    
    def train(self, x_train, y_train):
        # Змінюємо форму даних для MLP (з 4D в 2D)
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train_flat, y_train)
    
    def evaluate(self, x_test, y_test):
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        return self.model.score(x_test_flat, y_test)

def compare_models(cnn_model, mlp_model, x_test, y_test, x_train, y_train):
    print("\n=== Порівняння CNN та MLP ===")
    
    # Оцінка часу навчання (спрощено)
    import time
    
    # CNN
    start_time = time.time()
    cnn_model.train(x_train, y_train, epochs=5, verbose=0)
    cnn_time = time.time() - start_time
    
    # MLP
    start_time = time.time()
    mlp_model.train(x_train, y_train)
    mlp_time = time.time() - start_time
    
    # Оцінка точності
    cnn_acc = cnn_model.evaluate(x_test, y_test, verbose=0)[1]
    mlp_acc = mlp_model.evaluate(x_test, y_test)
    
    print("\nРезультати порівняння:")
    print(f"{'Метрика':<15} | {'CNN':<10} | {'MLP':<10}")
    print("-" * 40)
    print(f"{'Точність':<15} | {cnn_acc:.4f}    | {mlp_acc:.4f}")
    print(f"{'Час навчання':<15} | {cnn_time:.2f} сек | {mlp_time:.2f} сек")
    print(f"{'Параметри':<15} | {cnn_model.model.count_params():<10} | {mlp_model.model.n_layers_ * mlp_model.model.hidden_layer_sizes[0]:<10} (приблизно)")

if __name__ == "__main__":
    LOGGING_ENABLED = False
    TF_LOG_VERBOSITY = Utils.get_tf_log_verbosity(LOGGING_ENABLED)
    IS_DATASET_COLORED = True

    logger = OutputLogger(LOGGING_ENABLED)
    logger.start()

    Utils.set_np_tf_seed()
    Utils.set_tf_gpu()

    dataset_config = Utils.get_dataset_config(IS_DATASET_COLORED)
    dataset = KaggleDataset(dataset_config)

    cnn_model = CNNModel(
        dataset.get_sample_shape(),
        dataset.get_num_of_classes(),
    )

    cnn_model.build()

    x_train_part, x_val_part, y_train_part, y_val_part = train_test_split(
        dataset.x_train, dataset.y_train, test_size=0.2, random_state=42
    )

    tensorboard_cb = Utils.get_tensorboard_callback()

    cnn_model.train(
        x_train_part, y_train_part,
        validation_data=(x_val_part, y_val_part),
        verbose=TF_LOG_VERBOSITY,
        callbacks=[tensorboard_cb]
    )

    train_loss, train_acc = cnn_model.evaluate(
        dataset.x_train, dataset.y_train, TF_LOG_VERBOSITY
    )
    test_loss, test_acc = cnn_model.evaluate(
        dataset.x_test, dataset.y_test, TF_LOG_VERBOSITY
    )

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Analysis of convolution parameters after training the base model
    _ = analyze_conv_parameters(
        input_shape=dataset.get_sample_shape(),
        num_classes=dataset.get_num_of_classes(),
        x_train=x_train_part,
        y_train=y_train_part,
        x_val=x_val_part,
        y_val=y_val_part,
        verbosity=TF_LOG_VERBOSITY
    )

    architectures_to_evaluate = [
        {"name": "baseline", "num_conv": 1, "batch_norm": False, "dropout": False},
        {"name": "2conv", "num_conv": 2, "batch_norm": False, "dropout": False},
        {"name": "2conv_bn", "num_conv": 2, "batch_norm": True, "dropout": False},
        {"name": "2conv_bn_dropout", "num_conv": 2, "batch_norm": True, "dropout": True},
        {"name": "3conv_bn_dropout", "num_conv": 3, "batch_norm": True, "dropout": True},
    ]

    # Evaluate architectures with detailed metrics
    evaluate_architectures_detailed(
        input_shape=dataset.get_sample_shape(),
        num_classes=dataset.get_num_of_classes(),
        x_val=x_val_part,
        y_val=y_val_part,
        architectures=architectures_to_evaluate,
        verbosity=TF_LOG_VERBOSITY
    )

    print("\n=== Effect of Regularization and Initialization ===")

    regularization_configs = [
        {"name": "no_regularization", "dropout": False, "early_stop": False, "init": "glorot_uniform"},
        {"name": "dropout_only", "dropout": True, "early_stop": False, "init": "glorot_uniform"},
        {"name": "early_stop_only", "dropout": False, "early_stop": True, "init": "glorot_uniform"},
        {"name": "dropout+early_stop", "dropout": True, "early_stop": True, "init": "glorot_uniform"},
        {"name": "he_initializer", "dropout": False, "early_stop": False, "init": "he_uniform"},
    ]

    for config in regularization_configs:
        print(f"\nConfiguration: {config['name']}")
        model = CNNModel(
            input_shape=dataset.get_sample_shape(),
            num_classes=dataset.get_num_of_classes(),
            use_dropout=config["dropout"],
            kernel_initializer=config["init"]
        )
        model.build()
        model.train(
            x_train_part, y_train_part,
            validation_data=(x_val_part, y_val_part),
            epochs=15,
            verbose=TF_LOG_VERBOSITY,
            use_early_stopping=config["early_stop"]
        )
        val_loss, val_acc = model.evaluate(x_val_part, y_val_part, verbose=TF_LOG_VERBOSITY)
        print(f"Validation Accuracy: {val_acc:.4f}")
  
    class_names = None
    if not IS_DATASET_COLORED:
        class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    
    evaluate_model(cnn_model, dataset.x_test, dataset.y_test, class_names)
    
    load_and_predict_random_images(cnn_model, dataset.x_test, dataset.y_test, class_names)
    
    mlp_model = MLPModel(
        input_shape=dataset.get_sample_shape(),
        num_classes=dataset.get_num_of_classes(),
        hidden_layer_sizes=(128, 64)
    )
    mlp_model.build()
    
    compare_models(
        cnn_model=cnn_model,
        mlp_model=mlp_model,
        x_test=dataset.x_test,
        y_test=dataset.y_test,
        x_train=x_train_part,
        y_train=y_train_part
    )
    
    logger.stop()
