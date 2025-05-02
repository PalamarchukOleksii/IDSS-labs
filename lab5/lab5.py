import os
import sys
import pickle
import datetime
from typing import Tuple, Optional, List, Type
import numpy as np
import tensorflow as tf
from kaggle.api.kaggle_api_extended import KaggleApi
from keras.callbacks import (
    TensorBoard,
    Callback,
    ModelCheckpoint,
    EarlyStopping,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.keras.applications import (
    VGG19,
    InceptionV3,
    ResNet152V2,
    DenseNet201,
    EfficientNetB7,
    Xception,
)
from keras.callbacks import Callback
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow.summary as tf_summary
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

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
        data_config: DatasetConfig,
        download_dir: str = "dataset",
    ):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.config = data_config
        self.__download_path = os.path.join(script_dir, download_dir, self.config.name)
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

        self.__shuffle()
        self.__augment()
        self.__shuffle()
        self.__normalize()

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
        else:
            print(f"Found {len(physical_devices)} GPU(s)")
            for device in physical_devices:
                try:
                    tf.config.set_memory_growth(device, True)
                    print(f"Memory growth enabled for GPU: {device}")
                except RuntimeError as e:
                    print(f"Could not set memory growth for GPU: {device} - {e}")

    @staticmethod
    def get_tf_log_verbosity(log_to_file_flag: bool) -> int:
        return 2 if log_to_file_flag else 1

    @staticmethod
    def get_tensorboard_callback(
        model_name: str, log_dir_prefix="logs/fit"
    ) -> TensorBoard:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        base_log_dir = os.path.normpath(os.path.join(base_dir, log_dir_prefix))

        os.makedirs(base_log_dir, exist_ok=True)

        log_dir = os.path.join(
            base_log_dir,
            f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        os.makedirs(log_dir, exist_ok=True)

        print(f"TensorBoard logs will be saved to: {log_dir}")
        return TensorBoard(log_dir=log_dir, histogram_freq=1)


class MetricsLogger(Callback):
    def __init__(self, validation_data, log_dir):
        super().__init__()
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        val_data, val_labels = self.validation_data
        val_preds = self.model.predict(val_data, verbose=0)
        pred_classes = np.argmax(val_preds, axis=1)

        f1 = f1_score(val_labels, pred_classes, average="macro")
        try:
            auc = roc_auc_score(val_labels, val_preds, multi_class="ovo", average="macro")
        except ValueError:
            auc = float("nan")

        with self.file_writer.as_default():
            tf.summary.scalar("val_f1_score", f1, step=epoch)
            tf.summary.scalar("val_auc", auc, step=epoch)


class TransferLearningModel:
    MODEL_MAP = {
        "VGG19": VGG19,
        "InceptionV3": InceptionV3,
        "ResNet152V2": ResNet152V2,
        "DenseNet201": DenseNet201,
        "EfficientNetB7": EfficientNetB7,
        "Xception": Xception,
    }

    def __init__(
        self,
        base_model_name: str,
        input_shape: tuple[int, int, int],
        num_classes: int,
        fully_connected_layers: Optional[list[tuple[int, str]]] = None,
        use_dropout: bool = False,
        dropout_rate: float = 0.5,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        freeze_base_model: bool = True,
    ):
        if fully_connected_layers is None:
            fully_connected_layers = [(128, "relu")]

        if base_model_name not in self.MODEL_MAP:
            raise ValueError(
                f"Model {base_model_name} not supported. Choose from: {list(self.MODEL_MAP.keys())}"
            )

        self.base_model_name = base_model_name
        self.base_model_class: Type[Model] = self.MODEL_MAP[base_model_name]
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.fully_connected_layers = fully_connected_layers
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.freeze_base_model = freeze_base_model

        self.model: Optional[Model] = None
        self.base_model: Optional[Model] = None
        self.history = None
        self._build_model()

    def __get_optimizer(self):
        if self.optimizer_name == "adam":
            return Adam(learning_rate=self.learning_rate)
        if self.optimizer_name == "sgd":
            return SGD(learning_rate=self.learning_rate, momentum=0.9)
        if self.optimizer_name == "rmsprop":
            return RMSprop(learning_rate=self.learning_rate)
        if self.optimizer_name == "nadam":
            return Nadam(learning_rate=self.learning_rate)

        raise ValueError(
            f"Optimizer {self.optimizer_name} is not supported. Choose from: ['adam', 'sgd', 'rmsprop', 'nadam']"
        )

    def _build_model(self):
        print(f"Building model with base: {self.base_model_class.__name__}")

        print("Importing pre-trained weights from ImageNet...")
        self.base_model = self.base_model_class(
            weights="imagenet",
            include_top=False,
            input_shape=self.input_shape,
        )

        if self.freeze_base_model:
            print("Freezing base model weights...")
            self.base_model.trainable = False
        else:
            print("Base model weights are trainable")

        print("Building top layers...")
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)

        for i, (units, activation) in enumerate(self.fully_connected_layers):
            print(f"Adding FC layer {i+1}: {units} units with {activation} activation")
            x = Dense(units, activation=activation, name=f"fc_{i+1}")(x)
            if self.use_dropout:
                x = Dropout(self.dropout_rate, name=f"dropout_{i+1}")(x)

        print(f"Adding classification layer with {self.num_classes} classes")
        output = Dense(self.num_classes, activation="softmax", name="classification")(x)

        self.model = Model(
            inputs=self.base_model.input,
            outputs=output,
            name=f"{self.base_model_name}_transfer",
        )

        self.model.compile(
            optimizer=self.__get_optimizer(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        print(f"Model {self.base_model_name} built successfully")

    def fit(
        self,
        verbose,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[List[Callback]] = None,
    ):
        print(f"Starting training of {self.base_model_name}...")

        self.history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        print(f"Training of {self.base_model_name} completed")
        return self.history

    def evaluate(self, verbose, x_test: np.ndarray, y_test: np.ndarray):
        print(f"Evaluating {self.base_model_name} model...")
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=verbose)
        print(
            f"{self.base_model_name} test accuracy: {test_acc:.4f}, test loss: {test_loss:.4f}"
        )
        return test_loss, test_acc

    def summary(self):
        print(f"\n--- {self.base_model_name} Model Summary ---")
        self.model.summary()

    def fine_tune(
        self,
        verbose,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        num_layers_to_unfreeze: int = 10,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        callbacks: Optional[List[Callback]] = None,
    ):
        if not self.freeze_base_model:
            print("Base model is already trainable, skipping fine-tuning")
            return None

        print(f"Fine-tuning {self.base_model_name} model...")

        total_layers = len(self.base_model.layers)
        for layer in self.base_model.layers[
            -(min(num_layers_to_unfreeze, total_layers)) :
        ]:
            layer.trainable = True

        print(
            f"Unfrozen {min(num_layers_to_unfreeze, total_layers)} layers from the base model"
        )

        self.learning_rate = learning_rate
        self.model.compile(
            optimizer=self.__get_optimizer(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        fine_tune_history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        print(f"Fine-tuning of {self.base_model_name} completed")
        return fine_tune_history

class ImageRecognizer:
    @staticmethod
    def load_image_from_url(url, target_size):
        """Завантажити зображення з URL та підготувати його для моделі"""
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            img = img.resize(target_size)
            return img
        except Exception as e:
            print(f"Помилка завантаження зображення: {e}")
            return None

    @staticmethod
    def preprocess_image(img, model_name):
        """Попередня обробка зображення для конкретної моделі"""
        img_array = np.array(img) / 255.0
        
        # Специфічна підготовка для кожної моделі
        if model_name == "VGG19":
            img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
        elif model_name == "Xception":
            img_array = tf.keras.applications.xception.preprocess_input(img_array)
        elif model_name == "InceptionV3":
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
        elif model_name == "ResNet152V2":
            img_array = tf.keras.applications.resnet.preprocess_input(img_array)
        elif model_name == "DenseNet201":
            img_array = tf.keras.applications.densenet.preprocess_input(img_array)
        elif model_name == "EfficientNetB7":
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        return np.expand_dims(img_array, axis=0)

    @staticmethod
    def recognize_image(model, img_array, class_names):
        """Розпізнати зображення за допомогою навченої моделі"""
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        return class_names[predicted_class], confidence

    @staticmethod
    def display_results(img, model_name, class_name, confidence):
        """Відобразити результати розпізнавання"""
        plt.figure(figsize=(8, 4))
        plt.imshow(img)
        plt.title(f"Модель: {model_name}\nКлас: {class_name}\nВпевненість: {confidence:.2%}")
        plt.axis('off')
        plt.show()

class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate_model(
        self,
        verbose,
        model: TransferLearningModel,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):
        model_name = model.base_model_name
        test_loss, test_acc = model.evaluate(verbose, x_test, y_test)

        self.results[model_name] = {"accuracy": test_acc, "loss": test_loss}

        return test_loss, test_acc

    def compare_models(self):
        print("\n--- Model Comparison Results ---")
        sorted_results = sorted(
            self.results.items(), key=lambda x: x[1]["accuracy"], reverse=True
        )

        print("{:<15} {:<10} {:<10}".format("Model", "Accuracy", "Loss"))
        print("-" * 35)

        for model_name, metrics in sorted_results:
            print(
                "{:<15} {:<10.4f} {:<10.4f}".format(
                    model_name, metrics["accuracy"], metrics["loss"]
                )
            )

        best_model = sorted_results[0][0]
        print(
            f"\nBest model: {best_model} with accuracy: {self.results[best_model]['accuracy']:.4f}"
        )

        return sorted_results
    
    def recognize_test_images(self, image_urls, class_names, target_size=(224, 224)):
        """Розпізнати тестові зображення всіма навченими моделями"""
        for url in image_urls:
            print(f"\n{'='*50}")
            print(f"Розпізнавання зображення: {url}")
            print(f"{'='*50}")
            
            for model_name, model_data in self.results.items():
                # Завантажуємо та підготовлюємо зображення
                img = ImageRecognizer.load_image_from_url(url, target_size)
                if img is None:
                    continue
                
                # Підготовка зображення для конкретної моделі
                img_array = ImageRecognizer.preprocess_image(img, model_name)
                
                # Розпізнавання
                class_name, confidence = ImageRecognizer.recognize_image(
                    model_data["model"], 
                    img_array,
                    class_names
                )
                
                # Відображення результатів
                ImageRecognizer.display_results(img, model_name, class_name, confidence)


if __name__ == "__main__":
    LOGGING_ENABLED = True
    TF_LOG_VERBOSITY = Utils.get_tf_log_verbosity(LOGGING_ENABLED)

    logger = OutputLogger(LOGGING_ENABLED)
    logger.start()

    Utils.set_np_tf_seed()
    Utils.set_tf_gpu()

    dataset_config = DatasetConfig.traffic_signs()
    dataset = KaggleDataset(dataset_config)

    dataset_input_shape = dataset.get_sample_shape()
    dataset_num_classes = dataset.get_num_of_classes()
    print(f"Input shape: {dataset_input_shape}")
    print(f"Number of classes: {dataset_num_classes}")

    models_config = [
        {
            "name": "VGG19",
            "fully_connected_layers": [(256, "relu"), (128, "relu")],
            "use_dropout": True,
            "dropout_rate": 0.5,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "epochs": 5,
            "batch_size": 32,
        },
        # {
        #     "name": "ResNet152V2",
        #     "fully_connected_layers": [(512, "relu"), (256, "relu")],
        #     "use_dropout": True,
        #     "dropout_rate": 0.4,
        #     "optimizer": "adam",
        #     "learning_rate": 0.0005,
        #     "epochs": 5,
        #     "batch_size": 32,
        # },
        # {
        #     "name": "EfficientNetB7",
        #     "fully_connected_layers": [(512, "relu"), (256, "relu"), (128, "relu")],
        #     "use_dropout": True,
        #     "dropout_rate": 0.5,
        #     "optimizer": "adam",
        #     "learning_rate": 0.0005,
        #     "epochs": 5,
        #     "batch_size": 32,
        # },
        # {
        #     "name": "Xception",
        #     "fully_connected_layers": [(256, "relu"), (128, "relu")],
        #     "use_dropout": True,
        #     "dropout_rate": 0.5,
        #     "optimizer": "adam",
        #     "learning_rate": 0.001,
        #     "epochs": 5,
        #     "batch_size": 32,
        # },
    ]

    evaluator = ModelEvaluator()

    for config in models_config:
        print(f"\n{'='*50}")
        print(f"Training {config['name']} model")
        print(f"{'='*50}")

        tl_model = TransferLearningModel(
            base_model_name=config["name"],
            input_shape=dataset_input_shape,
            num_classes=dataset_num_classes,
            fully_connected_layers=config["fully_connected_layers"],
            use_dropout=config["use_dropout"],
            dropout_rate=config["dropout_rate"],
            optimizer=config["optimizer"],
            learning_rate=config["learning_rate"],
            freeze_base_model=True,
        )

        tl_model.summary()

        # TensorBoard callback
        tensorboard_callback = Utils.get_tensorboard_callback(model_name=config["name"])

        # Custom metrics callback
        log_dir = tensorboard_callback.log_dir
        custom_metrics_callback = MetricsLogger(
            validation_data=(dataset.x_val, dataset.y_val),
            log_dir=log_dir
        )

        history = tl_model.fit(
            verbose=TF_LOG_VERBOSITY,
            x_train=dataset.x_train,
            y_train=dataset.y_train,
            x_val=dataset.x_val,
            y_val=dataset.y_val,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            callbacks=[tensorboard_callback, custom_metrics_callback],
        )

        # Uncomment to enable fine-tuning
        # fine_tune_history = tl_model.fine_tune(
        #     verbose=TF_LOG_VERBOSITY,
        #     x_train=dataset.x_train,
        #     y_train=dataset.y_train,
        #     x_val=dataset.x_val,
        #     y_val=dataset.y_val,
        #     num_layers_to_unfreeze=5,
        #     epochs=5,
        #     batch_size=config["batch_size"],
        #     learning_rate=config["learning_rate"] / 10,
        # )

        evaluator.evaluate_model(
            TF_LOG_VERBOSITY, tl_model, dataset.x_test, dataset.y_test
        )

    best_models = evaluator.compare_models()
    
     # Приклад тестових зображень для розпізнавання
    test_images = [
        "https://example.com/traffic_sign1.jpg",
        "https://example.com/traffic_sign2.jpg",
        "https://example.com/traffic_sign3.jpg"
    ]
    
    # Отримуємо назви класів (вам потрібно адаптувати під ваш набір даних)
    class_names = [f"Class_{i}" for i in range(dataset_num_classes)] 
    
    # Розпізнавання тестових зображень
    evaluator.recognize_test_images(test_images, class_names)

    logger.stop()
