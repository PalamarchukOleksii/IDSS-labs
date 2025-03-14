import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

DATASET_ARCHIVE = "dataset.zip"
DATASET_DIRECTORY = "dataset"
DATASET_NAME = "fashion-mnist"

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATASET_ARCHIVE_PATH = os.path.join(SCRIPT_DIRECTORY, DATASET_ARCHIVE)
DATASET_EXTRACT_PATH = os.path.join(SCRIPT_DIRECTORY, DATASET_DIRECTORY)

TRAIN_CSV_FILE = os.path.join(DATASET_EXTRACT_PATH, f"{DATASET_NAME}_train.csv")
TEST_CSV_FILE = os.path.join(DATASET_EXTRACT_PATH, f"{DATASET_NAME}_test.csv")

LEARNING_RATE = 0.1
EPOCHS = 100


def extract_dataset(archive_path=DATASET_ARCHIVE_PATH, extract_path=SCRIPT_DIRECTORY):
    print("Starting to unpack the archive...")
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    with zipfile.ZipFile(archive_path) as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"The archive has been successfully unpacked to {extract_path}")


def load_dataset(csv_path, normalize=True, one_hot=True):
    data = pd.read_csv(csv_path)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    if normalize:
        X = X / 255.0

    if one_hot:
        y = np.array(y)
        num_classes = len(np.unique(y))
        y = np.eye(num_classes)[y]

    return X, y


class MathFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1.0, 0.0)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def cross_entropy_loss(predictions, targets):
        m = targets.shape[0]
        return -np.sum(targets * np.log(np.clip(predictions, 1e-9, 1))) / m

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def parametric_leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def parametric_leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def elu_derivative(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))


class NeuralNetwork(object):
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        model_name="model",
        learning_rate=0.01,
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.model_name = model_name

        self.activation_functions = {
            "relu": (MathFunctions.relu, MathFunctions.relu_derivative),
            "tanh": (MathFunctions.tanh, MathFunctions.tanh_derivative),
            "leaky_relu": (
                MathFunctions.leaky_relu,
                MathFunctions.leaky_relu_derivative,
            ),
            "elu": (MathFunctions.elu, MathFunctions.elu_derivative),
        }

        self.weights = []
        self.biases = []

        prev_layer_size = input_size

        for layer_size, _ in hidden_layers:
            w = np.random.normal(0, 0.1, (prev_layer_size, layer_size))
            b = np.zeros((1, layer_size))

            self.weights.append(w)
            self.biases.append(b)

            prev_layer_size = layer_size

        w = np.random.normal(0, 0.1, (prev_layer_size, output_size))
        b = np.zeros((1, output_size))

        self.weights.append(w)
        self.biases.append(b)

        self.layer_inputs = []
        self.layer_outputs = []

    def forward(self, X):
        self.layer_inputs = []
        self.layer_outputs = [X]

        for i, (_, activation_name) in enumerate(self.hidden_layers):
            activation_func, _ = self.activation_functions[activation_name]

            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            self.layer_inputs.append(z)

            a = activation_func(z)
            self.layer_outputs.append(a)

        z = np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]
        self.layer_inputs.append(z)

        output = MathFunctions.softmax(z)
        self.layer_outputs.append(output)

        return output

    def backward(self, X, y, output):
        m = X.shape[0]

        loss = MathFunctions.cross_entropy_loss(output, y)

        dz = output - y

        dw = []
        db = []

        dw_last = np.dot(self.layer_outputs[-2].T, dz) / m
        db_last = np.sum(dz, axis=0, keepdims=True) / m

        dw.insert(0, dw_last)
        db.insert(0, db_last)

        da_prev = np.dot(dz, self.weights[-1].T)

        for i in range(len(self.hidden_layers) - 1, -1, -1):
            _, derivative_func = self.activation_functions[self.hidden_layers[i][1]]

            dz = da_prev * derivative_func(self.layer_inputs[i])

            if i == 0:
                dw_i = np.dot(X.T, dz) / m
            else:
                dw_i = np.dot(self.layer_outputs[i].T, dz) / m

            db_i = np.sum(dz, axis=0, keepdims=True) / m

            dw.insert(0, dw_i)
            db.insert(0, db_i)

            if i > 0:
                da_prev = np.dot(dz, self.weights[i].T)

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw[i]
            self.biases[i] -= self.learning_rate * db[i]

        return loss

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=EPOCHS,
        verbose=True,
    ):
        losses_train, losses_val = [], []
        acc_train, acc_val = [], []

        for epoch in range(epochs):
            output = self.forward(X_train)
            loss = self.backward(X_train, y_train, output)

            losses_train.append(loss)
            losses_val.append(
                MathFunctions.cross_entropy_loss(self.forward(X_val), y_val)
            )
            acc_train.append(self.evaluate(X_train, y_train))
            acc_val.append(self.evaluate(X_val, y_val))

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}, Train Acc: {acc_train[-1]:.4f}, Val Acc: {acc_val[-1]:.4f}"
                )

        plot_training_progress(
            losses_train, losses_val, acc_train, acc_val, self.model_name
        )

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_classes)

        return accuracy


def plot_training_progress(losses_train, losses_val, acc_train, acc_val, model_name):
    plots_dir = os.path.join(SCRIPT_DIRECTORY, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    epochs = range(1, len(losses_train) + 1)

    plt.figure()
    plt.plot(epochs, losses_train, label="Train Loss")
    plt.plot(epochs, losses_val, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training Loss for {model_name}")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{model_name}_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, acc_train, label="Train Accuracy")
    plt.plot(epochs, acc_val, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Training Accuracy for {model_name}")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{model_name}_accuracy.png"))
    plt.close()


def find_best_learning_rate(
    model_class, X_train, y_train, X_val, y_val, learning_rates=[0.001, 0.01, 0.1, 1]
):
    best_lr = None
    best_acc = 0
    results = {}

    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        model = model_class(
            input_size=X_train.shape[1],
            hidden_size=128,
            output_size=y_train.shape[1],
            learning_rate=lr,
        )
        model.train(
            X_train, y_train, X_val, y_val, epochs=10, model_name=f"Test_LR_{lr}"
        )
        acc = model.evaluate(model, X_val, y_val)
        results[lr] = acc

        if acc > best_acc:
            best_acc = acc
            best_lr = lr

    print(f"Best learning rate: {best_lr} with accuracy: {best_acc:.4f}")
    return best_lr


class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.elapsed_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    def get_elapsed_time(self):
        return self.elapsed_time

    def print_elapsed_time(self, message=""):
        print(f"Time taken for {message}: {self.elapsed_time:.4f} seconds")


if __name__ == "__main__":
    extract_dataset()

    X_train, y_train = load_dataset(TRAIN_CSV_FILE)
    X_test, y_test = load_dataset(TEST_CSV_FILE)

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    print("\n1. Training a basic network with one hidden layer (ReLU)...")
    basic_model = NeuralNetwork(
        input_size=input_size,
        hidden_layers=[(10, "relu")],
        output_size=output_size,
        learning_rate=LEARNING_RATE,
    )

    timer = Timer()
    timer.start()
    basic_history = basic_model.train(X_train, y_train, X_test, y_test, epochs=1000)
    timer.stop()
    timer.print_elapsed_time("training")

    timer.start()
    train_accuracy = basic_model.evaluate(X_train, y_train)
    timer.stop()
    timer.print_elapsed_time("train validation")
    print(f"Basic model - Train Accuracy: {train_accuracy:.4f}")

    timer.start()
    val_accuracy = basic_model.evaluate(X_test, y_test)
    timer.stop()
    timer.print_elapsed_time("test validation")
    print(f"Basic model - Validation Accuracy: {val_accuracy:.4f}")
