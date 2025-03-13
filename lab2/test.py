import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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
    def parametric_leaky_relu(x, alpha):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def parametric_leaky_relu_derivative(x, alpha):
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def elu_derivative(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))


class NeuralNetwork(object):
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate

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

    def train(self, X, y, epochs=1000, verbose=True):
        loss_history = []

        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Compute loss (cross-entropy)
            loss = MathFunctions.cross_entropy_loss(output, y)
            loss_history.append(loss)

            # Backward pass and parameter update
            self.backward(X, y, output)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

        return loss_history

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_classes)

        return accuracy


# Get the directory where the current Python script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV files (adjust paths as needed)
train_csv_file = os.path.join(script_dir, "a-z/a-z-train.csv")
test_csv_file = os.path.join(script_dir, "a-z/a-z-test.csv")

# Load the datasets
train_data = pd.read_csv(train_csv_file)
test_data = pd.read_csv(test_csv_file)

# Separate features and labels for train and test datasets
X_train = train_data.iloc[:, 1:].values  # All columns except the first (label)
y_train = train_data.iloc[:, 0].values  # First column as the label

X_test = test_data.iloc[:, 1:].values  # All columns except the first (label)
y_test = test_data.iloc[:, 0].values  # First column as the label

# Normalize features (if needed, depending on the dataset)
X_train = X_train / 255.0  # Assuming pixel values in the range [0, 255]
X_test = X_test / 255.0

# Convert labels to one-hot encoding (assuming 10 classes for Fashion dataset)
num_classes = len(np.unique(y_train))  # This should be 10 for Fashion MNIST labels
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]

input_size = X_train.shape[1]  # This is the number of features (pixels)
output_size = num_classes  # 10 classes for Fashion MNIST

# Initialize and train the model
model = NeuralNetwork(
    input_size=input_size,
    hidden_layers=[(10, "relu")],
    output_size=output_size,
    learning_rate=0.1,
)

# Train the model on the dataset
loss_history = model.train(X_train, y_train_one_hot, epochs=1000, verbose=True)

# Evaluate the model on the test set
accuracy = model.evaluate(X_test, y_test_one_hot)
print(f"Test accuracy: {accuracy:.4f}")
