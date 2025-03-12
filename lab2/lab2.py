import zipfile
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATASET_ARCHIVE_PATH = os.path.join(SCRIPT_DIRECTORY,"dataset.zip")
DATASET_EXTRACT_PATH = os.path.join(SCRIPT_DIRECTORY, "dataset")

TRAIN_TEST_SPLIT = 0.8
USE_SEPARATE_DATASETS = True  # Set to True to use separate datasets (background and evaluation), False to mix them
LEARNING_RATE = 1
EPOCHS = 10

def extract_dataset(
    archive_path=DATASET_ARCHIVE_PATH, extract_path=DATASET_EXTRACT_PATH
):
    print("Starting to unpack the archive...")
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"The archive has been successfully unpacked to {extract_path}")

def load_images(root_dir):
    data = []
    targets = []
    classes = []
    class_to_idx = {}

    # Load classes (alphabets and characters)
    for alphabet_dir in sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    ):
        alphabet_path = os.path.join(root_dir, alphabet_dir)
        for character_dir in sorted(os.listdir(alphabet_path)):
            character_path = os.path.join(alphabet_path, character_dir)
            if os.path.isdir(character_path):
                class_name = f"{alphabet_dir}/{character_dir}"
                class_idx = len(classes)
                classes.append(class_name)
                class_to_idx[class_name] = class_idx

                # Add all samples for this character
                for img_file in sorted(os.listdir(character_path)):
                    if img_file.endswith(".png"):
                        img_path = os.path.join(character_path, img_file)
                        image = Image.open(img_path)
                        image = image.resize((105, 105))  # Resize to a standard size

                        # Convert to numpy array and normalize
                        img_array = np.array(image).astype(np.float32) / 255.0

                        # Normalize to [-1, 1]
                        img_array = (img_array - 0.5) / 0.5

                        data.append(img_array)
                        targets.append(class_idx)

    data = np.array(data)
    targets = np.array(targets)

    print(f"Loaded {len(data)} images with shape {data[0].shape}")

    return data, targets, classes


def split_dataset(data, targets, test_size=0.2):
    num_samples = len(data)
    num_train_samples = int(num_samples * (1 - test_size))

    indices = np.random.permutation(num_samples)

    train_indices = indices[:num_train_samples]
    val_indices = indices[num_train_samples:]

    return (
        data[train_indices],
        targets[train_indices],
        data[val_indices],
        targets[val_indices],
    )


def prepare_omniglot_dataset():
    # First, extract the dataset if it hasn't been extracted yet
    if not os.path.exists(DATASET_EXTRACT_PATH):
        extract_dataset()

    # Check if the dataset structure follows the expected format
    background_path = os.path.join(DATASET_EXTRACT_PATH, "images_background")
    evaluation_path = os.path.join(DATASET_EXTRACT_PATH, "images_evaluation")


    if USE_SEPARATE_DATASETS:
        # Use background for training and evaluation for validation
        print(
            "Using separate datasets for training (background) and validation (evaluation)"
        )
        train_data, train_targets, classes = load_images(background_path)
        val_data, val_targets, _ = load_images(evaluation_path)
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
    else:
        # Mix the data and split into 80% for training and 20% for validation
        print(
            "Mixing datasets (background and evaluation) and splitting into 80-20 for training-validation"
        )

        # Load data from both background and evaluation
        train_data_bg, train_targets_bg, classes_bg = load_images(background_path)
        val_data_eval, val_targets_eval, classes_eval = load_images(evaluation_path)

        # Concatenate the data
        full_data = np.concatenate([train_data_bg, val_data_eval], axis=0)
        full_targets = np.concatenate([train_targets_bg, val_targets_eval], axis=0)
        full_classes = list(set(classes_bg + classes_eval))  # Combine the class names

        print(f"Total dataset samples: {len(full_data)}")

        # Shuffle the data (random permutation)
        shuffled_indices = np.random.permutation(len(full_data))
        full_data = full_data[shuffled_indices]
        full_targets = full_targets[shuffled_indices]

        # Split the dataset into train and validation sets (80-20)
        num_samples = len(full_data)
        num_train_samples = int(num_samples * TRAIN_TEST_SPLIT)

        train_data = full_data[:num_train_samples]
        val_data = full_data[num_train_samples:]
        train_targets = full_targets[:num_train_samples]
        val_targets = full_targets[num_train_samples:]
        classes = full_classes

        print(
            f"Split into {len(train_data)} training and {len(val_data)} validation samples"
        )

    return train_data, train_targets, val_data, val_targets, classes



class SimpleNeuralNetwork(object):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=LEARNING_RATE):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # Ініціалізація вагів
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01

        # Ініціалізація зсувів
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, predictions, targets):
        m = targets.shape[0]
        return -np.sum(targets * np.log(predictions + 1e-9)) / m
    
    def forward(self, X):
        # Пряме поширення через прихований шар з ReLU
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)

        # Пряме поширення через вихідний шар з softmax
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.softmax(self.output_input)
        return self.output
     
    def backward(self, X, y_true):
        m = X.shape[0]
        output_error = self.output - y_true
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_output)
        
        # Оновлення вагів і зсувів
        self.weights_hidden_output -= self.lr * np.dot(self.hidden_output.T, output_error) / m
        self.bias_output -= self.lr * np.sum(output_error, axis=0, keepdims=True) / m
        self.weights_input_hidden -= self.lr * np.dot(X.T, hidden_error) / m
        self.bias_hidden -= self.lr * np.sum(hidden_error, axis=0, keepdims=True) / m
    
    def train(self, X_train, y_train, epochs=EPOCHS):
        for epoch in range(epochs):
            outputs = self.forward(X_train)
            loss = self.cross_entropy_loss(outputs, y_train)
            self.backward(X_train, y_train)
            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', learning_rate=0.01):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Ініціалізація вагів і зсувів для всіх шарів
        self.weights = []
        self.biases = []
        self.activations = []

        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
            
            if i < len(hidden_layers):
                if activation == 'relu':
                    self.activations.append(self.relu)
                elif activation == 'tanh':
                    self.activations.append(self.tanh)
            else:
                self.activations.append(self.softmax)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, predictions, targets):
        m = targets.shape[0]
        return -np.sum(targets * np.log(predictions + 1e-9)) / m
    
    def forward(self, X):
        self.layer_inputs = []  # Вхідні значення кожного шару перед активацією
        self.layer_outputs = [X]  # Вихідні значення після активації

        for i in range(len(self.weights)):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            self.layer_inputs.append(z)
            self.layer_outputs.append(self.activations[i](z))
        
        return self.layer_outputs[-1]
    
    def backward(self, X, y_true):
        m = X.shape[0]
        dL_dout = self.layer_outputs[-1] - y_true  # Градієнт на вихідному шарі
        gradients_w = []
        gradients_b = []

        for i in reversed(range(len(self.weights))):
            dL_db = np.sum(dL_dout, axis=0, keepdims=True) / m
            dL_dw = np.dot(self.layer_outputs[i].T, dL_dout) / m

            gradients_w.append(dL_dw)
            gradients_b.append(dL_db)

            if i > 0:
                if self.activations[i-1] == self.relu:
                    dL_dout = np.dot(dL_dout, self.weights[i].T) * self.relu_derivative(self.layer_inputs[i-1])
                elif self.activations[i-1] == self.tanh:
                    dL_dout = np.dot(dL_dout, self.weights[i].T) * self.tanh_derivative(self.layer_inputs[i-1])
        
        # Оновлення вагів та зміщень
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[len(self.weights) - 1 - i]
            self.biases[i] -= self.learning_rate * gradients_b[len(self.weights) - 1 - i]
    
    def train(self, X_train, y_train, epochs=10):
        for epoch in range(epochs):
            outputs = self.forward(X_train)
            loss = self.cross_entropy_loss(outputs, y_train)
            self.backward(X_train, y_train)
            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")


def plot_training_progress(losses_train, losses_val, acc_train, acc_val, model_name):
    """Будує та зберігає графіки втрат та accuracy."""
    plots_dir = os.path.join(SCRIPT_DIRECTORY, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    epochs = range(1, len(losses_train) + 1)

    # Графік втрат
    plt.figure()
    plt.plot(epochs, losses_train, label='Train Loss')
    plt.plot(epochs, losses_val, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for {model_name}')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_loss.png'))
    plt.close()

    # Графік accuracy
    plt.figure()
    plt.plot(epochs, acc_train, label='Train Accuracy')
    plt.plot(epochs, acc_val, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training Accuracy for {model_name}')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_accuracy.png'))
    plt.close()


def evaluate_accuracy(model, X, y_true):
    """Обчислення accuracy для переданих даних."""
    predictions = model.forward(X)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    return np.mean(predicted_classes == true_classes)

# Методи активації

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def parametric_leaky_relu(x, alpha):
    return np.where(x > 0, x, alpha * x)

def parametric_leaky_relu_derivative(x, alpha):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

# Функція для підбору швидкості навчання

def find_best_learning_rate(model_class, X_train, y_train, X_val, y_val, learning_rates=[0.001, 0.01, 0.1, 1]):
    best_lr = None
    best_acc = 0
    results = {}

    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        model = model_class(input_size=X_train.shape[1], hidden_size=128, output_size=y_train.shape[1],
                            learning_rate=lr)
        model.train(X_train, y_train, X_val, y_val, epochs=10, model_name=f"Test_LR_{lr}")
        acc = evaluate_accuracy(model, X_val, y_val)
        results[lr] = acc

        if acc > best_acc:
            best_acc = acc
            best_lr = lr

    print(f"Best learning rate: {best_lr} with accuracy: {best_acc:.4f}")
    return best_lr


# Оновлення методу train у SimpleNeuralNetwork та MultiLayerPerceptron

def train_with_logging(self, X_train, y_train, X_val, y_val, epochs=None, model_name="model"):
    if epochs is None:
        epochs = EPOCHS

    losses_train, losses_val = [], []
    acc_train, acc_val = [], []

    for epoch in range(epochs):
        outputs = self.forward(X_train)
        loss = self.cross_entropy_loss(outputs, y_train)
        self.backward(X_train, y_train)

        losses_train.append(loss)
        losses_val.append(self.cross_entropy_loss(self.forward(X_val), y_val))
        acc_train.append(evaluate_accuracy(self, X_train, y_train))
        acc_val.append(evaluate_accuracy(self, X_val, y_val))

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Train Acc: {acc_train[-1]:.4f}, Val Acc: {acc_val[-1]:.4f}")

    plot_training_progress(losses_train, losses_val, acc_train, acc_val, model_name)


# Додаємо новий метод до класів
SimpleNeuralNetwork.train = train_with_logging
MultiLayerPerceptron.train = train_with_logging


def measure_prediction_time(model, X_sample):
    """Вимірює час надання прогнозу мережею."""
    start_time = time.time()
    model.forward(X_sample)
    prediction_time = time.time() - start_time
    print(f"Prediction time: {prediction_time:.6f} seconds")
    return prediction_time


if __name__ == "__main__":
    # Process the dataset (викликається один раз)
    train_data, train_targets, val_data, val_targets, classes = prepare_omniglot_dataset()

    print(f"Number of classes: {len(classes)}")
    print("Example sample:")
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print("Dataset preparation complete!")

    input_size = train_data.shape[1] * train_data.shape[2]  # 105x105 -> 11025
    output_size = len(classes)

    # Підготовка даних для навчання
    X_train = train_data.reshape(len(train_data), -1)
    y_train = np.eye(output_size)[train_targets]  # One-hot encoding
    X_val = val_data.reshape(len(val_data), -1)
    y_val = np.eye(output_size)[val_targets]

    # Вибір функції активації
    # activation_function = leaky_relu  # Використовується Leaky ReLU
    # activation_function = parametric_leaky_relu  # Використовувати Parametric Leaky ReLU
    activation_function = elu  # Використовувати ELU

    # Вибір найкращої швидкості навчання
    best_lr = find_best_learning_rate(SimpleNeuralNetwork, X_train, y_train, X_val, y_val)

    # Simple Neural Network
    simple_nn = SimpleNeuralNetwork(input_size=input_size, hidden_size=128, output_size=output_size)
    print("Training Simple Neural Network...")
    simple_nn.train(X_train, y_train, X_val, y_val, model_name="SimpleNN")
    measure_prediction_time(simple_nn, X_val[:1])

    # Multi-Layer Perceptron
    mlp = MultiLayerPerceptron(input_size=input_size, hidden_layers=[128, 64], output_size=output_size, activation='relu', learning_rate=3)
    print("Training Multi-Layer Perceptron...")
    mlp.train(X_train, y_train, X_val, y_val, model_name="MLP")
    measure_prediction_time(mlp, X_val[:1])
