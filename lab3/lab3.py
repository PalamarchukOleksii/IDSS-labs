import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import tensorflow as tf

DATASET_ARCHIVE = "dataset.zip"
DATASET_DIRECTORY = "dataset"
TARGET_COLUMN = "Close"
SPLIT_RATIO = 0.8

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATASET_ARCHIVE_PATH = os.path.join(SCRIPT_DIRECTORY, DATASET_ARCHIVE)
DATASET_EXTRACT_PATH = os.path.join(SCRIPT_DIRECTORY, DATASET_DIRECTORY)
CHECKPOINT_DIR = os.path.join(SCRIPT_DIRECTORY, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model.weights.h5")

INFY_STOCK_CSV_PATH = os.path.join(DATASET_EXTRACT_PATH, "infy_stock.csv")
NIFTY_IT_INDEX_CSV_PATH = os.path.join(DATASET_EXTRACT_PATH, "nifty_it_index.csv")
TCS_STOCK_CSV_PATH = os.path.join(DATASET_EXTRACT_PATH, "tcs_stock.csv")

DATASET_VARIANT = "a"
B_DATASET_CSV = INFY_STOCK_CSV_PATH

LEARNING_RATE = 0.01
EPOCH = 100
BATCH_SIZE = 32
L1_LAMDA = 0.001


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


def get_dataset_b(csv_path, target_column):
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


def get_dataset(
    variant="a",
    b_dataset_csv_path=INFY_STOCK_CSV_PATH,
    b_dataset_target_column=TARGET_COLUMN,
):
    """Return dataset based on the variant selected."""
    if variant == "a":
        return get_dataset_a()
    elif variant == "b":
        return get_dataset_b(b_dataset_csv_path, b_dataset_target_column)
    else:
        raise ValueError(f"Invalid dataset variant: {variant}")


def split_dataset(x_data, y_data):
    """Split dataset into train and test based on the split ratio."""
    split_index = int(len(x_data) * SPLIT_RATIO)

    x_train_data = x_data[:split_index]
    y_train_data = y_data[:split_index]
    x_test_data = x_data[split_index:]
    y_test_data = y_data[split_index:]

    return x_train_data, y_train_data, x_test_data, y_test_data


def shuffle_dataset(x_data, y_data):
    """Shuffle dataset by examples, keeping labels and targets in sync."""
    indices = np.random.permutation(len(x_data))
    return x_data[indices], y_data[indices]


def prepare_dataset(x_data, y_data):
    """Shuffle and split dataset."""
    x_data, y_data = shuffle_dataset(x_data, y_data)
    return split_dataset(x_data, y_data)


def plot_infy_stock():
    """Plot INFY stock data with close, open, high, low prices, volume, and deliverable percentage."""
    data = pd.read_csv(INFY_STOCK_CSV_PATH)

    data["Date"] = pd.to_datetime(data["Date"])

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data["Date"], data["Close"], "b-", label="Close", linewidth=2)
    ax1.plot(data["Date"], data["Open"], "g-", label="Open")
    ax1.plot(data["Date"], data["High"], "r--", label="High")
    ax1.plot(data["Date"], data["Low"], "k--", label="Low")
    ax1.set_title("INFY Stock Price (Jan 2015)", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Price (₹)", fontsize=12)
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(data["Date"], data["Volume"], color="orange", alpha=0.7)
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(data["Date"], data["%Deliverble"] * 100, "g-", marker="o")
    ax3.set_ylabel("Deliverable %", fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.grid(True)

    date_format = mdates.DateFormatter("%Y-%m-%d")
    ax3.xaxis.set_major_formatter(date_format)
    ax3.set_xlabel("Date", fontsize=12)

    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def plot_nifty_it_index():
    """Plot NIFTY IT index data with OHLC prices, trading volume, and turnover."""
    data = pd.read_csv(NIFTY_IT_INDEX_CSV_PATH)

    data["Date"] = pd.to_datetime(data["Date"])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("Stock Market Data Analysis (Jan 1-9, 2015)", fontsize=16)

    ax1.plot(data["Date"], data["Open"], label="Open", marker="o")
    ax1.plot(data["Date"], data["High"], label="High", marker="^")
    ax1.plot(data["Date"], data["Low"], label="Low", marker="v")
    ax1.plot(data["Date"], data["Close"], label="Close", marker="s")
    ax1.set_ylabel("Price")
    ax1.set_title("OHLC Prices")
    ax1.legend()
    ax1.grid(True)

    ax2.bar(data["Date"], data["Volume"], color="blue", alpha=0.7)
    ax2.set_ylabel("Volume")
    ax2.set_title("Trading Volume")
    ax2.grid(True)

    ax3.bar(data["Date"], data["Turnover"] / 1e9, color="green", alpha=0.7)
    ax3.set_ylabel("Turnover (Billion)")
    ax3.set_title("Trading Turnover")
    ax3.grid(True)

    plt.xticks(rotation=45)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    plt.show()


def plot_tcs_stock():
    """Plot TCS stock data with close, open, high, low prices, volume, and deliverable percentage."""
    data = pd.read_csv(TCS_STOCK_CSV_PATH)

    data["Date"] = pd.to_datetime(data["Date"])

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data["Date"], data["Close"], "b-", label="Close", linewidth=2)
    ax1.plot(data["Date"], data["Open"], "g-", label="Open")
    ax1.plot(data["Date"], data["High"], "r--", label="High")
    ax1.plot(data["Date"], data["Low"], "k--", label="Low")
    ax1.set_title("INFY Stock Price (Jan 2015)", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Price (₹)", fontsize=12)
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(data["Date"], data["Volume"], color="orange", alpha=0.7)
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(data["Date"], data["%Deliverble"] * 100, "g-", marker="o")
    ax3.set_ylabel("Deliverable %", fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.grid(True)

    date_format = mdates.DateFormatter("%Y-%m-%d")
    ax3.xaxis.set_major_formatter(date_format)
    ax3.set_xlabel("Date", fontsize=12)

    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


def plot_a_dataset():
    """Plots Dataset A, including scattered data points and the true polynomial curve."""
    x_values, y_values = get_dataset_a()

    plt.figure(figsize=(8, 6))

    plt.scatter(x_values, y_values, label="Generated Data (with noise)", s=10)

    plt.title("Dataset A Visualization")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")

    plt.grid(True)

    coef = [-10, 2, 3]
    y_true = coef[0] + coef[1] * x_values + coef[2] * np.power(x_values, 2)
    plt.plot(
        x_values,
        y_true,
        color="red",
        linestyle="--",
        label="True Polynomial (y = 3x^2 + 2x - 10)",
    )

    plt.legend()
    plt.show()


def plot_dataset(variant="a", b_dataset_csv_path=INFY_STOCK_CSV_PATH):
    """Plots a specified dataset based on the dataset variant and B dataset CSV path."""
    if variant == "a":
        plot_a_dataset()
    elif variant == "b":
        if b_dataset_csv_path == INFY_STOCK_CSV_PATH:
            plot_infy_stock()
        elif b_dataset_csv_path == NIFTY_IT_INDEX_CSV_PATH:
            plot_nifty_it_index()
        elif b_dataset_csv_path == TCS_STOCK_CSV_PATH:
            plot_tcs_stock()
        else:
            raise ValueError(f"Invalid B dataset CSV path: {b_dataset_csv_path}")
    else:
        raise ValueError(f"Invalid dataset variant: {variant}")

def tune_l1_lambda(x_train, y_train, x_val, y_val, lambdas, degree=2):
    """Підбирає найкраще значення L1 регуляризації за допомогою валідаційної вибірки."""
    results = []
    for l1 in lambdas:
        print(f"\nTraining with L1 λ = {l1}")
        model = PolynomialRegression(degree=degree)
        model.linear_layer.kernel_regularizer = tf.keras.regularizers.L1(l1)
        model.linear_layer.bias_regularizer = tf.keras.regularizers.L1(l1)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
            loss="mse",
            metrics=["mse"]
        )
        _ = model(x_train[:1])
        history = model.fit(
            x_train, y_train,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=0,
            validation_data=(x_val, y_val)
        )
        final_val_loss = history.history['val_loss'][-1]
        print(f"λ = {l1} -> val_loss = {final_val_loss:.4f}")
        results.append((l1, final_val_loss))

    # Обрати найкращу λ
    best_lambda = min(results, key=lambda x: x[1])[0]
    print(f"\nBest L1 λ: {best_lambda}")
    return best_lambda


class PolynomialRegression(tf.keras.Model):

    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.poly_layer = tf.keras.layers.Lambda(
            lambda x: tf.concat([x**i for i in range(degree + 1)], axis=1))
        self.linear_layer = tf.keras.layers.Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.L1(L1_LAMDA),
            bias_regularizer=tf.keras.regularizers.L1(L1_LAMDA))

    def call(self, inputs):
        x_poly = self.poly_layer(inputs)
        return self.linear_layer(x_poly)


if __name__ == "__main__":
    plot_dataset(variant=DATASET_VARIANT, b_dataset_csv_path=B_DATASET_CSV)

    x, y = get_dataset(variant=DATASET_VARIANT, b_dataset_csv_path=B_DATASET_CSV)
    x_train, y_train, x_test, y_test = prepare_dataset(x, y)

    # Reshape and convert data
    x_train = x_train.reshape(-1, 1).astype('float32')
    y_train = y_train.reshape(-1, 1).astype('float32')
    x_test = x_test.reshape(-1, 1).astype('float32')
    y_test = y_test.reshape(-1, 1).astype('float32')

    # Додаткове розбиття: train → train + validation
    val_split = 0.2
    val_size = int(len(x_train) * val_split)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train2, y_train2 = x_train[val_size:], y_train[val_size:]

    # Автоматичний підбір найкращого L1 λ
    best_l1 = tune_l1_lambda(
        x_train2, y_train2, x_val, y_val,
        lambdas=[0.0, 0.0001, 0.001, 0.01, 0.1]
    )

    # Create and compile model with best L1 regularization
    model = PolynomialRegression(degree=2)
    model.linear_layer.kernel_regularizer = tf.keras.regularizers.L1(best_l1)
    model.linear_layer.bias_regularizer = tf.keras.regularizers.L1(best_l1)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mse']
    )

    # Callback to print loss every 10 epochs
    class PrintLossCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{EPOCH} - Loss: {logs['loss']:.4f} ")

    # Ensure directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Create callback for saving checkpoints
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # Build model manually using sample input
    _ = model(x_train[:1])  # Проганяємо один приклад для ініціалізації шарів

    # Restore weights if checkpoint exists
    if os.path.exists(CHECKPOINT_PATH):
        model.load_weights(CHECKPOINT_PATH)
        print(f"Model restored from checkpoint: {CHECKPOINT_PATH}")
    else:
        print("No checkpoint found. Training from scratch.")

    # Learning rate adjustment based on validation loss
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Train the model with mini-batch gradient descent
    history = model.fit(
        x_train, y_train,
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[
            PrintLossCallback(),
            checkpoint_callback,
            lr_scheduler,
            early_stopping
        ],
        validation_data=(x_test, y_test)
    )

    # Plot training loss with enhanced visualization
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss (MSE with L1 Regularization, λ={best_l1})', fontsize=14)
    plt.ylabel('Loss Value', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save plot with higher quality
    plot_path = os.path.join(SCRIPT_DIRECTORY, 'polynomial_regression_loss.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining loss plot saved to: {plot_path}")

    # Evaluate on test set
    test_loss, test_mse = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Test Results:")
    print(f"- Total Loss (MSE + L1): {test_loss:.4f}")
    print(f"- MSE Only: {test_mse:.4f}")
    print(f"- L1 Regularization Term: {test_loss - test_mse:.4f}")
