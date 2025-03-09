import zipfile
import os
import numpy as np
from PIL import Image

DATASET_ARCHIVE_PATH = "./dataset.zip"
DATASET_EXTRACT_PATH = "./dataset"
TRAIN_TEST_SPLIT = 0.8
USE_SEPARATE_DATASETS = True  # Set to True to use separate datasets (background and evaluation), False to mix them


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


if __name__ == "__main__":
    # Process the dataset
    train_data, train_targets, val_data, val_targets, classes = (
        prepare_omniglot_dataset()
    )

    print(f"Number of classes: {len(classes)}")

    # Example of checking the first sample
    print("Example sample:")
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

    print("Dataset preparation complete!")
