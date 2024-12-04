import os
import numpy as np
import cv2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Parameters
IMG_SIZE = (224, 224)  # Resize images to 224x224
NUM_CLASSES = 20  # Total number of individuals in the dataset
RIDB_PATH = "./RIDB"  # Path to your database

def group_images_by_person(directory):
    """
    Groups images by person ID based on filenames in the specified directory.
    Returns a dictionary mapping person IDs to their image file paths.
    """
    person_images = defaultdict(list)

    for filename in os.listdir(directory):
        if filename.lower().endswith(".jpg"):
            try:
                parts = filename.split("_")
                photo_id = parts[0]  # Example: IM000001
                person_id = parts[1].split(".")[0]  # Example: 1
                file_path = os.path.join(directory, filename)
                person_images[person_id].append(file_path)
            except IndexError:
                print(f"Filename format incorrect: {filename}")

    return person_images


def load_images_and_labels(person_images, img_size):
    """
    Loads images and their corresponding labels from the grouped images dictionary.
    """
    images = []
    labels = []
    label_mapping = {}  # Maps person ID to numeric label
    label_counter = 0

    for person_id, file_paths in sorted(person_images.items(), key=lambda x: int(x[0])):
        if person_id not in label_mapping:
            label_mapping[person_id] = label_counter
            label_counter += 1

        for img_path in file_paths:
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            images.append(img)
            labels.append(label_mapping[person_id])

    return np.array(images), np.array(labels), label_mapping


def preprocess_data():
    """
    Preprocess the images and labels for training and validation.
    """
    if not os.path.exists(RIDB_PATH):
        print(f"Path does not exist: {RIDB_PATH}")
        return None, None, None, None

    # Group images by person
    print("Grouping images by person...")
    person_images = group_images_by_person(RIDB_PATH)

    # Load images and labels
    print("Loading images and labels...")
    images, labels, label_mapping = load_images_and_labels(person_images, IMG_SIZE)
    print(f"Loaded {len(images)} images and {len(label_mapping)} classes.")

    # Normalize images
    images = images / 255.0  # Scale pixel values to [0, 1]

    # One-hot encode labels
    labels = to_categorical(labels, num_classes=NUM_CLASSES)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    # Run preprocessing and print data shapes
    X_train, X_val, y_train, y_val = preprocess_data()
    if X_train is not None:
        print("Data preprocessing completed.")
