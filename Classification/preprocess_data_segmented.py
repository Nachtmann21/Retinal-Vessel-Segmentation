import os
import numpy as np
import cv2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Parameters
IMG_SIZE = (224, 224)  # Resize images to 224x224
NUM_CLASSES = 20  # Total number of classes (individuals)
RIDB_PATH = "./RIDB_SEGM"  # Path to segmented vessel binary images


def group_images_by_person(directory):
    """
    Groups images by person ID based on filenames.
    """
    person_images = defaultdict(list)

    for filename in os.listdir(directory):
        if filename.lower().endswith("_bin_seg.png"):
            try:
                parts = filename.split("_")
                person_id = parts[1]  # Example: IM000001_1_bin_seg.png -> ID = 1
                file_path = os.path.join(directory, filename)
                person_images[person_id].append(file_path)
            except IndexError:
                print(f"Filename format incorrect: {filename}")

    return person_images

def load_images_and_labels(person_images, img_size):
    """
    Load binary images and their corresponding labels.
    """
    images, labels = [], []
    label_mapping = {}
    label_counter = 0

    for person_id, file_paths in sorted(person_images.items(), key=lambda x: int(x[0])):
        if person_id not in label_mapping:
            label_mapping[person_id] = label_counter
            label_counter += 1

        for img_path in file_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            img = cv2.resize(img, img_size)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels
            images.append(img_rgb)
            labels.append(label_mapping[person_id])

    return np.array(images), np.array(labels), label_mapping


def preprocess_data_segmented():
    """
    Preprocess binary segmented vessel images for training and validation.
    """
    if not os.path.exists(RIDB_PATH):
        print(f"Path does not exist: {RIDB_PATH}")
        return None, None, None, None

    print("Grouping images by person...")
    person_images = group_images_by_person(RIDB_PATH)

    print("Loading images and labels...")
    images, labels, label_mapping = load_images_and_labels(person_images, IMG_SIZE)

    print(f"Loaded {len(images)} images and {len(label_mapping)} classes.")

    images = images / 255.0  # Normalize to [0, 1]
    labels = to_categorical(labels, num_classes=NUM_CLASSES)

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("\nDataset Split Summary:")
    print(f"Training size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")
    return X_train, X_val, y_train, y_val
