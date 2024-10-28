# ==========================================================
#
#  This script prepares the HDF5 datasets for the AFIO database
#
# ============================================================

import os
import h5py
import numpy as np
from PIL import Image


# Function to write numpy arrays to an HDF5 file
def write_hdf5(arr, outfile):
    # Create an HDF5 file and store the data array
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


# ------------ Path of the images --------------------------------------------------------------
AFIO_DIR = './AFIO/'

# Paths for training images and ground truth labels
original_imgs_train = AFIO_DIR + "Training/Images/"
groundTruth_imgs_train = AFIO_DIR + "Training/Labels/"

# Paths for testing images and ground truth labels
IMG_TEST_DIR = AFIO_DIR + "Testing/Images/"
groundTruth_imgs_test = AFIO_DIR + "Testing/Labels/"
# ---------------------------------------------------------------------------------------------

# Defining some global variables for image dimensions and dataset paths
channels = 3  # Number of channels for RGB images
width = 1504  # Update with AFIO-specific width
height = 1000  # Update with AFIO-specific height
dataset_path = "./AFIO_datasets_training_testing/"  # Output path to save the HDF5 datasets



# Function to create datasets from image directories
# def get_datasets(imgs_dir, ground_truth_dir, n_imgs):
#     # Create empty numpy arrays to hold images and ground truth labels
#     imgs = np.empty((n_imgs, height, width, channels))  # For RGB images
#     ground_truth = np.empty((n_imgs, height, width))  # For grayscale ground truth labels
#     # border_masks = np.empty((n_imgs, height, width))  # For grayscale border masks
#
#     # Iterate through all files in the image directory
#     for path, subdirs, files in os.walk(imgs_dir):
#         i = 0
#         for file in files:
#             if not file.endswith('.jpg') or not file.endswith('.JPG'):
#                 # Skip files that are not JPG/jpg images
#                 continue
#
#             # Load the original image
#             print("original image: " + file)
#             img = Image.open(imgs_dir + file)
#             imgs[i] = np.asarray(img)  # Convert image to numpy array and store it
#
#             # Load the corresponding ground truth image (modify naming convention if needed)
#             ground_truth_name = file.replace(".jpg", "--vessels.jpg")  # Example naming convention
#             print("ground truth name: " + ground_truth_name)
#             g_truth = Image.open(ground_truth_dir + ground_truth_name).convert('L')  # Convert to grayscale
#             ground_truth[i] = np.asarray(g_truth)  # Store as numpy array
#
#             # TODO: Load the corresponding border mask (assumes specific naming convention) if needed
#
#             i += 1

#     # Print the range of pixel values for verification
#     print("imgs max: " + str(np.max(imgs)))
#     print("imgs min: " + str(np.min(imgs)))
#     print("groundTruth max: " + str(np.max(ground_truth)))
#     print("groundTruth min: " + str(np.min(ground_truth)))
#
#     # Ensure ground truth values are valid
#     assert (np.max(ground_truth) == 255)
#     assert (np.min(ground_truth) == 0)
#
#     # Reshape images and labels to match expected tensor format for deep learning
#     imgs = np.transpose(imgs, (0, 3, 1, 2))  # Change from (N, H, W, C) to (N, C, H, W)
#     assert (imgs.shape == (n_imgs, channels, height, width))  # Ensure correct shape
#     ground_truth = np.reshape(ground_truth, (n_imgs, 1, height, width))  # Add channel dimension
#     assert (ground_truth.shape == (n_imgs, 1, height, width))  # Ensure correct shape
#
#     return imgs, ground_truth
#
#
# # Create output directory if it does not exist
# if not os.path.exists(dataset_path):
#     os.makedirs(dataset_path)
#
# # Prepare the training datasets
# imgs_train, groundTruth_train = get_datasets(original_imgs_train, groundTruth_imgs_train, 21)
# print("saving train datasets")
# write_hdf5(imgs_train, dataset_path + "AFIO_dataset_img_train.hdf5")
# write_hdf5(groundTruth_train, dataset_path + "AFIO_dataset_groundTruth_train.hdf5")
#
# # Prepare the testing datasets
# img_test, groundTruth_test = get_datasets(IMG_TEST_DIR, groundTruth_imgs_test, 7)
# print("saving test datasets")
# write_hdf5(img_test, dataset_path + "AFIO_dataset_img_test.hdf5")
# write_hdf5(groundTruth_test, dataset_path + "AFIO_dataset_groundTruth_test.hdf5")
