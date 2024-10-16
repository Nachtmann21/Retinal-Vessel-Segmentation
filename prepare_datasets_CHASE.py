# ==========================================================
#
#  This script prepares the HDF5 datasets for the CHASE database
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
# Defining the paths to the training and testing images, labels, and masks
# These paths correspond to different subfolders in the CHASEDB1 directory
# 'Fold_1' here might refer to the cross-validation fold used in training/testing split
CHASE_DIR = './CHASEDB1/Fold_1/'
# Paths for training images, ground truth labels, and border masks
original_imgs_train = CHASE_DIR + "Training/Images/"
groundTruth_imgs_train = CHASE_DIR + "Training/Labels/"
borderMasks_imgs_train = CHASE_DIR + "Training/Masks/"
# Paths for testing images, ground truth labels, and border masks
IMG_TEST_DIR = CHASE_DIR + "Testing/Images/"
groundTruth_imgs_test = CHASE_DIR + "Testing/Labels/"
MASK_TEST_DIR = CHASE_DIR + "Testing/Masks/"
# ---------------------------------------------------------------------------------------------

# Defining some global variables for image dimensions and dataset paths
channels = 3  # Number of channels for RGB images
width = 999  # Width of the images (CHASE dataset-specific)
height = 960  # Height of the images (CHASE dataset-specific)
dataset_path = "./CHASE_datasets_training_testing/"  # Output path to save the HDF5 datasets

# Function to create datasets from image directories
def get_datasets(imgs_dir, groundTruth_dir, borderMasks_dir, Nimgs):
    # Create empty numpy arrays to hold images, ground truth labels, and border masks
    imgs = np.empty((Nimgs, height, width, channels))  # For RGB images
    groundTruth = np.empty((Nimgs, height, width))  # For grayscale ground truth labels
    border_masks = np.empty((Nimgs, height, width))  # For grayscale border masks

    # Iterate through all files in the image directory
    for path, subdirs, files in os.walk(imgs_dir):
        i = 0
        for file in files:
            if not file.endswith('.jpg'):
                # Skip files that are not JPEG images
                continue
            # Load the original image
            print("original image: " + file)
            img = Image.open(imgs_dir + file)
            imgs[i] = np.asarray(img)  # Convert image to numpy array and store it

            # Load the corresponding ground truth image (assumes specific naming convention)
            groundTruth_name = file[0:9] + "_1stHO.png"
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name).convert('L')  # Convert to grayscale
            groundTruth[i] = np.asarray(g_truth)  # Store as numpy array

            # Load the corresponding border mask (assumes specific naming convention)
            border_masks_name = "mask_" + file[6:9] + ".png"
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)  # Store as numpy array

            i += 1

    # Print the range of pixel values for verification
    print("imgs max: " + str(np.max(imgs)))
    print("imgs min: " + str(np.min(imgs)))
    print("groundTruth max: " + str(np.max(groundTruth)))
    print("groundTruth min: " + str(np.min(groundTruth)))
    # Assert statements to ensure ground truth and masks have valid pixel values (0-255)
    assert (np.max(groundTruth) == 255 and np.max(border_masks) == 255)
    assert (np.min(groundTruth) == 0 and np.min(border_masks) == 0)
    print("ground truth and border masks are correctly within pixel value range 0-255 (black-white)")

    # Reshape images and labels to match expected tensor format for deep learning
    imgs = np.transpose(imgs, (0, 3, 1, 2))  # Change from (N, H, W, C) to (N, C, H, W)
    assert (imgs.shape == (Nimgs, channels, height, width))  # Ensure correct shape
    groundTruth = np.reshape(groundTruth, (Nimgs, 1, height, width))  # Add channel dimension
    border_masks = np.reshape(border_masks, (Nimgs, 1, height, width))  # Add channel dimension
    assert (groundTruth.shape == (Nimgs, 1, height, width))  # Ensure correct shape
    assert (border_masks.shape == (Nimgs, 1, height, width))  # Ensure correct shape

    # Return the prepared datasets
    return imgs, groundTruth, border_masks

# Create output directory if it does not exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Prepare the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train, groundTruth_imgs_train,
                                                                 borderMasks_imgs_train, 21)
print("saving train datasets")
# Save the training datasets to HDF5 files
write_hdf5(imgs_train, dataset_path + "CHASE_dataset_img_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "CHASE_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train, dataset_path + "CHASE_dataset_mask_train.hdf5")

# Prepare the testing datasets
img_test, groundTruth_test, mask_test = get_datasets(IMG_TEST_DIR, groundTruth_imgs_test, MASK_TEST_DIR, 7)
print("saving test datasets")
# Save the testing datasets to HDF5 files
write_hdf5(img_test, dataset_path + "CHASE_dataset_img_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "CHASE_dataset_groundTruth_test.hdf5")
write_hdf5(mask_test, dataset_path + "CHASE_dataset_mask_test.hdf5")