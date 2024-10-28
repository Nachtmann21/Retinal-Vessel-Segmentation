import os
import sys
import time
import numpy as np
import cv2
from keras.models import model_from_json

# Custom libraries
sys.path.insert(0, './lib/')
from help_functions import *
from extract_patches import recompone_overlap, get_data_testing_overlap_direct
from pre_processing import my_pre_proc

# Experiment details
name_experiment = 'test_afio'
path_experiment = '../' + name_experiment + '/'
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)

# Dataset setup
path_data = '../AFIO/'  # Path to your AFIO dataset

# List of image IDs for testing
img_ids = ['IM000001', 'IM000004', 'IM000023', 'IM000024', 'IM000135']
test_imgs_original = [f"{path_data}{img_id}/{img_id}.JPG" for img_id in img_ids]
test_masks_original = [f"{path_data}{img_id}/mask.jpg" for img_id in img_ids]  # Mask files


# Preprocessing images without resizing
def preprocess_images_full_size(img_paths):
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image at path: {img_path}")
            continue  # Skip this image and move to the next one

        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img.transpose(0, 3, 1, 2)  # Convert to NCHW format
        img_processed = my_pre_proc(img)
        imgs.append(img_processed[0])
    return np.array(imgs)


# Preprocess the masks
def preprocess_masks_full_size(mask_paths):
    masks = []
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        mask = np.expand_dims(mask, axis=0)  # Add batch dimension
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        masks.append(mask)
    return np.array(masks)

# Load and preprocess images and masks
test_imgs_orig = preprocess_images_full_size(test_imgs_original)
test_masks_orig = preprocess_masks_full_size(test_masks_original)

# Model and patch dimensions
patch_height = 48
patch_width = 48
stride_height = 8
stride_width = 8

# Load model
model = model_from_json(open(path_experiment + name_experiment + '_architecture.json').read())
best_last = 'best'  # Set to 'best' or 'last' depending on which weights you want to use
model.load_weights(path_experiment + name_experiment + f'_{best_last}_weights.h5')

# Prepare patches for testing
patches_imgs_test, new_height, new_width = get_data_testing_overlap_direct(
    test_imgs=test_imgs_orig,
    patch_height=patch_height,
    patch_width=patch_width,
    stride_height=stride_height,
    stride_width=stride_width
)

# Run predictions
start = time.time()
predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
end = time.time()
print("Inference time (in seconds): ", end - start)

# Convert patches back to full-size images
pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")
pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)

# Apply the masks to the predicted images
for i, img_id in enumerate(img_ids):
    pred_img = pred_imgs[i, 0, :, :]  # Get the single channel (vessel segmentation result)
    mask = test_masks_orig[i, 0, :, :]  # Get the mask

    # Apply the mask: set pixels outside the FOV to zero
    pred_img_masked = pred_img * (mask / 255.0)  # Ensure the mask is binary (0 and 1)

    # Convert to 0-255 range for saving as an image
    pred_img_normalized = (pred_img_masked * 255).astype(np.uint8)

    # Ensure the image is in the correct 2D format
    pred_img_normalized = np.squeeze(pred_img_normalized)

    # Save the image using PIL instead of OpenCV
    from PIL import Image
    save_path = f"{path_experiment}{img_id}_segmented_masked.jpg"
    Image.fromarray(pred_img_normalized).save(save_path)
    print(f"Saved segmented image for {img_id} at {save_path}")
