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
path_experiment = './' + name_experiment + '/'
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)

# Dataset setup
path_data = './AFIO/'  # Path to your AFIO dataset
dataset_name = 'AFIO'

# List of image IDs for testing
img_ids = ['IM000001', 'IM000004', 'IM000023', 'IM000024', 'IM000135']
test_imgs_original = [f"{path_data}{img_id}/{img_id}.JPG" for img_id in img_ids]

# Load and preprocess images without resizing
def preprocess_images_full_size(img_paths):
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img.transpose(0, 3, 1, 2)  # Convert to NCHW format
        img_processed = my_pre_proc(img)
        imgs.append(img_processed[0])
    return np.array(imgs)

# Load and preprocess images
full_size = (1000, 1504)  # AFIO dimensions
test_imgs_orig = preprocess_images_full_size(test_imgs_original)

# Model and patch dimensions
patch_height = 48
patch_width = 48
stride_height = 16  # Adjusted to better handle the full-size image patches
stride_width = 16
average_mode = True

# Load model
model = model_from_json(open(path_experiment + name_experiment + '_architecture.json').read())
best_last = 'best'  # Set to 'best' or 'last' depending on which weights you want to use
model.load_weights(path_experiment + name_experiment + f'_{best_last}_weights.h5')

# Prepare patches for testing using the full-sized images
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
pred_imgs = pred_imgs[:, :, 0:full_size[0], 0:full_size[1]]  # Crop back to original dimensions

# Save segmented vessel images
for i, img_id in enumerate(img_ids):
    pred_img = pred_imgs[i, 0, :, :]  # Get the single channel (vessel segmentation result)

    # Convert to 0-255 range for saving as an image
    pred_img_normalized = (pred_img * 255).astype(np.uint8)

    # Save the segmented vessel image
    save_path = f"{path_experiment}{img_id}_segmented.jpg"
    cv2.imwrite(save_path, pred_img_normalized)
    print(f"Saved segmented image for {img_id} at {save_path}")

print("\nResults saved to 'test_afio' folder")
