import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_fundus_mask(image_path, output_path, erosion_iterations=2):
    # Read the original image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Apply a binary threshold to segment the fundus from the background
    _, thresholded = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask with the same dimensions as the image
    mask = np.zeros_like(gray)

    # Draw the largest contour (which should be the fundus)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Apply erosion to refine the mask
    kernel = np.ones((7, 7), np.uint8)  # Kernel size can be adjusted as needed
    mask = cv2.erode(mask, kernel, iterations=erosion_iterations)

    # Save the generated mask
    cv2.imwrite(output_path, mask)
    return mask

def process_all_images_in_afio(afio_directory):
    # Walk through all subdirectories and files in the AFIO directory
    for subdir, _, files in os.walk(afio_directory):
        for file in files:
            if file.endswith('.JPG'):
                image_path = os.path.join(subdir, file)
                mask_filename = 'mask.jpg'
                output_mask_path = os.path.join(subdir, mask_filename)

                # Generate and save the mask with erosion applied
                mask = generate_fundus_mask(image_path, output_mask_path, erosion_iterations=1)
                print(f"Generated mask for {image_path} and saved at {output_mask_path}")

# Example usage
afio_directory = '../AFIO/'  # Path to your AFIO directory
process_all_images_in_afio(afio_directory)
