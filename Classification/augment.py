import numpy as np
import tensorflow as tf

def augment_image(image):
    """
    Apply simple augmentations to an image without rotation.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image


def augment_triplet(anchor, positive, negative):
    """
    Apply augmentations to triplet images.
    """
    return augment_image(anchor), augment_image(positive), augment_image(negative)

