import tensorflow as tf

def augment_image(image):
    """
    Apply simple augmentations to an image.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    random_rotate = tf.keras.layers.RandomRotation(0.2)  # Rotate up to ±20%
    image = random_rotate(image)

    return image


def augment_triplet(anchor, positive, negative):
    """
    Apply augmentations to triplet images.
    """
    return augment_image(anchor), augment_image(positive), augment_image(negative)
