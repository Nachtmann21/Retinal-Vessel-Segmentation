from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K

class TripletLoss(Loss):
    def __init__(self, alpha=0.2, name='triplet_loss'):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        pos_dist = K.sum(K.square(anchor - positive), axis=-1)
        neg_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.maximum(pos_dist - neg_dist + self.alpha, 0.0)

def build_resnet_embedding(input_shape=(224, 224, 3)):
    """
    Build ResNet-based embedding model for grayscale images converted to 3 channels.
    """
    # Load ResNet50 model with pre-trained weights
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model layers initially

    # Add custom embedding layers
    x = Flatten()(base_model.output)
    x = Dense(256, activation="relu")(x)
    embedding = Dense(128, activation="linear", name="embedding")(x)

    model = Model(inputs=base_model.input, outputs=embedding)
    return model

def build_siamese_model(input_shape=(224, 224, 3)):
    """
    Build Siamese network with triplet loss.
    """
    embedding_model = build_resnet_embedding(input_shape)

    # Define triplet inputs
    input_anchor = Input(shape=input_shape, name="anchor")
    input_positive = Input(shape=input_shape, name="positive")
    input_negative = Input(shape=input_shape, name="negative")

    # Generate embeddings
    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    # Stack embeddings for loss calculation
    stacked_embeddings = K.stack([embedding_anchor, embedding_positive, embedding_negative], axis=1)

    # Final Siamese model
    siamese_model = Model(
        inputs=[input_anchor, input_positive, input_negative], outputs=stacked_embeddings
    )
    return siamese_model, embedding_model