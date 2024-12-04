import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def build_resnet_embedding(input_shape=(224, 224, 3)):
    """
    Build ResNet-based embedding model.
    """
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model layers initially

    # Add custom layers for embeddings
    x = Flatten()(base_model.output)
    x = Dense(256, activation="relu")(x)
    embedding = Dense(128, activation="linear", name="embedding")(x)

    model = Model(inputs=base_model.input, outputs=embedding)
    return model


def triplet_loss(alpha=0.2):
    """
    Define triplet loss for embeddings.
    """
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        pos_dist = K.sum(K.square(anchor - positive), axis=-1)
        neg_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.maximum(pos_dist - neg_dist + alpha, 0.0)
    return loss


def build_siamese_model(input_shape=(224, 224, 3)):
    """
    Build Siamese network with triplet loss.
    """
    embedding_model = build_resnet_embedding(input_shape)

    # Define inputs for triplet network
    input_anchor = Input(shape=input_shape, name="anchor")
    input_positive = Input(shape=input_shape, name="positive")
    input_negative = Input(shape=input_shape, name="negative")

    # Generate embeddings
    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    # Stack embeddings for loss calculation
    stacked_embeddings = Lambda(lambda x: K.stack(x, axis=1))(
        [embedding_anchor, embedding_positive, embedding_negative]
    )

    # Build final Siamese model
    siamese_model = Model(
        inputs=[input_anchor, input_positive, input_negative],
        outputs=stacked_embeddings
    )
    return siamese_model, embedding_model
