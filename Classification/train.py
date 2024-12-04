import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import build_siamese_model, triplet_loss
from augment import augment_triplet
from preprocess_data import preprocess_data


def data_generator(X, batch_size=16):
    """
    Generate triplets (anchor, positive, negative) for training.
    """
    while True:
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        # Select indices for anchor and positive
        anchor_idx = indices[:batch_size]
        positive_idx = anchor_idx  # Positive matches anchor

        # Select random negatives
        negative_idx = np.random.choice(indices, batch_size, replace=False)

        anchors = X[anchor_idx]
        positives = X[positive_idx]
        negatives = X[negative_idx]

        yield [anchors, positives, negatives], np.zeros((batch_size,))


def train_model():
    """
    Train the Siamese model with triplet loss.
    """
    # Load preprocessed data
    X_train, X_val, y_train, y_val = preprocess_data()

    # Build Siamese model
    siamese_model, embedding_model = build_siamese_model()

    # Compile model
    siamese_model.compile(optimizer=Adam(1e-4), loss=triplet_loss())

    # Train model
    train_gen = data_generator(X_train)
    val_gen = data_generator(X_val)

    siamese_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        steps_per_epoch=len(X_train) // 16,
        validation_steps=len(X_val) // 16
    )

    # Save models
    siamese_model.save("siamese_model.h5")
    embedding_model.save("embedding_model.h5")
    print("Model training completed and saved!")


if __name__ == "__main__":
    train_model()
