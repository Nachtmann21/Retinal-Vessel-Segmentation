import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from preprocess_data_segmented import preprocess_data_segmented
from model import build_siamese_model, TripletLoss
from augment import augment_image


def data_generator(X, y, batch_size=16):
    """
    Generate triplets (anchor, positive, negative) for training.
    """
    while True:
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        anchors, positives, negatives = [], [], []

        for _ in range(batch_size):
            anchor_idx = np.random.choice(indices)
            anchor_label = np.argmax(y[anchor_idx])

            # Select positive sample
            positive_idx = np.random.choice(indices[y.argmax(axis=1) == anchor_label])

            # Select negative sample
            negative_idx = np.random.choice(indices[y.argmax(axis=1) != anchor_label])

            anchors.append(augment_image(X[anchor_idx]))
            positives.append(augment_image(X[positive_idx]))
            negatives.append(augment_image(X[negative_idx]))

        yield [np.stack(anchors), np.stack(positives), np.stack(negatives)], np.zeros((batch_size,))


def train_model():
    """
    Train the Siamese model.
    """
    X_train, X_val, y_train, y_val = preprocess_data_segmented()

    # Build Siamese model
    siamese_model, embedding_model = build_siamese_model(input_shape=(224, 224, 3))
    siamese_model.compile(optimizer=Adam(1e-4), loss=TripletLoss())

    # Data generators
    train_gen = data_generator(X_train, y_train, batch_size=16)
    val_gen = data_generator(X_val, y_val, batch_size=16)

    # Early stopping
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # Train model
    siamese_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        steps_per_epoch=len(X_train) // 16,
        validation_steps=len(X_val) // 16,
        callbacks=[early_stopping],
    )

    # Save models
    embedding_model.save("embedding_model.h5")
    siamese_model.save("siamese_model.h5")
    print("Training completed and models saved!")


if __name__ == "__main__":

    train_model()
