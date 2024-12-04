import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model


def visualize_embeddings(embedding_model, X_val, y_val):
    """
    Visualize the embeddings using t-SNE.
    """
    # Generate embeddings
    embeddings = embedding_model.predict(X_val)

    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot embeddings
    plt.figure(figsize=(8, 8))
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=np.argmax(y_val, axis=1),
        cmap='viridis',
        s=20
    )
    plt.colorbar()
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()


def test_siamese_model(siamese_model, X_val):
    """
    Test the Siamese model with example pairs.
    """
    # Select test pairs
    anchor = np.expand_dims(X_val[0], axis=0)
    positive = np.expand_dims(X_val[1], axis=0)  # Same class as anchor
    negative = np.expand_dims(X_val[10], axis=0)  # Different class from anchor

    # Predict similarity
    result = siamese_model.predict([anchor, positive, negative])
    print("Similarity Scores (Anchor-Positive-Negative):", result)


if __name__ == "__main__":
    # Load trained models
    embedding_model = load_model('embedding_model.h5')
    siamese_model = load_model('siamese_model.h5', compile=False)

    # Load preprocessed data
    from preprocess_data import preprocess_data
    X_train, X_val, y_train, y_val = preprocess_data()

    # Visualize embeddings
    print("Visualizing embeddings...")
    visualize_embeddings(embedding_model, X_val, y_val)

    # Test Siamese model
    print("Testing Siamese model...")
    test_siamese_model(siamese_model, X_val)
