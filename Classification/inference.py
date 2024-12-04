import numpy as np
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine


def build_embedding_database(embedding_model, X_train, y_train, num_samples=10):
    """
    Build a database of embeddings for known images.
    """
    database = {}
    for i in range(num_samples):
        image = np.expand_dims(X_train[i], axis=0)
        label = np.argmax(y_train[i])
        embedding = embedding_model.predict(image)
        database[label] = embedding
    return database


def query_image(database, embedding_model, query_image):
    """
    Query the database with a new image.
    """
    query_embedding = embedding_model.predict(np.expand_dims(query_image, axis=0))

    # Compare with known embeddings
    similarities = {}
    for label, embedding in database.items():
        similarity = 1 - cosine(query_embedding, embedding)
        similarities[label] = similarity

    # Sort results by similarity
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


if __name__ == "__main__":
    # Load trained embedding model
    embedding_model = load_model('embedding_model.h5')

    # Load preprocessed data
    from preprocess_data import preprocess_data
    X_train, X_val, y_train, y_val = preprocess_data()

    # Build database of known embeddings
    print("Building embedding database...")
    database = build_embedding_database(embedding_model, X_train, y_train)

    # Query a new image
    print("Querying with a new image...")
    query_results = query_image(database, embedding_model, X_val[0])
    print("Query Results (Label: Similarity):", query_results)
