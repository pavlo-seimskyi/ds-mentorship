from sklearn.cluster import KMeans
import numpy as np


def apply_k_means(X_train, X_valid, ncols, n_clusters=3):
    """Fits X_train and predicts on X_valid.
    Adds the clusters as new columns to the right."""
    km = KMeans(n_clusters=n_clusters, random_state=42)
    X_train_clusters = km.fit(X_train[:, :ncols]).labels_.reshape(-1, 1)
    X_valid_clusters = km.predict(X_valid[:, :ncols]).reshape(-1, 1)

    X_train = np.concatenate((X_train, X_train_clusters), axis=1)
    X_valid = np.concatenate((X_valid, X_valid_clusters), axis=1)

    return X_train, X_valid
