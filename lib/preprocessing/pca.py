from sklearn.decomposition import PCA
import numpy as np


def apply_pca(X_train, X_valid, ncols, n_components=3):
    """Fits X_train and transforms X_valid."""
    pca = PCA(n_components=n_components)
    pc_X_train = pca.fit_transform(X_train[:, :ncols])
    pc_X_valid = pca.transform(X_valid[:, :ncols])

    X_train = np.concatenate((X_train, pc_X_train), axis=1)
    X_valid = np.concatenate((X_valid, pc_X_valid), axis=1)

    return X_train, X_valid
