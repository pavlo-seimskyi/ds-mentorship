from sklearn.preprocessing import StandardScaler


def standard_scale(X_train, X_valid):
    """Fits X_train and transforms X_valid using standard scaler."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    return X_train, X_valid
