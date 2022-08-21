import numpy as np
from sklearn.base import clone
import pandas as pd


def perform_cross_validation(X, y, model, preprocess_func, eval_func, cv_folds=10, **kwargs):
    """Run cross-validation with preprocessing on a specified model."""
    original_model = clone(model)
    total_rows = X.shape[0]
    metric_track = None
    splits = generate_cv_splits(total_rows, cv_folds)

    for valid_start, valid_end in splits:
        train_idx, valid_idx = get_train_valid_idx(valid_start, valid_end, total_rows)
        X_train, y_train, X_valid, y_valid = train_valid_split(X, y, train_idx, valid_idx)
        X_train, y_train, X_valid, y_valid = preprocess_func(X_train, X_valid, y_train, y_valid, **kwargs)

        # Re-instantiate the provided model
        model_ = clone(original_model)
        model_ = model_.fit(X_train, y_train)
        y_pred_log = model_.predict(X_valid)
        y_pred = np.exp(y_pred_log)

        # Evaluate and track performance
        current_metrics = pd.DataFrame(data=[eval_func(y_valid, y_pred)])
        if metric_track is None:
            metric_track = current_metrics.copy()
        else:
            metric_track = pd.concat((metric_track, current_metrics), axis=0) # append the new result

    return dict(metric_track.mean())


def generate_cv_splits(rows, cv=10):
    """Generate a list of start & end idx for cross validation."""
    step = rows // cv
    splits = []
    for split in range(0, cv):
        start = step*split
        end = start+step
        splits.append((start,  end))
    return splits


def get_train_valid_idx(valid_start_idx, valid_end_idx, total_rows):
    """Transform validation start and end indexes into a list of training and validation indexes."""
    valid_idx = np.arange(valid_start_idx, valid_end_idx)
    all_idx = np.arange(total_rows)
    train_mask = np.isin(all_idx, valid_idx, invert=True)
    train_idx = all_idx[train_mask]
    return train_idx, valid_idx


def train_valid_split(X, y, train_idx, valid_idx):
    """Split into train/test sets, based on index location."""
    X_train = X.copy().iloc[train_idx]
    y_train = y.copy().iloc[train_idx]
    X_valid = X.copy().iloc[valid_idx]
    y_valid = y.copy().iloc[valid_idx]
    return X_train, y_train, X_valid, y_valid
