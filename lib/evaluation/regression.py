from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import numpy as np


def evaluate_regression(y_test, y_pred, plot=False):
    """Evaluate performance based on MAE and RMSE."""
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    if plot is True:
        print(f'''MAE:\t{mae}\nRMSE:\t{rmse}''')
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test, y_pred, alpha=0.4)
        plt.plot(*2 * [np.arange(0, 500000)], color='r', ls='--')
        plt.xlabel('y true')
        plt.ylabel('y pred')
        plt.ylim(0, 500000)
        plt.xlim(0, 500000)
        plt.show()

    return {
        "rmse": rmse,
        "mae": mae,
    }
