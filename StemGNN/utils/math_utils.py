import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate(y,yhat):
    y = y.reshape(1, -1)
    yhat = yhat.reshape(1, -1)
    rmse = mean_squared_error(y, yhat, squared=False)
    wmape_score = np.abs(y - yhat).sum() / np.abs(y).sum()

    return rmse, wmape_score
