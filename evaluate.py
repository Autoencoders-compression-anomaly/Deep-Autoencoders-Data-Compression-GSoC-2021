import numpy as np
from sklearn.metrics import mean_squared_error


def evaluate_model(y_true, y_predicted):
    # MSE
    mse = mean_squared_error(y_true, y_predicted)
    # RMSE
    rmse = mean_squared_error(y_true, y_predicted, squared=False)
    # Residuals of the different variables [(output-input)/input]
    residuals = np.absolute((y_true - y_predicted))
    residuals = residuals.sum()/residuals.size

    print("---------------------------")
    print("MSE: {:.6f}".format(mse))
    print("RMSE: {:.6f}".format(rmse))
    print("Residual: {:.6f}".format(residuals))
    print("---------------------------")
