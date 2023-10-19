import numpy as np


def root_mean_squared_error(observed_value, predict_mean):
    se = (observed_value - predict_mean) ** 2
    return np.sqrt(np.mean(se, axis=0))


def negative_log_predictive_density(observed_value, predict_mean, predict_var):
    inner = np.log(predict_var) + (observed_value - predict_mean) ** 2 / predict_var
    return 0.5 * (np.log(2 * np.pi) + np.mean(inner, axis=0))


def mean_relative_absolute_error(observed_value, predicted_mean):
    ae = np.abs(observed_value - predicted_mean)
    return np.mean(ae / np.abs(observed_value), axis=0)
