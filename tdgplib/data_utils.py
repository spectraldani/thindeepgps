import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import sklearn.model_selection

from tdgplib.helper import TrainTestSplit


def scale_data(x, y):
    scaler = SimpleNamespace(x=sklearn.preprocessing.StandardScaler(), y=sklearn.preprocessing.StandardScaler())
    scaler.x.fit(x.train)
    x = x.apply(lambda x: scaler.x.transform(x))
    scaler.y.fit(y.train)
    y = y.apply(lambda y: scaler.y.transform(y))
    return x, y, scaler


def get_folds(x_all, y_all, n_fold, random_seed):
    kfold = sklearn.model_selection.KFold(max(2, n_fold), shuffle=True, random_state=random_seed)
    # noinspection PyArgumentList
    folds = [
        [TrainTestSplit(x_all[train], x_all[test]), TrainTestSplit(y_all[train], y_all[test])]
        for train, test in kfold.split(x_all, y_all)
    ]
    return folds[:n_fold]


def get_parabolas(random_seed, n_fold=2):
    s = 0.4
    n = 500
    rng = np.random.default_rng(19960111)
    m1, m2 = np.array([[-1, 1], [2, 1]])
    X1 = rng.multivariate_normal(m1, s * np.eye(2), size=n // 2)
    X2 = rng.multivariate_normal(m2, s * np.eye(2), size=n // 2)
    y1 = X1[:, 0] ** 2 + X1[:, 0]
    y2 = X2[:, 1] ** 2 + X2[:, 1]

    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([y1, y2], axis=0)[:, None]
    for X, y in get_folds(X, y, n_fold=n_fold, random_seed=random_seed):
        X, y, scalers = scale_data(X, y)
        yield X, y, scalers


def get_sinc(random_seed, n_fold=2, num_data=500):
    rng = np.random.default_rng(19960111)

    X = rng.random((num_data, 2)) * 2 - 1
    W = np.stack([
        np.sin(3.1414 * X[..., 0]) * X[..., 0] * 2,
        np.cos(3.1414 * X[..., 0]) * 2,
    ], axis=-1)[..., None]
    Z = (X[:, None] @ W)[..., 0, 0]
    y = (np.sinc(Z) - Z ** 2)[:, None]
    for X, y in get_folds(X, y, n_fold=n_fold, random_seed=random_seed):
        X, y, scalers = scale_data(X, y)
        yield X, y, scalers


def from_pickle(path, random_seed, n_fold=2):
    """
    Returns the dataset in `path` relative to the current working directory
    """
    with open(path, "rb") as f:
        X, y = pickle.load(f)
    if X.shape[0] > 1000 * n_fold:
        rng = np.random.default_rng(random_seed)
        warnings.warn(f'Dataset of size {X.shape[0]} is too large, subsampling to 1000 per fold')
        idx = rng.choice(len(X), 1000 * n_fold, replace=False)
        X, y = X[idx], y[idx]

    y = y.reshape(-1, 1)
    for X, y in get_folds(X, y, n_fold, random_seed):
        X, y, scalers = scale_data(X, y)
        yield X, y, scalers
