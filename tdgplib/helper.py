import collections
import random
from collections.abc import Mapping
from typing import Callable, Optional

import gpflow
import numpy as np
import sklearn.cluster
import tensorflow as tf


class TrainTestSplit(collections.namedtuple("TrainTestSplit", ["train", "test"]), Mapping):
    def replace(self, **kwargs):
        return self._replace(**kwargs)

    def sapply(self, f, *args, **kwargs):
        return TrainTestSplit(
            f(*self.train, *args, **kwargs), f(*self.test, *args, **kwargs)
        )

    def apply(self, f, *args, **kwargs):
        return TrainTestSplit(
            f(self.train, *args, **kwargs), f(self.test, *args, **kwargs)
        )

    def unzip(*args):
        return TrainTestSplit([a.train for a in args], [a.test for a in args])

    def zip(self):
        return (TrainTestSplit(*x) for x in zip(self.train, self.test))

    @staticmethod
    def from_sklearn(sklearn_split_ret):
        return [
            TrainTestSplit(sklearn_split_ret[i], sklearn_split_ret[i + 1])
            for i in range(0, len(sklearn_split_ret), 2)
        ]

    def __getitem__(self, k):
        if k == 'train':
            return self.train
        elif k == 'test':
            return self.test
        else:
            raise IndexError(f'k={k!r} must be "train" or "test"')

    def keys(self):
        return ['train', 'test']

    def items(self):
        return [('train', self.train), ('test', self.test)]

    def values(self):
        return [self.train, self.test]

    def __len__(self):
        return 2


def initial_inducing_points(x: np.ndarray, m: int, n_init=100, **kwargs) -> np.ndarray:
    if m < x.shape[0]:
        return (
            sklearn.cluster.KMeans(m, random_state=19960111, n_init=n_init, **kwargs)
            .fit(x)
            .cluster_centers_
        )
    else:
        return np.concatenate([x, np.random.randn(m - x.shape[0], x.shape[1])], 0)


def cholesky_logdet(chol: tf.Tensor, name=None) -> tf.Tensor:
    return tf.multiply(
        tf.constant(2, dtype=gpflow.config.default_float()),
        tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1),
        name=name,
    )


def jitter_matrix(*args, **kwargs) -> tf.Tensor:
    return gpflow.config.default_jitter() * tf.eye(
        *args, **kwargs, dtype=gpflow.config.default_float()
    )


def batched_identity(n: int, batch_dims) -> np.ndarray:
    return np.repeat(
        np.eye(n, dtype=gpflow.config.default_float())[None, ...],
        np.product(batch_dims),
        axis=0,
    ).reshape((*batch_dims, n, n))


def set_all_random_seeds(random_seed):
    tf.random.set_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def from_gpflow_to_flux(m, m_flux):
    m_v, d = m.Z_v.Z.shape
    m_u, q = m.Z_u.Z.shape

    m_flux.likelihood_layer.likelihood.variance.assign(m.likelihood.variance)
    m_flux.f_layers[0].inducing_variable.inducing_variable.Z.assign(m.Z_v.Z)
    m_flux.f_layers[0].q_mu.assign(tf.transpose(tf.reshape(m.qV.mean, (q * d, m_v))))
    m_flux.f_layers[0].q_sqrt.assign(
        tf.reshape(m.qV.chol_cov * tf.ones((q, d, m_v, m_v), dtype=m.qV.chol_cov.dtype), (q * d, m_v, m_v)))
    for i, k in enumerate(m.w_kerns):
        m_flux.f_layers[0].kernel.kernels[i * d].variance.assign(k.variance)
        m_flux.f_layers[0].kernel.kernels[i * d].lengthscales.assign(k.lengthscales)

    qu_mean, qu_cov = m.compute_optimal_qu()
    qu_chol_cov = tf.linalg.cholesky(qu_cov)

    m_flux.f_layers[-1].inducing_variable.inducing_variable.Z.assign(m.Z_u.Z)
    m_flux.f_layers[-1].q_mu.assign(qu_mean * m.sigma_f)
    m_flux.f_layers[-1].q_sqrt.assign(qu_chol_cov[None, :, :] * (m.sigma_f ** 2))
    m_flux.f_layers[-1].kernel.kernel.variance.assign(m.sigma_f ** 2)


def from_flux_to_gpflow(m, m_flux):
    m_v, d = m.Z_v.Z.shape
    m_u, q = m.Z_u.Z.shape

    m.likelihood.variance.assign(m_flux.likelihood_layer.likelihood.variance)
    m.Z_v.Z.assign(m_flux.f_layers[0].inducing_variable.inducing_variable.Z)
    m.qV.mean.assign(tf.reshape(tf.transpose(m_flux.f_layers[0].q_mu), (q, d, m_v)))

    q_sqrt = tf.reduce_mean(tf.reshape(m_flux.f_layers[0].q_sqrt, (q, d, m_v, m_v)), axis=1, keepdims=True).numpy()
    # Make diagonal positive
    q_sqrt[:, :, np.arange(m_v), np.arange(m_v)] = np.abs(q_sqrt[:, :, np.arange(m_v), np.arange(m_v)])
    m.qV.chol_cov.assign(q_sqrt)

    for i, k in enumerate(m.w_kerns):
        k.variance.assign(m_flux.f_layers[0].kernel.kernels[i * d].variance)
        k.lengthscales.assign(m_flux.f_layers[0].kernel.kernels[i * d].lengthscales)

    m.Z_u.Z.assign(m_flux.f_layers[-1].inducing_variable.inducing_variable.Z)
    m.sigma_f.assign(tf.math.sqrt(m_flux.f_layers[-1].kernel.kernel.variance))


# Adapted from https://github.com/GPflow/GPflow/blob/v2.8.0/gpflow/utilities/ops.py#L131
def op_matrix(op: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], X: tf.Tensor, X2: Optional[tf.Tensor]) -> tf.Tensor:
    if X2 is None:
        X2 = X
        diff = op(X[..., :, tf.newaxis, :], X2[..., tf.newaxis, :, :])
        return diff
    Xshape = tf.shape(X)
    X2shape = tf.shape(X2)
    X = tf.reshape(X, (-1, Xshape[-1]))
    X2 = tf.reshape(X2, (-1, X2shape[-1]))
    diff = op(X[:, tf.newaxis, :], X2[tf.newaxis, :, :])
    diff = tf.reshape(diff, tf.concat((Xshape[:-1], X2shape[:-1], [Xshape[-1]]), 0))
    return diff
