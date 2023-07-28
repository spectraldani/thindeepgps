import random
from typing import Optional, Callable

import gpflow
import numpy as np
import sklearn.cluster
import tensorflow as tf

__all__ = [
    'op_matrix', 'cholesky_logdet', 'jitter_matrix', 'batched_identity', 'set_all_random_seeds',
    'initial_inducing_points'
]


def set_all_random_seeds(random_seed):
    tf.random.set_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


# Adapted from https://github.com/GPflow/GPflow/blob/v2.8.0/gpflow/utilities/ops.py#L131
def op_matrix(op: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], x: tf.Tensor, x2: Optional[tf.Tensor]) -> tf.Tensor:
    if x2 is None:
        x2 = x
        diff = op(x[..., :, tf.newaxis, :], x2[..., tf.newaxis, :, :])
        return diff
    x_shape = tf.shape(x)
    x2_shape = tf.shape(x2)
    x = tf.reshape(x, (-1, x_shape[-1]))
    x2 = tf.reshape(x2, (-1, x2_shape[-1]))
    diff = op(x[:, tf.newaxis, :], x2[tf.newaxis, :, :])
    diff = tf.reshape(diff, tf.concat((x_shape[:-1], x2_shape[:-1], [x_shape[-1]]), 0))
    return diff


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


def initial_inducing_points(x: np.ndarray, m: int, n_init=100, **kwargs) -> np.ndarray:
    if m < x.shape[0]:
        return (
            sklearn.cluster.KMeans(m, random_state=19960111, n_init=n_init, **kwargs)
            .fit(x)
            .cluster_centers_
        )
    else:
        return np.concatenate([x, np.random.randn(m - x.shape[0], x.shape[1])], 0)
