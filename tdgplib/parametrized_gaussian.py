import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import Module, Parameter

from .bijectors import FillDiagonal
from .helper import batched_identity

__alL__ = ['MeanFieldGaussian']

float_type = gpflow.config.default_float()

covariance_matrix_bijector = tfp.bijectors.FillScaleTriL(diag_bijector=gpflow.utilities.positive(), diag_shift=None)
diagonal_variance_bijector = tfp.bijectors.Chain([
    FillDiagonal(),
    gpflow.utilities.positive()
], name='FillPositiveDiagonal')


class MeanFieldGaussian(Module):
    def __init__(
            self, n, batch_dims=(), full_cov=True, shared_cov=None, name=None
    ):
        super().__init__(name=name)
        if shared_cov is None:
            shared_cov = set()
        assert shared_cov.intersection(range(len(batch_dims))) == shared_cov, "shared_cov doesn't match batch_dims"

        dims = (*batch_dims, n)
        self.mean = Parameter(np.zeros(dims, dtype=float_type))

        shared_batch = tuple(
            d if i not in shared_cov else 1
            for i, d in enumerate(batch_dims)
        )

        if full_cov:
            self.chol_cov = Parameter(
                batched_identity(n, shared_batch),
                transform=covariance_matrix_bijector,
            )
        else:
            self.chol_cov = Parameter(
                batched_identity(n, shared_batch),
                transform=diagonal_variance_bijector,
            )

    @property
    def cov(self) -> tf.Tensor:
        return tf.matmul(self.chol_cov, self.chol_cov, transpose_b=True, name="cov")
