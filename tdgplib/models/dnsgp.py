from typing import Tuple, Optional

import gpflow
import gpflux
import tensorflow as tf
from gpflow import Parameter
from gpflow.base import TensorType
from gpflow.functions import MeanFunction
from gpflow.inducing_variables import MultioutputInducingVariables
from gpflow.kernels import SharedIndependent

from ..helper import op_matrix

__all__ = ['NonstationaryGPLayer', 'ScaleMixtureKernel']


# noinspection PyPep8Naming
class ScaleMixtureKernel(gpflow.kernels.Kernel):
    def __init__(self, input_dim, base_kernel, lengthscales_dim, scaling_offset=0.0, link_function=tf.exp):
        """
        base_kernel.input_dim == data_dim: dimension of original (first-layer) input
        lengthscales_dim: output dimension of previous layer
        For the nonstationary deep GP, the `positivity` is actually part of the *model*
        specification and does change model behaviour.
        `scaling_offset` behaves equivalent to a constant mean function in the previous
        layer; when giving the previous layer a constant mean function, it should be set
        to non-trainable.
        When the positivity is chosen to be exp(), then the scaling offset is also
        equivalent to the base_kernel lengthscale, hence the default is 0 and non-trainable.
        For other positivity choices, it may be useful to enable scaling_offset to be trainable.
        Scaling offset can be scalar or be a vector of length `data_dim`.
        """
        self.input_dim = input_dim + lengthscales_dim
        self.data_dim = input_dim
        self.lengthscales_dim = lengthscales_dim
        if not isinstance(base_kernel, gpflow.kernels.Stationary):
            raise TypeError("base kernel must be stationary")
        # must call super().__init__() before adding parameters:
        super().__init__(slice(input_dim + lengthscales_dim))

        if lengthscales_dim not in (self.data_dim, 1):
            raise ValueError("lengthscales_dim must be equal to base_kernel's input_dim or 1")

        self.base_kernel = base_kernel
        self.scaling_offset = Parameter(scaling_offset)
        self.link_function = link_function

    def _split_scale(self, X):
        if X is None:
            return None, None
        original_input = X[..., :self.data_dim]
        log_lengthscales = X[..., self.data_dim:]
        lengthscales = self.link_function(log_lengthscales + self.scaling_offset)
        return original_input, lengthscales

    @staticmethod
    def _avg_lengthscales(l1, l2=None) -> TensorType:
        return tf.divide(op_matrix(tf.add, tf.square(l1), tf.square(l2) if l2 is not None else None), 2)

    def _prefactor(self, l1, l2=None, avg_lengthscales=None) -> TensorType:
        if avg_lengthscales is None:
            avg_lengthscales = self._avg_lengthscales(l1, l2)
        if self.lengthscales_dim == self.data_dim:
            return tf.reduce_prod(
                tf.sqrt(op_matrix(tf.multiply, l1, l2) / avg_lengthscales),
                axis=-1
            )
        else:
            return tf.pow(tf.sqrt(op_matrix(tf.multiply, l1, l2) / avg_lengthscales)[..., 0], self.data_dim)

    @staticmethod
    def _quadratic(X, X2, avg_lengthscales):
        return tf.reduce_sum(tf.square(gpflow.utilities.ops.difference_matrix(X, X2)) / avg_lengthscales, axis=-1)

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        X, l1 = self._split_scale(X)
        X2, l2 = self._split_scale(X2)
        avg_lengthscales = self._avg_lengthscales(l1, l2)
        prefactor = self._prefactor(l1, l2, avg_lengthscales)
        r2 = self._quadratic(X, X2, avg_lengthscales)
        return prefactor * self.base_kernel.K_r2(r2)

    def K_diag(self, X):
        return self.base_kernel.K_diag(X)


class NonstationaryGPLayer(gpflux.layers.GPLayer):
    def __init__(
            self,
            kernel: SharedIndependent,
            inducing_variable: MultioutputInducingVariables,
            num_data: int,
            mean_function: Optional[MeanFunction] = None,
            **kwargs
    ):
        assert all(isinstance(k, ScaleMixtureKernel) for k in kernel.latent_kernels), \
            'Kernels must be ScaleMixtureKernel'
        assert all(ip.Z.shape[1] == kernel.latent_kernels[0].input_dim for ip in inducing_variable.inducing_variables), \
            f'Pseudo-points dimension must be {kernel.latent_kernels[0].input_dim}: ' \
            f'{[ip.Z.shape[1] for ip in inducing_variable.inducing_variables]}'
        super().__init__(kernel, inducing_variable, num_data, mean_function, **kwargs)
        self.data_dim = kernel.latent_kernels[0].data_dim

    def predict(
            self,
            concat_inputs: TensorType,
            *,
            full_cov: bool = False,
            full_output_cov: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        inputs = concat_inputs[..., :self.data_dim]
        mean_function = self.mean_function(inputs)
        mean_cond, cov = gpflow.conditionals.conditional(
            concat_inputs,
            self.inducing_variable,
            self.kernel,
            self.q_mu,
            q_sqrt=self.q_sqrt,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            white=self.whiten,
        )
        return mean_cond + mean_function, cov
