from typing import Optional, List, Union, Tuple, Callable

import gpflow
import gpflux
import numpy as np
import tensorflow as tf
from gpflow.base import RegressionData, AnyNDArray
from gpflow.kernels import SharedIndependent
from sklearn.decomposition import PCA

from .tdgp import TDGPLayer
from .dnsgp import NonstationaryGPLayer, ScaleMixtureKernel
from .gpflux_models import DeepGP
from .. import helper

__all__ = ['make_dkl', 'make_compositional_dgp', 'make_dnsgp', 'make_tdgp_flux']
# noinspection PyTypeChecker
float_type: np.dtype = gpflow.config.default_float()


def make_dkl(
        data: RegressionData,
        inducing_output_layer: Union[tf.Tensor, AnyNDArray, Tuple[int, int]],
        num_samples: Optional[int] = None,
        nn_layers: Optional[List[tf.keras.layers.Layer]] = None,
) -> DeepGP:
    x, y = data
    num_data, input_dim = x.shape
    output_dim = y.shape[1]

    if type(inducing_output_layer) is not tuple:
        m_v, latent_dim = inducing_output_layer.shape
        inducing_variable = gpflux.helpers.construct_basic_inducing_variables(
            m_v, latent_dim, output_dim, True, inducing_output_layer
        )
    else:
        m_v, latent_dim = inducing_output_layer
        inducing_variable = gpflux.helpers.construct_basic_inducing_variables(m_v, latent_dim, output_dim, True)

    if nn_layers is None:
        nn_layers = [
            # tf.keras.layers.Dense(1000, activation="relu"),
            tf.keras.layers.Dense(500, activation="relu"),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(latent_dim, activation="linear"),
        ]
    assert nn_layers[-1].units == latent_dim

    gp_layer = gpflux.layers.GPLayer(
        gpflux.helpers.construct_basic_kernel(
            gpflow.kernels.SquaredExponential(lengthscales=tf.ones(latent_dim)),
            output_dim, True
        ),
        inducing_variable, num_data=num_data, num_latent_gps=output_dim,
        mean_function=gpflow.mean_functions.Zero(), num_samples=num_samples,
        name='output_field'
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian())
    return DeepGP(
        [*nn_layers, tf.keras.layers.BatchNormalization(), gp_layer], likelihood_layer,
        input_dim=input_dim, target_dim=output_dim, num_data=num_data
    )


def make_compositional_dgp(
        data: RegressionData,
        inducing_hidden_layer: Union[tf.Tensor, AnyNDArray, Tuple[int, int]],
        inducing_output_layer: Union[tf.Tensor, AnyNDArray, Tuple[int, int]],
        num_samples: Optional[int] = None,
) -> DeepGP:
    x, y = data
    num_data = len(x)
    output_dim = y.shape[1]
    if type(inducing_hidden_layer) is not tuple:
        pseudo_inputs_hidden = inducing_hidden_layer
        m_v, input_dim = pseudo_inputs_hidden.shape
    else:
        m_v, input_dim = inducing_hidden_layer
        pseudo_inputs_hidden = helper.initial_inducing_points(x, m_v)
    assert x.shape[1] == input_dim

    if type(inducing_output_layer) is not tuple:
        pseudo_latent_inputs = inducing_output_layer
        m_u, latent_dim = pseudo_latent_inputs.shape
    else:
        m_u, latent_dim = inducing_output_layer

    hidden_layer = gpflux.layers.GPLayer(
        gpflux.helpers.construct_basic_kernel(
            gpflow.kernels.SquaredExponential(lengthscales=tf.ones(input_dim)),
            latent_dim, True
        ),
        gpflux.helpers.construct_basic_inducing_variables(m_v, input_dim, latent_dim, True, pseudo_inputs_hidden),
        num_data=num_data, mean_function=gpflux.helpers.construct_mean_function(x, input_dim, latent_dim),
        num_samples=num_samples,
        name='hidden_field'
    )

    if type(inducing_output_layer) is tuple:
        pseudo_latent_inputs = helper.initial_inducing_points(hidden_layer.mean_function(x), m_u)

    output_layer = gpflux.layers.GPLayer(
        gpflux.helpers.construct_basic_kernel(
            gpflow.kernels.SquaredExponential(lengthscales=tf.ones(latent_dim)),
            output_dim, True
        ),
        gpflux.helpers.construct_basic_inducing_variables(m_u, latent_dim, output_dim, True, pseudo_latent_inputs),
        num_data=num_data, num_latent_gps=output_dim,
        mean_function=gpflow.mean_functions.Zero(), num_samples=num_samples,
        name='output_field'
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian())
    return DeepGP(
        [hidden_layer, output_layer], likelihood_layer,
        input_dim=input_dim, target_dim=output_dim, num_data=num_data
    )


def make_dnsgp(
        data: RegressionData,
        inducing_lengthscale_layer: Union[tf.Tensor, AnyNDArray, Tuple[int, int]],
        inducing_output_layer: Union[tf.Tensor, AnyNDArray, Tuple[int, int]],
        num_samples: Optional[int] = None,
        lengthscale_link_function: Callable = tf.exp
) -> DeepGP:
    x, y = data
    num_data = len(x)
    output_dim = y.shape[1]
    if type(inducing_lengthscale_layer) is not tuple:
        pseudo_inputs_hidden = inducing_lengthscale_layer
        m_v, input_dim = pseudo_inputs_hidden.shape
    else:
        m_v, input_dim = inducing_lengthscale_layer
        pseudo_inputs_hidden = helper.initial_inducing_points(x, m_v)
    assert x.shape[1] == input_dim

    lengthscale_layer = gpflux.layers.GPLayer(
        gpflux.helpers.construct_basic_kernel(
            gpflow.kernels.SquaredExponential(lengthscales=tf.ones(input_dim)),
            input_dim, True
        ),
        gpflux.helpers.construct_basic_inducing_variables(m_v, input_dim, input_dim, True, pseudo_inputs_hidden),
        num_data=num_data, num_latent_gps=input_dim,
        mean_function=gpflow.mean_functions.Zero(output_dim=input_dim), num_samples=num_samples,
        name='lengthscale_field'
    )

    if type(inducing_output_layer) is not tuple:
        pseudo_inputs_output = inducing_output_layer
        m_u, input_dim = pseudo_inputs_output.shape
    else:
        m_u, input_dim = inducing_output_layer
        pseudo_inputs_output = helper.initial_inducing_points(x, m_u)
    assert x.shape[1] == input_dim
    # Add the inducing lengthscales...
    pseudo_inputs_output = np.concatenate([
        pseudo_inputs_output, lengthscale_layer.mean_function(pseudo_inputs_output)
    ], axis=-1)

    standard_kernel = gpflow.kernels.SquaredExponential()
    gpflow.set_trainable(standard_kernel.lengthscales, False)
    ns_kernel = ScaleMixtureKernel(input_dim, standard_kernel, input_dim, link_function=lengthscale_link_function)

    output_layer = NonstationaryGPLayer(
        SharedIndependent(ns_kernel, output_dim),
        gpflux.helpers.construct_basic_inducing_variables(
            m_u, input_dim + input_dim, output_dim, True, pseudo_inputs_output
        ),
        num_data=num_data, num_latent_gps=output_dim,
        mean_function=gpflow.mean_functions.Zero(), num_samples=num_samples,
        name='output_field'
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian())
    return DeepGP(
        [lengthscale_layer, output_layer], likelihood_layer,
        input_dim=input_dim, target_dim=output_dim, num_data=num_data
    )


def make_tdgp_flux(
        data: RegressionData,
        inducing_lengthscale_layer: Union[tf.Tensor, AnyNDArray, Tuple[int, int]],
        inducing_output_layer: Union[tf.Tensor, AnyNDArray, Tuple[int, int]],
        num_samples: Optional[int] = None,
        diag_qmu=True
) -> DeepGP:
    x, y = data
    num_data = len(x)
    output_dim = y.shape[1]
    if type(inducing_lengthscale_layer) is not tuple:
        pseudo_inputs_hidden = inducing_lengthscale_layer
        m_v, input_dim = pseudo_inputs_hidden.shape
    else:
        m_v, input_dim = inducing_lengthscale_layer
        pseudo_inputs_hidden = helper.initial_inducing_points(x, m_v)
    assert x.shape[1] == input_dim

    if type(inducing_output_layer) is not tuple:
        pseudo_inputs_output = inducing_output_layer
        m_u, latent_dim = pseudo_inputs_output.shape
    else:
        m_u, latent_dim = inducing_output_layer

    if diag_qmu:
        assert latent_dim == input_dim
        latent_gp_dim = input_dim
        lengthscale_layer_kernel = gpflux.helpers.construct_basic_kernel(
            gpflow.kernels.SquaredExponential(lengthscales=tf.ones(input_dim)),
            latent_gp_dim, True
        )
    else:
        latent_gp_dim = input_dim * latent_dim
        lengthscale_layer_kernel = gpflux.helpers.construct_basic_kernel(
            [
                k
                for _ in range(latent_dim)
                for k in input_dim * [gpflow.kernels.SquaredExponential(lengthscales=tf.ones(input_dim))]
            ],
            latent_gp_dim, False
        )

    lengthscale_layer = gpflux.layers.GPLayer(
        lengthscale_layer_kernel,
        gpflux.helpers.construct_basic_inducing_variables(m_v, input_dim, latent_gp_dim, True, pseudo_inputs_hidden),
        num_data=num_data, num_latent_gps=latent_gp_dim,
        mean_function=gpflow.mean_functions.Zero(output_dim=latent_gp_dim), num_samples=num_samples,
        whiten=False,
        name='inverse_lengthscale_field'
    )
    if diag_qmu:
        lengthscale_layer.q_mu.assign(1 / np.std(x) * tf.ones((m_v, input_dim)))
    else:
        pca_components = PCA(latent_dim).fit(x).components_.astype(float_type).reshape(-1)
        lengthscale_layer.q_mu.assign(np.repeat(pca_components[None, :], m_v, axis=0))

    if type(inducing_output_layer) is tuple:
        if diag_qmu:
            pseudo_inputs_output = helper.initial_inducing_points(x / np.std(x), m_u)
        else:
            pseudo_inputs_output = helper.initial_inducing_points(
                x @ pca_components.reshape(latent_dim, input_dim).T,
                m_u
            )

    standard_kernel = gpflow.kernels.SquaredExponential()
    gpflow.set_trainable(standard_kernel.lengthscales, False)

    output_layer = TDGPLayer(
        gpflux.helpers.construct_basic_kernel(standard_kernel, output_dim, True),
        gpflux.helpers.construct_basic_inducing_variables(m_u, input_dim, output_dim, True, pseudo_inputs_output),
        num_data=num_data, input_dim=input_dim, num_latent_gps=output_dim, diag_qmu=diag_qmu,
        mean_function=gpflow.mean_functions.Zero(), num_samples=num_samples,
        whiten=False,
        name='output_field'
    )

    f_layers = [lengthscale_layer, output_layer]
    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian())
    return DeepGP(
        f_layers, likelihood_layer,
        input_dim=input_dim, target_dim=output_dim, num_data=num_data
    )
