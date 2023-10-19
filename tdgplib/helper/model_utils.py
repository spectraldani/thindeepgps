from typing import Union

import gpflow
import gpflux
import numpy as np
import tensorflow as tf

__all__ = ['get_likelihood_variance', 'get_elbo', 'export_model', 'import_model']


def get_likelihood_variance(m: object) -> gpflow.Parameter:
    if isinstance(m, gpflow.models.GPModel):
        return m.likelihood.variance
    elif isinstance(m, gpflux.models.DeepGP):
        return m.likelihood_layer.likelihood.variance
    elif isinstance(m, tf.keras.Model):
        assert isinstance(m.layers[-1], gpflux.layers.LikelihoodLayer), 'LikelihoodLayer has to have the last one'
        return m.layers[-1].likelihood.variance
    else:
        raise NotImplementedError(f'Unknown model type {type(m)}')


def get_elbo(m: Union[gpflow.models.GPModel, gpflux.models.DeepGP], train_data=None) -> tf.Tensor:
    if isinstance(m, gpflow.models.GPModel):
        return m.maximum_log_likelihood_objective()
    else:
        assert train_data is not None
        return m.elbo(train_data)


def export_model(f: str, m: Union[gpflow.models.GPModel, gpflux.models.DeepGP, tf.keras.Model]):
    if isinstance(m, gpflux.models.DeepGP):
        m.as_prediction_model().save_weights(f + '.tf')
    elif isinstance(m, gpflow.models.GPModel):
        np.savez(f + '.npz', **{k: v.numpy() for k, v in gpflow.utilities.parameter_dict(m).items()})
    elif isinstance(m, tf.keras.Model):
        m.save_weights(f + '.tf')
    else:
        raise NotImplementedError(f'Type {type(m)!r} not supported')


def import_model(f: str, m: Union[gpflow.models.GPModel, gpflux.models.DeepGP, tf.keras.Model]):
    if isinstance(m, gpflux.models.DeepGP):
        m.as_prediction_model().load_weights(f + '.tf')
    elif isinstance(m, gpflow.models.GPModel):
        gpflow.utilities.multiple_assign(m, np.load(f + '.npz', allow_pickle=True))
    elif isinstance(m, tf.keras.Model):
        m.load_weights(f + '.tf')
    else:
        raise NotImplementedError(f'Type {type(m)!r} not supported')
