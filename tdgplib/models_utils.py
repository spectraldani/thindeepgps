import logging
from typing import Union, Optional

import gpflow
import gpflux
import numpy as np
import tensorflow as tf
from gpflow.base import RegressionData

__all__ = ['get_likelihood_variance', 'train_model', 'export_model', 'import_model']


def get_likelihood_variance(m: object) -> gpflow.Parameter:
    if isinstance(m, gpflow.models.GPModel):
        return m.likelihood.variance
    elif isinstance(m, gpflux.models.DeepGP):
        return m.likelihood_layer.likelihood.variance
    elif isinstance(m, tf.keras.Model):
        assert isinstance(m.layers[-1], gpflux.layers.LikelihoodLayer), 'LikelihoodLayer has to have the last one'
        return m.layers[-1].likelihood.variance
    else:
        raise NotImplemented(f'Unknown model type {type(m)}')


def run_iterative_opt(opt, loss, vars, steps, monitor, first_step=0):
    for i in tf.range(steps):
        opt.minimize(loss, vars)
        monitor(first_step + i)


class CallbackFromMonitor(tf.keras.callbacks.Callback):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        self.monitor(epoch)


TF_LOGGER = logging.getLogger("tensorflow")

_default_schedule = (
    # (iterations , variance, step size)
    (500, 0.01, 0.1),
    (1_500, 0.01, 0.01),
    (5_000, True, 0.001)
)


def train_model(
        m: Union[gpflow.models.GPModel, gpflux.models.DeepGP, tf.keras.Model],
        monitor,
        train_data: Optional[RegressionData] = None,
        schedule=_default_schedule
):
    if isinstance(m, gpflux.models.DeepGP):
        m = m.as_training_model()
    if isinstance(m, gpflow.models.GPModel):
        def train(m, steps, step_size, initial=0):
            opt = tf.optimizers.Adam(step_size)
            return tf.function(run_iterative_opt)(
                opt, m.training_loss, m.trainable_variables, steps, monitor, initial
            )
    elif isinstance(m, tf.keras.Model):
        assert train_data is not None

        def train(m, steps, step_size, initial=0):
            m.compile(tf.optimizers.Adam(step_size))
            return m.fit(
                {"inputs": train_data[0], "targets": train_data[1]},
                batch_size=len(train_data[0]),
                epochs=initial + steps, initial_epoch=initial,
                verbose=0, callbacks=[CallbackFromMonitor(monitor)]
            )
    else:
        raise NotImplementedError(f'Unknown type {type(m)!r}')

    variance_parameter = get_likelihood_variance(m)

    print('Training', m)
    last_initial = 0
    for i, (steps, train_variance, step_size) in enumerate(schedule):
        print(f'#{i} round. {steps} - variance? {train_variance} step {step_size:.3e}')
        if isinstance(train_variance, bool):
            gpflow.set_trainable(variance_parameter, train_variance)
        else:
            gpflow.set_trainable(variance_parameter, False)
            variance_parameter.assign(train_variance)
        if steps > 0:
            curr_level = TF_LOGGER.getEffectiveLevel()
            TF_LOGGER.setLevel(logging.ERROR)
            train(m, tf.convert_to_tensor(steps, dtype=tf.int64), step_size, last_initial)
            TF_LOGGER.setLevel(curr_level)
        last_initial = steps


def export_model(f: str, m: Union[gpflow.models.GPModel, gpflux.models.DeepGP, tf.keras.Model]):
    if isinstance(m, gpflux.models.DeepGP):
        m.as_prediction_model().save_weights(f + '.tf')
    elif isinstance(m, gpflow.models.GPModel):
        np.savez(f + '.npz', **{k: v.numpy() for k, v in gpflow.utilities.parameter_dict(m).items()})
    elif isinstance(m, tf.keras.Model):
        m.save_weights(f + '.tf')
    else:
        raise NotImplemented(f'Type {type(m)!r} not supported')


def import_model(f: str, m: Union[gpflow.models.GPModel, gpflux.models.DeepGP, tf.keras.Model]):
    if isinstance(m, gpflux.models.DeepGP):
        m.as_prediction_model().load_weights(f + '.tf')
    elif isinstance(m, gpflow.models.GPModel):
        gpflow.utilities.multiple_assign(m, np.load(f + '.npz', allow_pickle=True))
    elif isinstance(m, tf.keras.Model):
        m.load_weights(f + '.tf')
    else:
        raise NotImplemented(f'Type {type(m)!r} not supported')
