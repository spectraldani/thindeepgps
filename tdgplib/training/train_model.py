import logging
from typing import Union, Optional

import gpflow
import gpflux
import tensorflow as tf
from gpflow.base import RegressionData

from .run_optimizer import run_optimizer
from ..helper.model_utils import get_likelihood_variance

__all__ = ['train_model']

TF_LOGGER = logging.getLogger("tensorflow")

_default_schedule = (
    # (iterations , variance, step size)
    (500, 0.01, 0.1),
    (1_500, 0.01, 0.01),
    (5_000, True, 0.001)
)


def train_model(
        m: Union[gpflow.models.GPModel, gpflux.models.DeepGP, tf.keras.Model],
        monitor: Optional[gpflow.monitor.Monitor] = None,
        train_data: Optional[RegressionData] = None,
        schedule=_default_schedule
):
    variance_parameter = get_likelihood_variance(m)

    print('Start training')
    last_initial = 0
    for i, (steps, train_variance, step_size) in enumerate(schedule):
        print(f'#{i} round. {steps} - variance {train_variance} step {step_size:.3e}')
        opt = tf.optimizers.Adam(step_size)
        if isinstance(train_variance, bool):
            gpflow.set_trainable(variance_parameter, train_variance)
        else:
            gpflow.set_trainable(variance_parameter, False)
            variance_parameter.assign(train_variance)
        if steps > 0:
            steps = tf.convert_to_tensor(steps, dtype=tf.int64)
            curr_level = TF_LOGGER.getEffectiveLevel()
            TF_LOGGER.setLevel(logging.ERROR)
            run_optimizer(m, opt, steps, last_initial, monitor, train_data)
            TF_LOGGER.setLevel(curr_level)
        last_initial = steps
