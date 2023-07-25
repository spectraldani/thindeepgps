from typing import Optional

import gpflux
import tensorflow as tf
from gpflow.base import TensorType

from .tdgp import TDGPLayer
from .dnsgp import NonstationaryGPLayer

__all__ = ['DeepGP']


# Adapted from the original source code
# https://github.com/secondmind-labs/GPflux/blob/3eeec8afd77c6f05f3ebd1d6ea6c6ef2b9221b79/gpflux/models/deep_gp.py#L150-L180
class DeepGP(gpflux.models.DeepGP):
    # noinspection PyCallingNonCallable
    def _evaluate_deep_gp(
            self,
            inputs: TensorType,
            targets: Optional[TensorType],
            training: Optional[bool] = None,
            up_to_layer: int = -1
    ) -> tf.Tensor:
        """
        Evaluate ``f(x) = fₙ(⋯ (f₂(f₁(x))))`` on the *inputs* argument.

        Layers that inherit from :class:`~gpflux.layers.LayerWithObservations`
        are passed the additional keyword argument ``observations=[inputs,
        targets]`` if *targets* contains a value, or ``observations=None`` when
        *targets* is `None`.
        """
        features = inputs
        assert -len(self.f_layers) <= up_to_layer < len(self.f_layers)
        up_to_layer = up_to_layer % len(self.f_layers)

        # NOTE: we cannot rely on the `training` flag here, as the correct
        # symbolic graph needs to be constructed at "build" time (before either
        # fit() or predict() get called).
        if targets is not None:
            observations = [inputs, targets]
        else:
            # TODO would it be better to simply pass [inputs, None] in this case?
            observations = None

        for i, layer in enumerate(self.f_layers):
            if isinstance(layer, gpflux.layers.LayerWithObservations):
                features = layer(features, observations=observations, training=training)
            elif isinstance(layer, (NonstationaryGPLayer, TDGPLayer)):
                # If there is MC samples, then shape is (n_samples, n_data, n_dim)
                if len(features.shape) == 3:
                    inputs_and_features = tf.concat([
                        tf.repeat(inputs[None], features.shape[0], axis=0),
                        features
                    ], axis=-1)
                else:
                    inputs_and_features = tf.concat([inputs, features], axis=-1)
                features = layer(inputs_and_features, training=training)
            else:
                features = layer(features, training=training)
            if i == up_to_layer:
                return features
        return features
