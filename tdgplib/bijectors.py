import tensorflow as tf
import tensorflow_probability as tfp


class FillDiagonal(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name='fill_diag'):
        with tf.name_scope(name) as name:
            super(FillDiagonal, self).__init__(
                forward_min_event_ndims=1,
                inverse_min_event_ndims=2,
                is_constant_jacobian=True,
                validate_args=validate_args,
                name=name)

    def _forward(self, x):
        return tf.linalg.diag(x)

    def _inverse(self, y):
        return tf.linalg.diag_part(y)

    @staticmethod
    def _forward_log_det_jacobian(x):
        return tf.zeros([], dtype=x.dtype)

    @staticmethod
    def _inverse_log_det_jacobian(y):
        return tf.zeros([], dtype=y.dtype)

    @property
    def _is_permutation(self):
        return True


class FillMasked(tfp.bijectors.Bijector):
    def __init__(self, mask, validate_args=False, name='fill_mask'):
        with tf.name_scope(name) as name:
            super(FillMasked, self).__init__(
                forward_min_event_ndims=1,
                inverse_min_event_ndims=len(mask.shape),
                is_constant_jacobian=True,
                validate_args=validate_args,
                name=name)
        self.mask = mask
        self.required_x_shape = tf.reduce_sum(tf.cast(mask, tf.int64))

    def _forward(self, x):
        assert x.shape == self.required_x_shape, f'Shaped mismatched: expected {self.required_x_shape.numpy()}, got {x.shape}'
        return tf.scatter_nd(tf.where(self.mask), x, self.mask.shape)

    def _forward_event_shape_tensor(self, x_shape):
        assert x_shape == self.required_x_shape, f'Shaped mismatched: expected {self.required_x_shape.numpy()}, got {x_shape}'
        return self.mask.shape

    def _inverse(self, y):
        assert y.shape == self.mask.shape, f'Shaped mismatched: expected {self.mask.shape}, got {y.shape}'
        return tf.boolean_mask(y, self.mask)

    def _inverse_event_shape_tensor(self, y_shape):
        assert y_shape == self.mask.shape, f'Shaped mismatched: expected {self.mask.shape}, got {y_shape}'
        return self.required_x_shape

    @staticmethod
    def _forward_log_det_jacobian(x):
        return tf.zeros([], dtype=x.dtype)

    @staticmethod
    def _inverse_log_det_jacobian(y):
        return tf.zeros([], dtype=y.dtype)

    @property
    def _is_permutation(self):
        return True
