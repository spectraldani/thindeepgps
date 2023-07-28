import numpy as np
import tensorflow as tf


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
