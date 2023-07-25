from typing import List, Optional, Tuple

import gpflow
import gpflux
import numpy as np
import tensorflow as tf
from gpflow.base import InputData, RegressionData, MeanAndVariance, TensorType
from gpflow.models.util import InducingVariablesLike
from sklearn.decomposition import PCA

from ..helper import jitter_matrix, cholesky_logdet, initial_inducing_points, batched_identity
from ..parametrized_gaussian import MeanFieldGaussian

__all__ = ['TDGP', 'TDGPLayer']

# noinspection PyTypeChecker
float_type: np.dtype = gpflow.config.default_float()


class TDGP(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(
            self,
            data: RegressionData,
            inducing_f: InducingVariablesLike,
            inducing_w: InducingVariablesLike,
            w_kerns: List[gpflow.kernels.Kernel],
            full_qcov=False,
            diag_qmu=False
    ):
        # noinspection PyTypeChecker
        super().__init__(None, gpflow.likelihoods.Gaussian(), None, data[1].shape[1])
        self.data = gpflow.models.util.data_input_to_tensor(data)
        X, Y = data
        self.full_n = X.shape[0]

        if type(inducing_f) is not tuple:
            self.Z_u = gpflow.models.util.inducingpoint_wrapper(inducing_f)
            m_u, Q = self.Z_u.Z.shape
        else:
            m_u, Q = inducing_f
        if type(inducing_w) is not tuple:
            self.Z_v = gpflow.models.util.inducingpoint_wrapper(inducing_w)
            m_v, D = self.Z_v.Z.shape
        else:
            m_v, D = inducing_w

        n = X.shape[0]
        Dy = Y.shape[1]
        assert D == X.shape[1]
        assert m_v <= n and m_u <= n

        self.s_kern = gpflow.kernels.RBF()
        gpflow.set_trainable(self.s_kern, False)

        # Q kernels
        self.w_kerns = w_kerns

        # Parameters
        self.likelihood.variance.assign(0.01 * np.var(Y))
        self.sigma_f = gpflow.base.Parameter(
            np.std(Y), dtype=float_type, transform=gpflow.utilities.positive()
        )

        # q(V) distribution
        # mean: Q * D * m_v
        # cov:  Q * m_v * m_v
        self.qV = MeanFieldGaussian(
            m_v,
            (Q, D),
            # cov(V[:,d1,:]) == cov(V[:,d2,:]) for every d1 and d2
            shared_cov={1},
            full_cov=full_qcov,
        )

        if diag_qmu:
            # Q * D * m_v
            self.qV.mean.assign(np.repeat(np.eye(D)[:, :, None], m_v, axis=2))
        else:
            # noinspection PyUnresolvedReferences
            pca_components = PCA(Q).fit(X).components_.astype(float_type)
            # X_proj = X @ pca_components.T
            # input_scales = np.sqrt((2 / np.ptp(X, axis=0) ** 2).reshape(1, -1))
            # Q * D * m_v
            self.qV.mean.assign(np.repeat(pca_components[:, :, None], m_v, axis=2))

        cov_factor = 1 / D + (0.001 / D)
        batch_eye = batched_identity(m_v, (Q,))
        # FIXME
        # Q * 1 * m_v * m_v
        self.qV.chol_cov.assign((cov_factor * batch_eye)[:, None, :])

        self.qu = None

        if type(inducing_f) is tuple:
            self.Z_u = gpflow.models.util.inducingpoint_wrapper(
                initial_inducing_points((X @ self.qV.mean[:, :, 0].numpy().T), m_u)
            )
        if type(inducing_w) is tuple:
            self.Z_v = gpflow.models.util.inducingpoint_wrapper(
                initial_inducing_points(X, m_v)
            )

    # m_u * m_u
    def compute_Ku(self):
        return tf.identity(
            gpflow.covariances.dispatch.Kuu(self.Z_u, self.s_kern, jitter=0), name="Ku"
        )

    def compute_Ku_star(self, Z_star):
        return gpflow.covariances.dispatch.Kuf(self.Z_u, self.s_kern, Z_star)

    # Q * m_v * m_v
    def compute_Kv(self):
        return tf.stack(
            [gpflow.covariances.dispatch.Kuu(self.Z_v, k, jitter=0) for k in self.w_kerns]
        )

    # Q * n * n
    def compute_Kw(self, X=None, diag=False):
        if X is None:
            X = self.data[0]
        if not diag:
            return tf.stack([k.K(X) for k in self.w_kerns], name="Kw")
        else:
            return tf.stack([k.K_diag(X) for k in self.w_kerns], name="Kw")

    # Q * m_v * n
    def compute_Kvw(self, X=None):
        if X is None:
            X = self.data[0]
        return tf.stack(
            [gpflow.covariances.dispatch.Kuf(self.Z_v, k, X) for k in self.w_kerns], name="Kvw"
        )

    #
    def compute_optimal_qu(self) -> MeanAndVariance:
        """
        :return: q(u) ~ Normal with shape (m_u,Dy)
        """
        m_u = tf.shape(self.Z_u.Z)[0]
        sigma2 = tf.identity(self.likelihood.variance, name="sigma2")
        Psi1 = self.compute_Psi1()
        Psi2 = self.compute_Psi2()

        Ku = self.compute_Ku()
        Ku_Psi2 = sigma2 * Ku + Psi2
        chol_Ku_Psi2 = tf.linalg.cholesky(Ku_Psi2 + jitter_matrix(m_u), name="chol_Ku_Psi2")

        # m_u * Dy
        Psi1_Y = tf.matmul(Psi1, self.data[1], transpose_a=True)

        mean = Ku @ tf.linalg.cholesky_solve(chol_Ku_Psi2, Psi1_Y)
        cov = sigma2 * Ku @ tf.linalg.cholesky_solve(chol_Ku_Psi2, Ku)

        return mean, cov

    def compute_qs(self, Z=None) -> MeanAndVariance:
        """
        :return: q(s) ~ Normal with shape (n,Dy)
        """
        if Z is None:
            Z = self.Z_u.Z
        m_u = tf.shape(self.Z_u.Z)[0]
        Q = tf.shape(self.Z_u.Z)[1]

        # m_u * Dy, m_u * m_u
        qu_mean, qu_cov = self.compute_optimal_qu()

        Ku = self.compute_Ku()
        chol_Ku = tf.linalg.cholesky(Ku + jitter_matrix(m_u), name="chol_Ku")

        # m_u * n
        Ku_star = gpflow.covariances.dispatch.Kuf(self.Z_u, self.s_kern, Z)

        # m_u * n
        alpha = tf.linalg.cholesky_solve(chol_Ku, Ku_star)

        # n * Dy
        mean = tf.matmul(alpha, qu_mean, transpose_a=True)

        Kstar = self.s_kern.K(Z)
        cov = tf.matmul(alpha, (Ku - qu_cov) @ alpha, transpose_a=True)
        cov = Kstar - cov

        return mean, cov

    def compute_qf(self, W, X: Optional[tf.Tensor] = None, full_cov=True) -> MeanAndVariance:
        """
        :arg W:
        :arg X:
        :arg full_cov:
        :return: q(f|W) ~ Normal with shape (n,Dy)
        """
        # n * D
        if X is None:
            X = self.data[0]

        # n * Q * D
        # W

        m_u = tf.shape(self.Z_u.Z)[0]

        # n * Q
        Wx = tf.matmul(
            # n * 1 * D
            X[:, None, :],
            # n * Q * D
            W,
            transpose_b=True,
        )[:, 0, :]

        # m: m_u * Dy
        # S: m_u * m_u
        m, S = self.compute_optimal_qu()

        # m_u * m_u
        Ku = self.compute_Ku()
        chol_Ku = tf.linalg.cholesky(Ku + jitter_matrix(m_u), name="chol_Kv")

        # m_u * n
        Kuf = self.sigma_f * self.compute_Ku_star(Wx)

        # m_u * n
        alpha = tf.linalg.cholesky_solve(chol_Ku, Kuf)

        # n * Dy
        mean = tf.matmul(alpha, m, transpose_a=True)

        if full_cov:
            Kf = tf.pow(self.sigma_f, 2) * self.s_kern.K(Wx)
            # n * n
            cov = Kf - tf.matmul(alpha, (Ku - S) @ alpha, transpose_a=True)
        else:
            # n
            Kf = tf.pow(self.sigma_f, 2) * self.s_kern.K_diag(Wx)
            # n * m_u * 1
            alpha = tf.transpose(alpha[:, :, None], [1, 0, 2])
            # n * 1 * 1
            cov = tf.matmul(alpha, (Ku - S) @ alpha, transpose_a=True)
            # n
            cov = Kf - cov[:, 0, 0]

        return mean, cov

    def compute_qW(self, X: Optional[tf.Tensor] = None, full_cov=True) -> MeanAndVariance:
        """
        :arg X:
        :arg full_cov:
        :return: q(W) ~ Normal with shape (Q,D,n,1)
        """
        if X is None:
            X = self.data[0]
        m_v = tf.shape(self.Z_v.Z)[0]
        D = tf.shape(X)[1]
        Q = tf.shape(self.Z_u.Z)[1]

        Kv = self.compute_Kv()
        Kvw = self.compute_Kvw(X)

        chol_Kv = tf.linalg.cholesky(Kv + jitter_matrix(m_v), name="chol_Kv")

        # FIXME
        # Q * m_v * m_v
        q_cov = self.qV.cov[:, 0, :, :]

        # Q * m_v * n
        alpha = tf.linalg.cholesky_solve(chol_Kv, Kvw)

        # Q * D * n * 1
        a_tilde = tf.matmul(
            # Q * 1 * m_v * n
            alpha[:, None, :, :],
            # Q * D * m_v * 1
            self.qV.mean[:, :, :, None],
            transpose_a=True,
            name="a_tilde",
        )

        if full_cov:
            # Q * n * n
            Kw = self.compute_Kw(X, diag=False)

            # Q * n * n
            B_tilde = Kw - tf.matmul(alpha, (Kv - q_cov) @ alpha, transpose_a=True)
            # Q * 1 * n * n
            B_tilde = B_tilde[:, None]
        else:
            # Q * n
            Kw = self.compute_Kw(X, diag=True)

            # Q * n * m_v * 1
            alpha = tf.transpose(alpha[:, :, :, None], [0, 2, 1, 3])
            # Q * n * 1 * 1
            B_tilde = tf.matmul(alpha, (Kv - q_cov)[:, None] @ alpha, transpose_a=True)

            # Q * 1 * n
            B_tilde = (Kw - B_tilde[:, :, 0, 0])[:, None, :]

        return a_tilde, B_tilde

    def compute_psi0(self, X=None) -> tf.Tensor:
        """:return: psi_0: (1)"""
        if X is None:
            X = self.data[0]
        n = tf.shape(X)[0]
        return tf.multiply(tf.cast(n, float_type), tf.pow(self.sigma_f, 2), name="psi0")

    def compute_Psi1(self, X=None) -> tf.Tensor:
        """:return: Psi_1: (n,m)"""
        if X is None:
            X = self.data[0]

        # ã  Q * D * n * 1
        # B~ Q * 1 * n
        a_tilde, B_tilde = self.compute_qW(X, full_cov=False)

        # Q * n
        X_a_tilde = tf.matmul(
            # 1 * n * 1 * D
            X[None, :, None, :],
            # Q * n * D * 1
            tf.transpose(a_tilde, [0, 2, 1, 3]),
        )[:, :, 0, 0]

        # Q * n * m_u
        top = tf.subtract(
            # Q * n * 1
            X_a_tilde[:, :, None],
            # Q * 1 * m_u
            tf.transpose(self.Z_u.Z)[:, None, :],
        )
        top = tf.pow(top, 2)

        # Q * n * 1
        bot = (
                  # Q * n
                      B_tilde[:, 0, :]
                      # 1 * n
                      * tf.reduce_sum(tf.pow(X, 2), axis=1)[None, :]
                      + 1
              )[:, :, None]

        # Q * n * m_u
        Psi1 = tf.exp(-0.5 * top / bot) / tf.sqrt(bot)
        return tf.multiply(self.sigma_f, tf.reduce_prod(Psi1, axis=0), name="Psi1")

    def compute_batched_Psi2(self, X=None) -> tf.Tensor:
        """:return: Psi_2: (n,m,m)"""
        if X is None:
            X = self.data[0]
        n = tf.shape(X)[0]
        D = tf.shape(X)[1]
        m_u = tf.shape(self.Z_u.Z)[0]
        Q = tf.shape(self.Z_u.Z)[1]

        # ã  Q * D * n * 1
        # B~ Q * 1 * n
        a_tilde, B_tilde = self.compute_qW(X, full_cov=False)

        # Q * n * 1 * 1
        a_tilde_X = tf.matmul(
            # 1 * n * 1 * D
            X[None, :, None, :],
            # Q * n * D * 1
            tf.transpose(a_tilde, [0, 2, 1, 3]),
        )

        # Q * 1 * m_u * m_u
        Z_bar = tf.transpose(
            0.5 * (self.Z_u.Z[:, None, :] + self.Z_u.Z[None, :, :]), [2, 0, 1]
        )[:, None, :, :]

        # Q * D * n * m_u * m_u
        top = tf.subtract(
            # Q * n * 1 * 1
            a_tilde_X,
            # Q * 1 * m_u * m_u
            Z_bar,
        )
        top = tf.pow(top, 2)

        # Q * n
        bot = 2 * B_tilde[:, 0, :] * tf.reduce_sum(X * X, axis=1)[None, :] + 1

        # Q * n * 1 * 1
        bot = bot[:, :, None, None]

        # Q * D * n * m_u * m_u
        Psi2 = tf.exp(-top / bot) / tf.sqrt(bot)

        # n * m_u * m_u
        Psi2 = tf.reduce_prod(Psi2, axis=0)

        # m_u * m_u * Q
        Z_diff = tf.pow(self.Z_u.Z[:, None, :] - self.Z_u.Z[None, :, :], 2)

        # 1 * m_u * m_u
        Z_diff = tf.exp(-tf.reduce_sum(Z_diff, axis=2) / 4)[None, :, :]

        return tf.multiply(tf.pow(self.sigma_f, 2), Z_diff * Psi2, name="Psi2")

    def compute_Psi2(self, X=None) -> tf.Tensor:
        """:return: sum[Psi_2, 0]: (n,m)"""
        return tf.reduce_sum(self.compute_batched_Psi2(X), axis=0, name="Psi2")

    def compute_KL_qV(self):
        m_v = tf.shape(self.Z_v.Z)[0]
        D = tf.shape(self.data[0])[1]
        Q = tf.shape(self.Z_u.Z)[1]

        # Q * m_v * m_v
        Kv = self.compute_Kv()
        chol_Kv = tf.linalg.cholesky(Kv + jitter_matrix(m_v), name="chol_Kv")

        # FIXME
        # Q * m_v * m_v
        q_cov = self.qV.cov[:, 0, :, :]

        # Q * m_v * m_v
        Kvi_q_cov = tf.linalg.cholesky_solve(chol_Kv, q_cov)

        KL = tf.cast(D, float_type) * tf.reduce_sum(
            cholesky_logdet(chol_Kv)
            - cholesky_logdet(self.qV.chol_cov[:, 0, :, :])
            + tf.linalg.trace(Kvi_q_cov)
        )

        # Q * D * m_v * 1
        batched_q_mu = self.qV.mean[:, :, :, None]

        # Q * D * m_v * m_v
        batched_chol_Kv = tf.tile(chol_Kv[:, None, :, :], [1, D, 1, 1])

        # Q * D * m_v * 1
        KvMu = tf.linalg.cholesky_solve(batched_chol_Kv, batched_q_mu, name="KvMu")
        # Q * D
        MuKvMu = tf.matmul(batched_q_mu, KvMu, transpose_a=True)[:, :, 0, 0]

        KL += tf.reduce_sum(MuKvMu, axis=[0, 1])

        KL += -tf.cast(m_v * Q * D, float_type)
        return 0.5 * KL

    def compute_KL_qu(self):
        m_u = tf.shape(self.Z_u.Z)[0]
        Dy = tf.cast(tf.shape(self.data[1])[1], float_type)

        # m_u * m_u
        Ku = self.compute_Ku()
        chol_Ku = tf.linalg.cholesky(Ku + jitter_matrix(m_u), name="chol_Ku")

        # m: m_u * Dy
        # S: m_u * m_u
        m, S = self.compute_optimal_qu()

        if self.qu is not None:
            chol_S = self.qu.chol_cov
        else:
            chol_S = tf.linalg.cholesky(S + jitter_matrix(m_u), name="chol_S")

        KuiS = tf.linalg.cholesky_solve(chol_Ku, S)

        # m_u * Dy
        Kum = tf.linalg.cholesky_solve(chol_Ku, m)
        # ()
        mahal = tf.reduce_sum(m * Kum)

        KL = (
                mahal
                + Dy * cholesky_logdet(chol_Ku)
                - Dy * cholesky_logdet(chol_S)
                + Dy * tf.linalg.trace(KuiS)
                - Dy * tf.cast(m_u, float_type)
        )
        return 0.5 * KL

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        n = tf.shape(self.data[0])[0]
        m_v = tf.shape(self.Z_v.Z)[0]
        D = tf.shape(self.data[0])[1]
        Dy = tf.cast(tf.shape(self.data[1])[1], float_type)
        m_u = tf.shape(self.Z_u.Z)[0]
        Q = tf.shape(self.Z_u.Z)[1]
        sigma2 = tf.identity(self.likelihood.variance, name="sigma2")
        # m_u * m_u
        Ku = self.compute_Ku()
        chol_Ku = tf.linalg.cholesky(Ku + jitter_matrix(m_u), name="chol_Ku")
        # (,)
        YY = tf.reduce_sum(self.data[1] ** 2)
        # (,)
        psi0 = self.compute_psi0()
        # m_u * m_u
        Psi2 = self.compute_Psi2()
        # m_u * m_u
        KuiPsi2 = tf.linalg.cholesky_solve(chol_Ku, Psi2)

        # p(y|f) - <U>_q(u)
        F1 = -YY / sigma2
        F1 -= Dy * psi0 / sigma2
        F1 += Dy * tf.linalg.trace(KuiPsi2) / sigma2
        F1 -= Dy * tf.cast(n, float_type) * tf.math.log(2 * tf.cast(np.pi, float_type))
        F1 -= Dy * tf.cast(n - m_u, float_type) * tf.math.log(sigma2)

        # n * m_u
        Psi1 = self.compute_Psi1()
        # m_u * Dy
        Psi1_Y = tf.matmul(Psi1, self.data[1], transpose_a=True)
        # m_u * m_u
        Ku_Psi2 = sigma2 * Ku + Psi2
        # m_u * m_u
        chol_Ku_Psi2 = tf.linalg.cholesky(Ku_Psi2 + jitter_matrix(m_u), name="chol_Ku_Psi2")

        # KL(q(u)||p(u)) + <U>_q(u)
        F1 += Dy * cholesky_logdet(chol_Ku)
        F1 -= Dy * cholesky_logdet(chol_Ku_Psi2)
        F1 += tf.divide(
            tf.linalg.trace(
                tf.matmul(
                    Psi1_Y,
                    tf.linalg.cholesky_solve(chol_Ku_Psi2, Psi1_Y),
                    transpose_a=True,
                )
            ),
            sigma2,
        )

        KL = self.compute_KL_qV()
        return tf.squeeze((0.5 * F1) - KL)

    # q(F*) ~ Diagonal normal with shape (n*, Dy)

    def predict_f(self, X_new: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        assert not full_cov
        assert not full_output_cov
        m_u = tf.shape(self.Z_u.Z)[0]
        n_new = tf.shape(X_new)[0]
        sigma2 = tf.identity(self.likelihood.variance, name="sigma2")
        # m_u * m_u
        Ku = self.compute_Ku()
        # n * m_u
        Psi1 = self.compute_Psi1()
        # m_u * m_u
        Psi2 = self.compute_Psi2()
        Ku_Psi2 = sigma2 * Ku + Psi2
        chol_Ku_Psi2 = tf.linalg.cholesky(Ku_Psi2 + jitter_matrix(m_u))
        # m_u * Dy
        alpha = tf.linalg.cholesky_solve(
            chol_Ku_Psi2, tf.matmul(Psi1, self.data[1], transpose_a=True)
        )
        # n_new * m_u
        Psi1_star = self.compute_Psi1(X_new)
        # n_new * Dy
        mean = Psi1_star @ alpha
        chol_Ku = tf.linalg.cholesky(Ku + jitter_matrix(m_u))
        # n_new * m_u * m_u
        batched_chol_Ku = tf.tile(chol_Ku[None, ...], [n_new, 1, 1])
        # n_new * m_u * m_u
        batched_Psi2_star = self.compute_batched_Psi2(X_new)
        # Dy * 1 * m_u * m_u
        alpha_alpha = tf.matmul(
            tf.transpose(alpha)[:, :, None], tf.transpose(alpha)[:, None, :]
        )[:, None]
        # n_new * m_u * m_u
        batched_chol_Ku_Psi2 = tf.tile(chol_Ku_Psi2[None, ...], [n_new, 1, 1])
        # n_new * Dy
        trace_terms = tf.transpose(
            # 0
            sigma2
            # n_new
            * tf.linalg.trace(tf.linalg.cholesky_solve(batched_chol_Ku_Psi2, batched_Psi2_star))
            # n_new
            - tf.linalg.trace(tf.linalg.cholesky_solve(batched_chol_Ku, batched_Psi2_star))
            # Dy * n_new
            + tf.linalg.trace(tf.matmul(alpha_alpha, batched_Psi2_star))
        )
        variance = trace_terms - mean ** 2 + self.sigma_f ** 2
        result = mean, variance
        return result

    def sample_W(self, X=None, samples=200):
        if X is None:
            X = self.data[0]

        n = tf.shape(X)[0]
        D = tf.shape(X)[1]
        Q = tf.shape(self.Z_u.Z)[1]

        W_eps = tf.random.normal((samples, Q, D, n), dtype=float_type)

        m, S = self.compute_qW(X, full_cov=False)
        m = m[..., 0]

        # samples * Q * D * n
        W = m + W_eps * tf.sqrt(S)

        return W

    def sample_f(self, X=None, samples=200):
        if X is None:
            X = self.data[0]
        n = tf.shape(X)[0]
        Dy = tf.shape(self.data[1])[1]

        # samples * Q * D * n
        W_samples = self.sample_W(X, samples)

        # samples * n * Q * D
        W_samples = tf.transpose(W_samples, [0, 3, 1, 2])

        f_eps = tf.random.normal((samples, n, Dy), dtype=float_type)

        # samples * n * Dy, samples * n
        m, S = tf.map_fn(
            lambda W: self.compute_qf(W, X, full_cov=False),
            W_samples,
            dtype=(float_type, float_type),
        )

        f = m + f_eps * tf.sqrt(S[..., None])

        return f

    def sample_y(self, X=None, samples=200):
        if X is None:
            X = self.data[0]
        n = tf.shape(X)[0]
        Dy = tf.shape(self.data[1])[1]

        f = self.sample_f(X, samples)
        eps = tf.random.normal((samples, n, Dy), dtype=float_type)

        y = f + eps * tf.sqrt(self.likelihood.variance)

        return y

    def extract_parameters(self):
        return {
            'likelihood.variance': self.likelihood.variance.numpy(),
            'w_kerns': [
                {
                    'variance': k.variance.numpy(),
                    'lengthscales': k.lengthscales.numpy()
                }
                for k in self.w_kerns
            ],
            'sigma_f': self.sigma_f.numpy(),
            'qV.mean': self.qV.mean.numpy(),
            'qV.chol_cov': self.qV.chol_cov.numpy(),
            'Z_u.Z': self.Z_u.Z.numpy(),
            'Z_v.Z': self.Z_v.Z.numpy(),
        }


class TDGPLayer(gpflux.layers.GPLayer):
    def __init__(
            self,
            *args,
            diag_qmu=True,
            input_dim=-1,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.diag_qmu = diag_qmu
        self.latent_dim = self.inducing_variable.shape[1]
        self.input_dim = input_dim
        self.reshaper = tf.keras.layers.Reshape(
            (self.latent_dim, input_dim),
            input_shape=(self.latent_dim * input_dim,),
            dtype=float_type
        )

    def predict(
            self,
            concat_inputs: TensorType,
            *,
            full_cov: bool = False,
            full_output_cov: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        inputs = self.compute_projection(concat_inputs)
        return super().predict(inputs, full_cov=full_cov, full_output_cov=full_output_cov)

    def _matrix_shape(self, input_shape):
        assert sum(x is None for x in input_shape) <= 1, 'Only one None is supported'
        output_shape = tuple(
            -1 if x is None else x
            for x in input_shape[:-1]
        )
        return output_shape + (self.latent_dim, self.input_dim)

    def compute_projection(self, concat_inputs):
        inputs = concat_inputs[..., :self.input_dim]
        inverse_lengthscale = concat_inputs[..., self.input_dim:]
        if self.diag_qmu:
            return inputs * inverse_lengthscale
        else:
            inverse_lengthscale_matrix = tf.reshape(
                inverse_lengthscale,
                self._matrix_shape(inverse_lengthscale.shape)
            )
            return tf.matmul(
                inputs[..., None, :],
                inverse_lengthscale_matrix,
                transpose_b=True
            )[..., 0, :]
