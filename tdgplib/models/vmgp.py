import gpflow
import numpy as np
import tensorflow as tf

from helper import jitter_matrix, cholesky_logdet

__all__ = ['VariationalMahalanobisGP']

# noinspection PyTypeChecker
float_type: np.dtype = gpflow.config.default_float()


# noinspection PyPep8Naming
class VariationalMahalanobisGP(gpflow.models.GPModel, gpflow.models.InternalDataTrainingLossMixin):
    def __init__(
            self,
            data: gpflow.base.RegressionData,
            inducing_variable: gpflow.models.util.InducingVariablesLike,
            noise_variance: float = 1.0,
    ):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        # noinspection PyTypeChecker
        super().__init__(None, likelihood, None, num_latent_gps=data[1].shape[-1])
        self.data = gpflow.models.util.data_input_to_tensor(data)
        self.num_data = self.data[0].shape[0]
        self.inducing_variable = gpflow.models.util.inducingpoint_wrapper(inducing_variable)

        self.std_rbf = gpflow.kernels.SquaredExponential(1, 1)
        gpflow.set_trainable(self.std_rbf, False)

        D = self.data[0].shape[1]
        K = self.inducing_variable.Z.shape[1]
        # Parameters
        self.sigma_f = gpflow.base.Parameter(
            1., dtype=float_type, transform=gpflow.utilities.positive()
        )
        self.q_mu = gpflow.base.Parameter(tf.zeros((K, D)), dtype=float_type)
        self.q_cov = gpflow.base.Parameter(
            (1 / D + (0.001 / D) * np.random.randn(K, D)),
            dtype=float_type,
            transform=gpflow.utilities.positive(),
        )

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    # 1
    def compute_psi0(self):
        n = tf.shape(self.data[0])[0]
        return tf.multiply(tf.cast(n, float_type), self.sigma_f ** 2, name="psi0")

    # n * m
    def compute_Psi1(self, X=None):
        if X is None:
            X = self.data[0]

        # K * n
        q_mu_X = tf.tensordot(self.q_mu, X, [[1], [1]], name="q_mu_x")

        with tf.name_scope("Psi1"):
            # K * n * m
            top = tf.subtract(
                q_mu_X[:, :, None], tf.transpose(self.inducing_variable.Z, [1, 0])[:, None, :]
            )
            top = tf.pow(top, 2, name="top")

            # K * n
            bot = 1 + tf.reduce_sum(self.q_cov[:, None, :] * tf.pow(X[None], 2), axis=2)
            # K * n * 1
            bot = tf.identity(bot[:, :, None], name="bot")

            # K * n * m
            Psi1 = tf.math.exp(-0.5 * top / bot) / tf.sqrt(bot)

        return tf.multiply(self.sigma_f, tf.reduce_prod(Psi1, axis=0), name="Psi1")

    # n * m * m
    def compute_batched_Psi2(self, X=None):
        if X is None:
            X = self.data[0]

        # K * n
        q_mu_X = tf.tensordot(self.q_mu, X, [[1], [1]], name="q_mu_x")

        with tf.name_scope("Psi2"):
            # m * m * K
            Z_diff = tf.subtract(
                self.inducing_variable.Z[:, None, :], self.inducing_variable.Z[None, :, :], name="Z_diff"
            )

            # m * m * K
            Z_bar = 0.5 * (self.inducing_variable.Z[:, None, :] + self.inducing_variable.Z[None, :, :])

            # K * n * m * m
            top = tf.subtract(
                q_mu_X[:, :, None, None], tf.transpose(Z_bar, [2, 0, 1])[:, None, :, :]
            )
            top = tf.pow(top, 2)

            # K * n
            bot = (
                    2 * tf.reduce_sum(self.q_cov[:, None, :] * tf.pow(X[None], 2), axis=2)
                    + 1
            )

            # K * n * 1 * 1
            bot = tf.identity(bot[:, :, None, None], name="bot")

            # K * n * m * m
            right = tf.math.exp(-top / bot) / tf.sqrt(bot)
            # n * m * m
            right = tf.reduce_prod(right, axis=0, name="right")

            # m * m
            left = tf.multiply(
                tf.pow(self.sigma_f, 2),
                tf.math.exp(-0.25 * tf.reduce_sum(tf.pow(Z_diff, 2), axis=2)),
                name="left",
            )
        return tf.multiply(left, right, name="BatchedPsi2")

    def compute_Psi2(self, X=None):
        return tf.reduce_sum(self.compute_batched_Psi2(X), axis=0, name="Psi2")

    def elbo(self):
        X, y = self.data
        n = tf.shape(X)[0]
        D = tf.shape(X)[1]
        m = tf.shape(self.inducing_variable.Z)[0]
        sigma2 = self.likelihood.variance

        # m * m
        Ku = gpflow.covariances.Kuu(self.inducing_variable, self.std_rbf, jitter=gpflow.config.default_jitter())
        chol_Ku = tf.linalg.cholesky(Ku)

        F1 = (
                -tf.cast(n, float_type) * tf.math.log(2 * tf.cast(np.pi, float_type))
                - tf.cast(n - m, float_type) * tf.math.log(sigma2)
                + cholesky_logdet(chol_Ku)
        )

        # 1 * 1
        YY = tf.matmul(y, y, transpose_a=True)

        # m * m
        Psi2 = self.compute_Psi2()

        # m * m
        Ku_Psi2 = sigma2 * Ku + Psi2
        # m * m
        chol_Ku_Psi2 = tf.linalg.cholesky(Ku_Psi2 + jitter_matrix(m), name="chol_Ku_Psi2")

        F1 += -cholesky_logdet(chol_Ku_Psi2) - YY / sigma2

        # n * m
        Psi1 = self.compute_Psi1()

        # m * 1
        Psi1_Y = tf.matmul(Psi1, y, transpose_a=True)

        F1 += tf.divide(
            tf.matmul(
                Psi1_Y, tf.linalg.cholesky_solve(chol_Ku_Psi2, Psi1_Y), transpose_a=True
            ),
            sigma2,
        )

        # 1
        psi0 = self.compute_psi0()

        # m * m
        KuiPsi2 = tf.linalg.cholesky_solve(chol_Ku, Psi2)

        F1 += -psi0 / sigma2 + tf.linalg.trace(KuiPsi2) / sigma2

        KL = tf.reduce_sum(tf.math.log(self.q_cov), axis=1)
        KL -= tf.cast(D, float_type) * tf.math.log(
            tf.reduce_sum(self.q_cov + tf.pow(self.q_mu, 2), axis=1)
        )
        KL += tf.cast(D, float_type) * tf.math.log(tf.cast(D, float_type))
        KL = tf.reduce_sum(KL)

        return tf.squeeze(0.5 * (F1 + KL))

    def predict_f(
            self, Xnew: gpflow.base.InputData, full_cov: bool = False, full_output_cov: bool = False
    ):
        assert not full_cov and not full_output_cov, 'Not implemented'

        X, y = self.data
        nNew = tf.shape(Xnew)[0]
        m = tf.shape(self.inducing_variable.Z)[0]

        sigma2 = self.likelihood.variance
        Ku = gpflow.covariances.Kuu(self.inducing_variable, self.std_rbf, jitter=gpflow.config.default_jitter())
        chol_Ku = tf.linalg.cholesky(Ku, name="chol_Ku")

        Psi1 = self.compute_Psi1()
        Psi2 = self.compute_Psi2()

        Ku_Psi2 = sigma2 * Ku + Psi2
        chol_Ku_Psi2 = tf.linalg.cholesky(Ku_Psi2 + jitter_matrix(m), name="chol_Ku_Psi2")

        Psi1_Y = tf.matmul(Psi1, y, transpose_a=True)
        alpha = tf.linalg.cholesky_solve(chol_Ku_Psi2, Psi1_Y)

        Psi1new = self.compute_Psi1(Xnew)

        mean = Psi1new @ alpha

        Psi2new = self.compute_batched_Psi2(Xnew)
        var = tf.linalg.trace(
            sigma2
            * tf.linalg.cholesky_solve(tf.tile(chol_Ku_Psi2[None], [nNew, 1, 1]), Psi2new)
            - tf.linalg.cholesky_solve(tf.tile(chol_Ku[None], [nNew, 1, 1]), Psi2new)
            + tf.tile(tf.matmul(alpha, alpha, transpose_b=True)[None], [nNew, 1, 1])
            @ Psi2new
        )
        var = tf.reshape(var, (-1, 1)) + self.sigma_f ** 2 + sigma2 - mean ** 2

        return mean, var
