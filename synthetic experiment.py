# %%
import gpflow
import gpflux.models
import numpy as np
import tensorflow as tf
import uq360.metrics

import tdgplib
import tdgplib.helper
import tdgplib.models

RANDOM_SEED = 42

# %%
dummy_monitor = gpflow.monitor.Monitor(gpflow.monitor.MonitorTaskGroup(
    gpflow.monitor.ExecuteCallback(lambda *args, **kwargs: None),
    period=10000
))


def train_models(models, x, y):
    for model_name, model in models.items():
        print(model_name)
        if isinstance(model, gpflux.models.DeepGP):
            model = model.as_training_model()
        tdgplib.training.train_model(model, dummy_monitor, train_data=(x.train, y.train))


def calc_metrics(y_test, y_pred_mean, y_pred_var):
    metrics = {'mrae': tdgplib.metrics.mean_relative_absolute_error(y_test, y_pred_mean).squeeze().item()}
    for k in ["rmse", "nll", "auucc_gain", "picp", "piw", "r2"]:
        try:
            metrics.update(uq360.metrics.regression_metrics.compute_regression_metrics(
                y_test,
                y_mean=y_pred_mean.numpy(),
                y_lower=(y_pred_mean - 2 * np.sqrt(y_pred_var)).numpy(),
                y_upper=(y_pred_mean + 2 * np.sqrt(y_pred_var)).numpy(),
                option=k
            ))
        except ValueError as e:
            print(f'Error computing metric {k!r}:', e)
            metrics[k] = np.nan
    return pd.Series(metrics, dtype=float)


# %%
def true_data(X):
    W = np.stack([
        np.sin(3.1414 * X[..., 0]) * X[..., 0] * 2,
        np.cos(3.1414 * X[..., 0]) * 2,
    ], axis=-1)[..., None]
    Z = (X[:, None] @ W)[..., 0, 0]
    y = (np.sinc(Z) - Z ** 2)[:, None]
    return W, y


# %%
def predict_y(m, x):
    likelihood_variance = tdgplib.helper.get_likelihood_variance(m)
    f_pred_mean, f_pred_var = m.predict_f(x)
    if len(f_pred_mean.shape) == 3:
        f_pred_mean = tf.reduce_mean(f_pred_mean, axis=0)
        f_pred_var = tf.reduce_mean(f_pred_var, axis=0)
    y_pred_var = f_pred_var + likelihood_variance
    return f_pred_mean, y_pred_var


def evaluate_models(models, x, y):
    return pd.DataFrame({
        k: calc_metrics(y.test, *predict_y(m, x.test))
        for k, m in models.items()
    })


# %%
tdgplib.helper.set_all_random_seeds(RANDOM_SEED)
dataset = list(tdgplib.data_utils.get_sinc(RANDOM_SEED, n_fold=1, num_data=350))

# %%
rng = np.random.default_rng(19960111)
num_data = 350
n_fold = 1

X = rng.random((num_data, 2)) * 2 - 1
W = np.stack([
    np.sin(3.1414 * X[..., 0]) * X[..., 0] * 2,
    np.cos(3.1414 * X[..., 0]) * 2,
], axis=-1)[..., None]
Z = (X[:, None] @ W)[..., 0, 0]
y = (np.sinc(Z) - Z ** 2)[:, None]
X, y = tdgplib.data_utils.get_folds(X, y, n_fold=1, random_seed=RANDOM_SEED)[0]

# %%
X, y, scaler = tdgplib.data_utils.scale_data(X, y)
n, D = X.train.shape
m_v = 25
m_u, Q, = 50, D
Z_v = (m_v, D)
Z_u = (m_u, Q)

# %%
models = {}
tdgplib.helper.set_all_random_seeds(RANDOM_SEED)
models["sgpr"] = gpflow.models.SGPR(
    (X.train, y.train), gpflow.kernels.RBF(lengthscales=np.ones(D)),
    inducing_variable=tdgplib.helper.initial_inducing_points(X.train, Z_u[0])
)
tdgplib.helper.set_all_random_seeds(RANDOM_SEED)
models["vmgp"] = tdgplib.models.TDGP(
    (X.train, y.train), Z_u, Z_v,
    [gpflow.kernels.RBF(lengthscales=np.ones(D)) for _ in range(Q)],
    full_qcov=True, diag_qmu=False
)

tdgplib.helper.set_all_random_seeds(RANDOM_SEED)
models['dkl'] = tdgplib.models.make_dkl(
    (X.train, y.train),
    inducing_output_layer=Z_u
)

tdgplib.helper.set_all_random_seeds(RANDOM_SEED)
models['dgp'] = tdgplib.models.make_compositional_dgp(
    (X.train, y.train),
    inducing_hidden_layer=Z_v,
    inducing_output_layer=Z_u,
    num_samples=50
)

tdgplib.helper.set_all_random_seeds(RANDOM_SEED)
models['dnsgp'] = tdgplib.models.make_dnsgp(
    (X.train, y.train),
    inducing_lengthscale_layer=Z_v,
    inducing_output_layer=Z_u,
    num_samples=50
)

# %%
train_models(models, X, y)

# %%
import pandas as pd

tdgplib.helper.set_all_random_seeds(RANDOM_SEED)
metric_samples = pd.concat({
    i: evaluate_models(models, X, y)
    for i in range(10)
}, axis=0).swaplevel().sort_index()
print(metric_samples.groupby(level=0).mean().round(3))

# %% [markdown]
# # Plotting

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import cplot
import math

# %%
y_preds = pd.concat({
    k: pd.DataFrame(tf.concat(predict_y(model, X.test), axis=-1).numpy(), columns=['mean', 'var'])
    for k, model in models.items()
}, axis=1)

errors = y.test - y_preds.loc[:, pd.IndexSlice[:, 'mean']]

# %% [markdown]
# ## True $y$ versus predicted $\hat{y}$

# %%
ncols = 4
nrows = math.ceil(len(models) / ncols)
f, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)
for ax in axs.flat:
    ax.set_visible(False)
for i, (k, m) in enumerate(models.items()):
    ax = axs.flat[i]
    ax.set_visible(True)
    mean = y_preds[(k, 'mean')]
    y_std = np.sqrt(tdgplib.helper.get_likelihood_variance(m).numpy())
    std = np.sqrt(y_preds[(k, 'var')])
    lims = np.array([
        min(y.test.min(), mean.min()),
        max(y.test.max(), mean.max()),
    ])
    # plt.plot(lims,lims,'k-',zorder=2)
    ax.fill_between(lims, lims - 2 * y_std, lims + 2 * y_std, color='k', zorder=0)
    ax.errorbar(y.test, mean, yerr=2 * y_std, fmt='C1.', capsize=1)
    ax.set_title(k)
f.savefig('true_vs_predicted.png', bbox_inches='tight')

# %% [markdown]
# ## Predicted $\hat{y}$ in a grid

# %%
grid_lims = np.array([
    [-2, 2], [-2, 2]
])
d = grid_lims.shape[0]
grid_size = 75
grid_input = np.stack(np.meshgrid(*[np.linspace(*grid_lims[i], grid_size) for i in range(d)], indexing='ij'), axis=-1)
grid_delta = (grid_lims[:, 1] - grid_lims[:, 0]) / (grid_size - 1)
grid_shape = grid_input.shape[:-1]
full_input = grid_input.reshape(-1, d)

full_W, full_y = true_data(
    scaler.x.inverse_transform(full_input)
)
full_y = scaler.y.transform(full_y)

# %%
grid_preds = pd.concat({
    k: pd.DataFrame(tf.concat(predict_y(model, full_input), axis=-1).numpy(), columns=['mean', 'var'])
    for k, model in models.items()
}, axis=1)
grid_preds['true', 'mean'] = full_y

# %%
full_y_levels = [-5.6, -4.8, -4., -3.2, -2.4, -1.6, -0.8, 0., 0.8, 1.6]


# %%
def nlpd(observed_value, predict_mean, predict_var):
    n = observed_value.shape[0]
    inner = np.log(predict_var) + (observed_value - predict_mean) ** 2 / predict_var
    return 0.5 * (np.log(2 * np.pi) + inner)


# %% [markdown]
# ## Figure - Function outputs and their NLPD

# %%
labels = dict(
    true='True function',
    sgpr='Shallow SGP',
    dkl='DKL',
    dgp='CDGP',
    dnsgp='DNSGP',
    vmgp='TDGP'
)
to_plot = [
    'true', 'sgpr', 'dkl',
    'dgp', 'dnsgp', 'vmgp'
]
ncols = len(to_plot)
nrows = 2
output_pmesh = {}
nlpd_pmesh = {}
f, axs = plt.subplots(nrows, ncols, figsize=(12, 3), sharex=False, sharey=False, tight_layout=False)
for ax in axs.flat:
    ax.set_visible(False)
for i, k in enumerate(to_plot):
    m = models.get(k)
    ax = axs[0, i]
    ax.set_visible(True)
    ax.set_title(labels[k])
    if isinstance(m, tdgplib.models.TDGP):
        Z = m.Z_v.Z
    elif isinstance(m, gpflux.models.DeepGP) and isinstance(m.f_layers[0], gpflux.layers.GPLayer):
        Z = m.f_layers[0].inducing_variable.inducing_variables[0].Z
    else:
        Z = None
    clim = [min(y.test.min(), y.train.min()), max(y.test.max(), y.train.max()), ]
    output_pmesh[k] = ax.pcolormesh(
        grid_input[..., 0], grid_input[..., 1],
        grid_preds[k, 'mean'].values.reshape(grid_shape),
        shading='nearest', vmin=clim[0], vmax=clim[1], zorder=0
    )
    output_pmesh[k].set_rasterized(True)
    ax.contour(
        grid_input[..., 0], grid_input[..., 1], grid_preds[k, 'mean'].values.reshape(*grid_shape), colors='k', zorder=1,
        levels=full_y_levels
    )
    if m is not None:
        ax = axs[1, i]
        ax.set_visible(True)
        mean = grid_preds[(k, 'mean')]
        y_var = tdgplib.helper.get_likelihood_variance(m).numpy()
        std = np.sqrt(grid_preds[(k, 'var')] + y_var)
        nlpd_pmesh[k] = ax.pcolormesh(
            grid_input[..., 0], grid_input[..., 1],
            (
                nlpd(grid_preds[('true', 'mean')].values.reshape(-1, 1), mean.values.reshape(-1, 1),
                     std.values.reshape(-1, 1))
            ).reshape(grid_shape),
            shading='nearest', zorder=0,
        )
        nlpd_pmesh[k].set_rasterized(True)
    if k == 'true':
        ax.scatter(X.train[:, 0], X.train[:, 1], marker='.', edgecolor='w', linewidth=1, color='r', zorder=2)
ax = axs[0, 0]
ax.set_ylabel('Function output', fontweight='bold')
ax = axs[1, 1]
ax.set_ylabel('NLPD', fontweight='bold')
for ax in [*axs[0, 1:], *axs[1, 2:]]:
    ax.set_yticklabels([])

f.set_constrained_layout(True)
ticks = np.array([-1.5, 1])
ticks[0] = np.floor(ticks[0] * 10) / 10
ticks[1:-1] = np.around(ticks[1:-1], 1)
ticks[-1] = np.ceil(ticks[-1] * 10) / 10
for v in output_pmesh.values():
    v.set_clim([ticks[0], ticks[-1]])
cbar = f.colorbar(list(output_pmesh.values())[0], ax=[*axs[0]], pad=0.01, fraction=0.1)
cbar.set_ticks(ticks)

ticks = np.array([-1, 2])
ticks[0] = np.floor(ticks[0] * 10) / 10
ticks[1:-1] = np.around(ticks[1:-1], 1)
ticks[-1] = np.ceil(ticks[-1] * 10) / 10
for v in nlpd_pmesh.values():
    v.set_clim([ticks[0], ticks[-1]])
cbar = f.colorbar(list(nlpd_pmesh.values())[0], ax=[*axs[1]], pad=0.01, fraction=0.1)
cbar.set_ticks(ticks)

for ax in axs.flat:
    ax.set_xlim(grid_lims[0])
    ax.set_ylim(grid_lims[1])
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
for ax in axs[0, 1:].flat:
    ax.set_xticklabels([])

f.savefig('output_and_nlpd.png', bbox_inches='tight')

# %% [markdown]
# ## Figure - Latent spaces and lengthscale fields

# %%
labels = dict(
    true='True latent space',
    dkl='DKL',  # - NN output',
    dgp='CDGP',  # - Hidden layer',
    vmgp='TDGP',  # - $\mathbf{W}(\mathbf{x})\,\mathbf{x}$',
    dnsgp='DNSGP',
    id='Identity latent space'
)

extent = [*grid_lims.flat]
alpha = 2

to_plot = [
    'dkl', 'dgp', 'dnsgp', 'vmgp'
]
ncols = len(to_plot)
nrows = 2
f, axs = plt.subplots(nrows, ncols, figsize=np.array((2, 1.5)) * np.array((ncols, nrows)), sharex=True, sharey=True,
                      tight_layout=True)
for i, k in enumerate(to_plot):
    m = models.get(k)
    ax = axs[0, i]
    ax.set_visible(True)
    if isinstance(m, tdgplib.models.TDGP):
        hidden = tf.matmul(
            # n x 1 x D
            full_input[:, None],
            # n x D x Q
            tf.transpose(m.compute_qW(full_input)[0][..., 0], [2, 1, 0]),
        )[:, 0, :].numpy().reshape(*grid_shape, -1)
    elif k.startswith('vmgp'):
        hidden = m._evaluate_deep_gp(full_input, None, up_to_layer=-2).mean()
        hidden = m.f_layers[-1].compute_projection(
            tf.concat([full_input, hidden], axis=-1)
        ).numpy().reshape(*grid_shape, -1)
    elif k == 'dkl':
        hidden = m._evaluate_deep_gp(full_input, None, up_to_layer=-2).numpy().reshape(*grid_shape, -1)
    elif k == 'dgp':
        hidden = m._evaluate_deep_gp(full_input, None, up_to_layer=-2).mean().numpy().reshape(*grid_shape, -1)
    elif k == 'dnsgp':
        hidden = m._evaluate_deep_gp(full_input, None, up_to_layer=-2).mean()
        hidden = m.f_layers[-1].kernel.kernel.link_function(hidden + m.f_layers[-1].kernel.kernel.scaling_offset)
        hidden = (full_input / hidden).numpy().reshape(*grid_shape, -1)
    elif k == 'sgpr':
        hidden = m.kernel.lengthscales
        hidden = (full_input / hidden).numpy().reshape(*grid_shape, -1)
    elif k == 'true':
        hidden = tf.matmul(
            # n x 1 x D
            scaler.x.inverse_transform(full_input)[:, None],
            # n x D x Q
            full_W,
        )[:, 0, :].numpy().reshape(*grid_shape, -1)
        hidden = np.concatenate([hidden, np.zeros_like(hidden)], axis=-1)
    elif k == 'id':
        hidden = grid_input
    else:
        raise Exception(k)
    ax.set_title(labels.get(k))
    if k == 'dnsgp':
        ax.set_visible(True)
        ax.plot([-1, 1], [-1, 1], 'r', lw=5)
        ax.plot([-1, 1], [1, -1], 'r', lw=5)
    else:
        im = ax.imshow(
            np.moveaxis(cplot.get_srgb1(hidden[:, :, 0] + hidden[:, :, 1] * 1j, saturation_adjustment=alpha), 0, 1),
            origin='lower',
            extent=extent,
            aspect='auto',
            interpolation='gaussian',
            zorder=0
        )
        im.set_rasterized(True)
        ax.contour(grid_input[..., 0], grid_input[..., 1], full_y.reshape(*grid_shape), colors='k', zorder=1)

    ax = axs[1, i]
    ax.set_visible(True)
    if isinstance(m, tdgplib.models.TDGP):
        # n x D x Q
        W = tf.transpose(m.compute_qW(full_input)[0][..., 0], [2, 1, 0]).numpy()
        eigen = tf.linalg.eigvals(tf.linalg.matmul(W, W, transpose_b=True)).numpy().real.reshape(*grid_shape, -1)
    elif k == 'sgpr':
        W = 1 / m.kernel.lengthscales ** 2
        eigen = np.repeat(W[None, :], len(full_input), 0).reshape(*grid_shape, -1)
    elif k == 'dnsgp':
        hidden = m._evaluate_deep_gp(full_input, None, up_to_layer=-2).mean()
        hidden = m.f_layers[-1].kernel.kernel.link_function(hidden + m.f_layers[-1].kernel.kernel.scaling_offset)
        W = 1 / hidden ** 2
        eigen = W.numpy().reshape(*grid_shape, -1)
    else:
        ax.set_visible(False)

    if k in ['dgp', 'dkl']:
        ax.set_visible(True)
        ax.plot([-1, 1], [-1, 1], 'r', lw=5)
        ax.plot([-1, 1], [1, -1], 'r', lw=5)
    else:
        clim = [0.1, 1]
        pmesh0 = ax.pcolormesh(
            grid_input[..., 0], grid_input[..., 1], eigen.sum(-1) / eigen.sum(-1).max(),
            shading='nearest',
            vmin=clim[0], vmax=clim[1],
            cmap='Blues', zorder=0
        )
        pmesh0.set_rasterized(True)
        ax.contour(grid_input[..., 0], grid_input[..., 1], full_y.reshape(*grid_shape), colors='k', zorder=1)
for ax in axs.flat:
    ax.xaxis.set_ticks(np.arange(-2, 2 + 1, 1))
    ax.yaxis.set_ticks(np.arange(-2, 2 + 1, 1))
axs[1, 2].yaxis.set_ticks(np.arange(-2, 2 + 1, 1))

ax = axs[0, 0]
ax.set_ylabel('Latent\nspace', fontweight='bold')
ax = axs[1, 0]
ax.set_ylabel('Inverse\nlengthscales', fontweight='bold')

f.set_constrained_layout(True)
cbar = f.colorbar(pmesh0, ax=[*axs[1].flat], pad=0.01, fraction=0.1)
cbar.ax.yaxis.set_ticks([0, 0.5, 1]);

z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
rgb_vals = cplot.get_srgb1(z, abs_scaling=lambda z: np.full_like(z, 0.5), saturation_adjustment=alpha)
rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
newcmp = mpl.colors.ListedColormap(rgba_vals)
norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
cbar = f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=newcmp), ax=[*axs[0].flat], pad=0.01, fraction=0.1)
cbar.set_label(r'$\theta$', rotation=0, ha="center", va="top")
cbar.ax.yaxis.set_label_coords(0.5, -0.03)
cbar.set_ticks([-np.pi, 0, np.pi])
cbar.set_ticklabels([r"$-\pi$", '0', r"$\pi$"])
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cbar = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.gray),
    ax=[*axs[0].flat], pad=0.01, fraction=0.1
)
cbar.set_label('$r$', rotation=0, ha="center", va="top")
cbar.ax.yaxis.set_label_coords(0.5, -0.03)
cbar.set_ticks([0.0, 1 / (1 + 1), 1.0])
cbar.set_ticklabels(["0", "1", "$\\infty$", ])
f.savefig('latent_spaces.png', bbox_inches='tight')

# %% [markdown]
# ## Figure - Correlation at a point

# %%
labels = dict(
    sgpr='Shallow SGP',
    dnsgp='DNSGP',
    vmgp='TDGP',
    dkl='DKL',
    dgp='DGP',
)

to_plot = ['sgpr', 'dkl', 'dgp', 'dnsgp', 'vmgp']
ncols = len(to_plot)
nrows = math.ceil(len(to_plot) / ncols)
f, axs = plt.subplots(nrows, ncols, figsize=(12, 2), sharex=True, sharey=True, tight_layout=True)
for ax in axs.flat:
    ax.set_visible(False)
for i, k in enumerate(to_plot):
    ax = axs.flat[i]
    ax.set_visible(True)
    ax.set_title(labels[k])
    m = models.get(k)
    if isinstance(m, tdgplib.models.TDGP):
        hidden = tf.matmul(
            # n x 1 x D
            full_input[:, None],
            # n x D x Q
            tf.transpose(m.compute_qW(full_input)[0][..., 0], [2, 1, 0]),
        )[:, 0, :].numpy()
        Kf = m.s_kern.K(hidden).numpy()
    elif k == 'dkl':
        hidden = m._evaluate_deep_gp(full_input, None, up_to_layer=-2).numpy()
        Kf = m.f_layers[-1].kernel.kernel.K(hidden).numpy()
    elif k == 'dgp':
        hidden = m._evaluate_deep_gp(full_input, None, up_to_layer=-2).mean().numpy()
        Kf = m.f_layers[-1].kernel.K(hidden).numpy()
    elif k == 'dnsgp':
        hidden = m._evaluate_deep_gp(full_input, None, up_to_layer=-2).mean().numpy()
        hidden = np.concatenate([full_input, hidden], axis=-1)
        Kf = m.f_layers[-1].kernel.kernel.K(hidden).numpy()
    elif k == 'sgpr':
        Kf = m.kernel.K(full_input).numpy()
    else:
        raise Exception(k)
    if len(Kf.shape) == 4:
        Kf = Kf[:, 0, :, 0]
    chosen_i = 40 * grid_shape[0] + 60
    correlation = Kf[chosen_i, :] / np.sqrt(Kf[chosen_i, chosen_i] * np.diag(Kf))
    ax.plot(*full_input[chosen_i], 'Xk', markeredgecolor='w')
    pmesh0 = ax.pcolormesh(
        grid_input[..., 0], grid_input[..., 1], correlation.reshape(*grid_shape),
        shading='nearest',
        vmin=0, vmax=1,
        zorder=0
    )
    pmesh0.set_rasterized(True)
    ax.contour(grid_input[..., 0], grid_input[..., 1], full_y.reshape(*grid_shape), colors='k', zorder=1)
axs.flat[0].xaxis.set_ticks(np.arange(-2, 2 + 1, 1))
axs.flat[0].yaxis.set_ticks(np.arange(-2, 2 + 1, 1));
f.tight_layout(rect=(0, 0, 0.89, 1))
cbar = f.colorbar(pmesh0, ax=[*axs.flat], pad=0.01, fraction=0.1)
cbar.ax.yaxis.set_ticks([0, 0.5, 1])
f.savefig('correlation.png', bbox_inches='tight')

# %% [markdown]
# ## Figure - Dimensional reduction

# %%
labels = dict(sgpr='Shallow SGP', dnsgp='DNSGP', vmgp='TDGP', dkl='DKL', dgp='DGP', )
to_plot = ['sgpr', 'dkl', 'dgp', 'vmgp']
ncols = len(to_plot)
nrows = math.ceil(len(to_plot) / ncols)
f, axs = plt.subplots(nrows, ncols, figsize=(12, 2), sharex=True, sharey=True, tight_layout=True)
for ax in axs.flat:
    ax.set_visible(False)
for i, k in enumerate(to_plot):
    m = models.get(k)
    if isinstance(m, tdgplib.models.TDGP):
        values = np.array([k.variance.numpy().item() for k in m.w_kerns])
    elif k.startswith('vmgp'):
        values = np.array([k.variance.numpy().item() for k in dict.fromkeys(m.f_layers[0].kernel.kernels)])
    elif k == 'dkl':
        values = 1 / m.f_layers[-1].kernel.kernel.lengthscales.numpy()
    elif k == 'dgp':
        values = 1 / m.f_layers[-1].kernel.kernel.lengthscales.numpy()
    elif k == 'dnsgp':
        continue
    elif k == 'sgpr':
        values = 1 / m.kernel.lengthscales.numpy()
    else:
        raise Exception(k)
    ax = axs.flat[i]
    ax.set_visible(True)
    values = np.atleast_1d(values)
    values = np.sort(values)[::-1]
    ax.bar(np.arange(len(values)), values / values.max())
    ax.set_title(labels[k])
    ax.set_xticks(np.arange(len(values)), labels=['Most relevant', 'Least relevant'])
    ax.set_xlabel('Latent dimension')
axs.flat[0].set_xlabel('Input dimension')
axs.flat[0].set_ylabel('% of largest', fontsize=9)
f.savefig('dim_red.png', bbox_inches='tight')
