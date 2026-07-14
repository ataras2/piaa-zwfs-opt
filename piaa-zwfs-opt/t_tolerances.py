# %%
import os

# # Must be set before JAX is imported!
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# 64 bit jax
os.environ["JAX_ENABLE_X64"] = "False"

import dlWFSoptimize as dlwfs
import matplotlib.pyplot as plt
import jax.numpy as np
import os
import asdf
import numpy as onp
import zodiax as zdx
import jax.random as jr

import jax
import matplotlib.colors as mcolors
import scienceplots

plt.style.use(["science"])
plt.rcParams.update({"font.size": 8})

jax.devices()
# %%
ModelCore = dlwfs.ModelCore

fig_pth = os.path.join(
    "..",
    "results",
    "figures",
)

# %%
pth = os.path.join(
    "..",
    "results",
    "dl_optimizations",
    "pre_det_change_opts",
    "Fourier_basis_5_terms_Z_band_with_PIAA_annular_mask",
)

model, _ = dlwfs.load_model(pth)

# %%
plt.imshow(model.img())
plt.xticks([])
plt.yticks([])
# %%
loss_type = "cov_trace_normed"


def loss_fn(p, model):
    # scale decentres to be in units of micron
    model = model.set(p)
    return model.loss(loss_type=loss_type)


# %%

include_decentres = True


params_to_hessian = [
    "params.piaa1_coeffs",
    "params.piaa2_coeffs",
    "params.piaa1_f_rho_max",
    "params.piaa2_f_rho_max",
    "params.phase_mask_coeffs",
    # "params.piaa1_lens1_decentre",
    # "params.piaa1_lens2_decentre",
    # "params.piaa2_lens1_decentre",
    # "params.piaa2_lens2_decentre",
    # "params.piaa_distance",
]

if include_decentres:
    params_to_hessian += [
        "params.piaa1_lens1_decentre",
        "params.piaa1_lens2_decentre",
        "params.piaa2_lens1_decentre",
        "params.piaa2_lens2_decentre",
    ]

params = {}
for param in params_to_hessian:
    params[param] = model.get(param)

n_params = jax.flatten_util.ravel_pytree(params)[0].size
print(f"N params: {n_params}")

# %%

loss_fn_jit = jax.jit(lambda p: loss_fn(p, model))

opt_value = loss_fn_jit(params)
opt_value

# %%
n_samples = 1000

# samples = jr.normal(
#     jr.key(0), (n_samples, jax.flatten_util.ravel_pytree(params)[0].size)
# )

# mean value will be the optimal points, i.e. the current parameters
mean, unflatten = jax.flatten_util.ravel_pytree(params)
# covariance will be a block diagonal matrix with each block as a simple diagonal
sigmas = {
    "params.piaa2_f_rho_max": 1e-3 * params["params.piaa2_f_rho_max"],
    "params.piaa1_f_rho_max": 1e-3 * params["params.piaa1_f_rho_max"],
    "params.piaa1_coeffs": 40e-3,
    "params.piaa2_coeffs": 40e-3,
    "params.phase_mask_coeffs": 2e-3,
    "params.piaa1_lens1_decentre": 5e0,
    "params.piaa1_lens2_decentre": 5e0,
    "params.piaa2_lens1_decentre": 5e0,
    "params.piaa2_lens2_decentre": 5e0,
}

Cov = np.zeros((n_params, n_params))
idxs = unflatten(np.arange(n_params))
for param, sigma in sigmas.items():
    Cov = Cov.at[idxs[param], idxs[param]].set(sigma**2)


def eval_sample(sample):
    return loss_fn_jit(unflatten(sample))


res = jax.lax.map(
    eval_sample,
    jr.multivariate_normal(jr.key(0), mean, Cov, (n_samples,)),
    batch_size=3,
)

res.shape

# %%
plt.hist(res, bins=20)
plt.axvline(opt_value, color="r", label="opt value")
plt.xlabel("Loss value")
plt.ylabel("Count")
plt.yscale("log")


def sub_opt_frac(res, opt_value):
    return res / opt_value - 1


print(f"Summary statistics (on sub-optimal fraction):")
print(f"Mean: {sub_opt_frac(res, opt_value).mean():.2e}")
print(f"Std: {sub_opt_frac(res, opt_value).std():.2e}")
print(f"Median: {np.median(sub_opt_frac(res, opt_value)):.2e}")


# %%
def mc_sample_fn(
    fn,
    sample_generator,
    unflatten,
    n_samples,
    key,
    batch_size=1,
):
    """
    Monte Carlo sampling of a function.

    Args:
        fn: function of the form fn(params, args) that returns a value
        sample_generator: function that generates samples of the parameters, e.g.
            jr.normal(key, shape)
        unflatten: function to unflatten the sampled parameters to pytree form
        n_samples: number of samples to generate
        key: jax.random.PRNGKey for random number generation
        batch_size: number of samples to evaluate in a batch
    """
    samples = sample_generator(key, (n_samples,))

    def eval_sample(sample):
        return fn(unflatten(sample))

    return jax.lax.map(eval_sample, samples, batch_size=batch_size)

n_samples = 10
res_tmp = mc_sample_fn(
    loss_fn_jit,
    lambda key, shape: jr.multivariate_normal(key, mean, Cov, shape),
    unflatten,
    n_samples,
    jr.key(0),
    batch_size=3,
)

# %%
