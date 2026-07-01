# %%
import os

# # Must be set before JAX is imported!
os.environ["JAX_PLATFORM_NAME"] = "cpu"
# 64 bit jax
os.environ["JAX_ENABLE_X64"] = "False"

import dlWFSoptimize as dlwfs
import matplotlib.pyplot as plt
import jax.numpy as np
import os
import asdf
import numpy as onp
import zodiax as zdx

import jax
import matplotlib.colors as mcolors
import scienceplots

plt.style.use(["science"])
plt.rcParams.update({"font.size": 8})

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
model.loss(loss_type=loss_type)


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


def loss_fn(p, model):
    # scale decentres to be in units of micron
    model = model.set(p)
    return model.loss(loss_type=loss_type)


loss_fn(params, model), loss_fn(
    jax.tree_util.tree_map_with_path(
        lambda path, val: (
            val + 1000 if "params.piaa2_lens2_decentre" in path[-1].key else val
        ),
        params,
    ),
    model,
)
# %%
H, unflatten = zdx.hessian(
    lambda p: loss_fn(p, model),
    params,
    nbatches=n_params,
)
H.shape
# %%
plt.imshow(H, norm=mcolors.CenteredNorm(vcenter=0), cmap="RdBu_r")
plt.colorbar()

plt.figure()
idxs = unflatten(np.arange(jax.flatten_util.ravel_pytree(params)[0].size))

plt.imshow(
    H[idxs["params.phase_mask_coeffs"], :][:, idxs["params.phase_mask_coeffs"]],
    norm=mcolors.CenteredNorm(vcenter=0),
    cmap="RdBu_r",
)
plt.colorbar()
plt.title("Hessian of phase mask coefficients")


plt.figure()
idxs = unflatten(np.arange(jax.flatten_util.ravel_pytree(params)[0].size))
params_to_plot = [
    "params.piaa2_lens1_decentre",
    "params.piaa2_lens2_decentre",
]
idxs_to_plot = []
for param in params_to_plot:
    idxs_to_plot += list(idxs[param])

plt.imshow(
    H[idxs_to_plot, :][:, idxs_to_plot],
    norm=mcolors.CenteredNorm(vcenter=0),
    cmap="RdBu_r",
)
plt.colorbar()
plt.title("Hessian of PIAA decentres")


plt.figure()
idxs = unflatten(np.arange(jax.flatten_util.ravel_pytree(params)[0].size))
params_to_plot = [
    "params.piaa1_coeffs",
    "params.piaa2_coeffs",
    "params.piaa1_f_rho_max",
    "params.piaa2_f_rho_max",
]
idxs_to_plot = []
for param in params_to_plot:
    if idxs[param].size > 1:
        idxs_to_plot += list(idxs[param])
    else:
        idxs_to_plot.append(idxs[param])

plt.imshow(
    H[idxs_to_plot, :][:, idxs_to_plot],
    norm=mcolors.CenteredNorm(vcenter=0),
    cmap="RdBu_r",
)
plt.colorbar()
plt.title("Hessian of PIAA coefficients and f_rho_max")


# %%
loss_fn_jit = jax.jit(lambda p: loss_fn(p, model))


# %%