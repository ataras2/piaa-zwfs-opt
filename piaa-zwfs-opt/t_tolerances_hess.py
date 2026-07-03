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

optimal_value = loss_fn(params, model)


loss_fn(params, model), loss_fn(
    jax.tree_util.tree_map_with_path(
        lambda path, val: (
            val + 1 if "params.piaa1_lens2_decentre" in path[-1].key else val
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

H.block_until_ready()

H.shape
# %%

# show each block in the figure with a colored rectangle
groups = [
    ("params.phase_mask_coeffs",),
    (
        "params.piaa1_coeffs",
        "params.piaa1_f_rho_max",
        "params.piaa2_coeffs",
        "params.piaa2_f_rho_max",
    ),
    (
        "params.piaa1_lens1_decentre",
        "params.piaa1_lens2_decentre",
    ),
    (
        "params.piaa2_lens1_decentre",
        "params.piaa2_lens2_decentre",
    ),
]

group_colors = [f"C{i}" for i in range(len(groups))]

idx_per_group = []
for group in groups:
    idxs = []
    for param in group:
        a = unflatten(np.arange(jax.flatten_util.ravel_pytree(params)[0].size))[param]
        if a.size > 1:
            idxs += list(a)
        else:
            idxs.append(a)
    idx_per_group.append(idxs)

# make a list of the new indicies in this order to reoganize the Hessian
new_order = []
for idxs in idx_per_group:
    new_order += idxs

H_reordered = H[new_order, :][:, new_order]
H_normed = H_reordered / optimal_value

plt.figure(dpi=200)
plt.imshow(H_normed, norm=mcolors.CenteredNorm(vcenter=0), cmap="RdBu_r")
plt.colorbar()

# now the rectangle positions have changed, so we need to recalculate them
rect_positions = []
cur_pos = 0
for g in groups:
    group_size = sum(
        [
            unflatten(np.arange(jax.flatten_util.ravel_pytree(params)[0].size))[
                param
            ].size
            for param in g
        ]
    )
    rect_positions.append((cur_pos, cur_pos + group_size))
    cur_pos += group_size

plt.axis("off")

for i, (start, end) in enumerate(rect_positions):
    plt.gca().add_patch(
        plt.Rectangle(
            (start - 0.5, start - 0.5),
            end - start,
            end - start,
            fill=False,
            edgecolor=group_colors[i],
            linewidth=0.5,
        )
    )
    # plt.text(
    #     (start + end) / 2,
    #     (start + end) / 2,
    #     f"Group {i+1}",
    #     color="black",
    #     fontsize=8,
    #     ha="center",
    #     va="center",
    # )


plt.xlabel("Parameter index")
plt.ylabel("Parameter index")
plt.title("Hessian of all parameters")

plt.figure()
idxs = unflatten(np.arange(jax.flatten_util.ravel_pytree(params)[0].size))

plt.imshow(
    (H_normed)[idxs["params.phase_mask_coeffs"], :][
        :, idxs["params.phase_mask_coeffs"]
    ],
    norm=mcolors.CenteredNorm(vcenter=0),
    cmap="RdBu_r",
)
plt.colorbar()
plt.title("Phase mask coefficients")


plt.figure()
idxs = unflatten(np.arange(jax.flatten_util.ravel_pytree(params)[0].size))
params_to_plot = [
    "params.piaa1_lens1_decentre",
    "params.piaa1_lens2_decentre",
    "params.piaa2_lens1_decentre",
    "params.piaa2_lens2_decentre",
]
idxs_to_plot = []
for param in params_to_plot:
    idxs_to_plot += list(idxs[param])

plt.imshow(
    (H_normed)[idxs_to_plot, :][:, idxs_to_plot],
    norm=mcolors.CenteredNorm(vcenter=0),
    cmap="RdBu_r",
)
plt.colorbar()
plt.title("Hessian of PIAA decentres")


plt.figure()
idxs = unflatten(np.arange(jax.flatten_util.ravel_pytree(params)[0].size))
params_to_plot = [
    "params.piaa1_coeffs",
    "params.piaa1_f_rho_max",
    "params.piaa2_coeffs",
    "params.piaa2_f_rho_max",
]
idxs_to_plot = []
for param in params_to_plot:
    if idxs[param].size > 1:
        idxs_to_plot += list(idxs[param])
    else:
        idxs_to_plot.append(idxs[param])

plt.imshow(
    (H_normed)[idxs_to_plot, :][:, idxs_to_plot],
    norm=mcolors.CenteredNorm(vcenter=0),
    cmap="RdBu_r",
)
plt.colorbar()
plt.title("Hessian of PIAA coefficients and f_rho_max")

# %%
from matplotlib.gridspec import GridSpec

# 2x3 grid

fig = plt.figure(figsize=(10, 6), dpi=200)
gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)

axs = {}
axs["all"] = plt.subplot(gs[0, 0])
axs["piaa_surf"] = plt.subplot(gs[0, 1])
axs["piaa_decentres1"] = plt.subplot(gs[1, 0])
axs["piaa_decentres2"] = plt.subplot(gs[1, 1])
axs["phase_mask"] = plt.subplot(gs[0, 2])
axs["phase_mask_in_mask_space"] = plt.subplot(gs[1, 2])

# all is the same as above
im = axs["all"].imshow(H_normed, norm=mcolors.CenteredNorm(vcenter=0), cmap="RdBu_r")
cbar = fig.colorbar(im, ax=axs["all"])
axs["all"].set_title("Hessian of all parameters")
axs["all"].axis("off")

# rectangles for groups
for i, (start, end) in enumerate(rect_positions):
    axs["all"].add_patch(
        plt.Rectangle(
            (start - 0.5, start - 0.5),
            end - start,
            end - start,
            fill=False,
            edgecolor=group_colors[i],
            linewidth=1.5,
        )
    )

# piaa surf time
idxs_to_plot = np.arange(rect_positions[1][0], rect_positions[1][1])
im = axs["piaa_surf"].imshow(
    (H_normed)[idxs_to_plot, :][:, idxs_to_plot],
    norm=mcolors.CenteredNorm(vcenter=0),
    cmap="RdBu_r",
)
cbar = fig.colorbar(im, ax=axs["piaa_surf"])
axs["piaa_surf"].set_title("PIAA parameters")
param_names = (
    "params.piaa1_coeffs",
    "params.piaa1_f_rho_max",
    "params.piaa2_coeffs",
    "params.piaa2_f_rho_max",
)
xlabels = []
for param in param_names:
    a = unflatten(np.arange(jax.flatten_util.ravel_pytree(params)[0].size))[param]
    name = param.split(".")[-1]
    if a.size > 1:
        xlabels += [f"{name}[{i}]" for i in range(a.size)]
    else:
        xlabels.append(name)
axs["piaa_surf"].set_xticks(np.arange(len(xlabels)))
axs["piaa_surf"].set_xticklabels(xlabels, rotation=60, fontsize=6)
# remove minor ticks
axs["piaa_surf"].tick_params(axis="x", which="minor", bottom=False)
axs["piaa_surf"].tick_params(axis="y", which="minor", left=False)
axs["piaa_surf"].tick_params(axis="x", which="minor", top=False)
axs["piaa_surf"].tick_params(axis="y", which="minor", right=False)