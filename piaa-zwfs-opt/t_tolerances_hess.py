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
axs["piaa_surf"] = plt.subplot(gs[1, 0])
axs["piaa_decentres1"] = plt.subplot(gs[1, 1])
axs["piaa_decentres2"] = plt.subplot(gs[1, 2])
axs["phase_mask"] = plt.subplot(gs[0, 1])
axs["phase_mask_in_mask_space"] = plt.subplot(gs[0, 2])

axs_arr = [
    [axs["all"], axs["phase_mask"], axs["phase_mask_in_mask_space"]],
    [axs["piaa_surf"], axs["piaa_decentres1"], axs["piaa_decentres2"]],
]


def format_axs_border(ax, color=None, linewidth=2.0):
    # remove minor ticks
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="y", which="minor", left=False)
    ax.tick_params(axis="x", which="minor", top=False)
    ax.tick_params(axis="y", which="minor", right=False)

    if color is not None:
        ax.spines["top"].set_color(color)
        ax.spines["bottom"].set_color(color)
        ax.spines["left"].set_color(color)
        ax.spines["right"].set_color(color)

    if linewidth is not None:
        ax.spines["top"].set_linewidth(linewidth)
        ax.spines["bottom"].set_linewidth(linewidth)
        ax.spines["left"].set_linewidth(linewidth)
        ax.spines["right"].set_linewidth(linewidth)


# all is the same as above
im = axs["all"].imshow(H_normed, norm=mcolors.CenteredNorm(vcenter=0), cmap="RdBu_r")
cbar = fig.colorbar(im, ax=axs["all"], fraction=0.046, pad=0.04)
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
cbar = fig.colorbar(im, ax=axs["piaa_surf"], fraction=0.046, pad=0.04)
axs["piaa_surf"].set_title("PIAA parameters")
param_names = (
    "params.piaa1_coeffs",
    "params.piaa1_f_rho_max",
    "params.piaa2_coeffs",
    "params.piaa2_f_rho_max",
)

nice_label = {
    "params.piaa1_coeffs": "L1 $Q_{{\\mathrm BFS}}$",
    "params.piaa1_f_rho_max": "L1 f($\\rho_{\\mathrm{max}}$)",
    "params.piaa2_coeffs": "L2 $Q_{{\\mathrm BFS}}$",
    "params.piaa2_f_rho_max": "L2 f($\\rho_{\\mathrm{max}}$)",
}

xlabels = []
for param in param_names:
    a = unflatten(np.arange(jax.flatten_util.ravel_pytree(params)[0].size))[param]
    name = param.split(".")[-1]
    if a.size > 1:
        tmp = []
        for i in range(a.size):
            tmp.append(f"{nice_label[param]}[{i}]")
        xlabels += tmp
    else:
        xlabels.append(f"{nice_label[param]}")

axs["piaa_surf"].set_xticks(np.arange(len(xlabels)))
axs["piaa_surf"].set_xticklabels(xlabels, rotation=90, fontsize=6)
# rm yticks
axs["piaa_surf"].set_yticks(np.arange(len(xlabels)))
axs["piaa_surf"].set_yticklabels([])

format_axs_border(axs["piaa_surf"], color=group_colors[1], linewidth=2)

# now to do the mask block
im = axs["phase_mask"].imshow(
    (H_normed)[idxs["params.phase_mask_coeffs"], :][
        :, idxs["params.phase_mask_coeffs"]
    ],
    norm=mcolors.CenteredNorm(vcenter=0),
    cmap="RdBu_r",
)
cbar = fig.colorbar(im, ax=axs["phase_mask"], fraction=0.046, pad=0.04)
axs["phase_mask"].set_title("Phase mask coefficients")
axs["phase_mask"].set_xticks(np.arange(len(idxs["params.phase_mask_coeffs"])))
axs["phase_mask"].set_xticklabels([])
axs["phase_mask"].set_yticks(np.arange(len(idxs["params.phase_mask_coeffs"])))
axs["phase_mask"].set_yticklabels([])
format_axs_border(axs["phase_mask"], color=group_colors[0], linewidth=2)

# now the mask in mask space
# H_mask = (H_normed)[idxs["params.phase_mask_coeffs"], :][
#     :, idxs["params.phase_mask_coeffs"]
# ]
# mask = model.get("tel.ThreePlaneSystem").mask_at_focus
# pattern = mask.set({"coefficients": np.diag(H_mask)}).eval_basis() * 1e6

# im = axs["phase_mask_in_mask_space"].imshow(pattern, cmap="viridis")
# # cbar
# cbar = fig.colorbar(im, ax=axs["phase_mask_in_mask_space"], fraction=0.046, pad=0.04)
# axs["phase_mask_in_mask_space"].set_title("Diagonal Hessian in mask space")
# axs["phase_mask_in_mask_space"].set_xticks([])
# axs["phase_mask_in_mask_space"].set_yticks([])
# format_axs_border(axs["phase_mask_in_mask_space"], color=group_colors[0], linewidth=2)

# xscale = model.tel.ThreePlaneSystem.focus_pixel_scale/(model.tel.wavelengths.mean()/(6.5*model.tel.Magnifier.magnification))

xvals = np.arange(len(idxs["params.phase_mask_coeffs"]))
axs["phase_mask_in_mask_space"].plot(
    xvals,
    np.diag(
        (H_normed)[idxs["params.phase_mask_coeffs"], :][
            :, idxs["params.phase_mask_coeffs"]
        ]
    ),
    color=group_colors[0],
)
format_axs_border(axs["phase_mask_in_mask_space"], color=group_colors[0], linewidth=2)
axs["phase_mask_in_mask_space"].set_title("Diagonal Hessian in mask space")
axs["phase_mask_in_mask_space"].set_xlabel("Radius index")


# now the two decenters
piaa1_idxs = np.arange(rect_positions[2][0], rect_positions[2][1])
im = axs["piaa_decentres1"].imshow(
    (H_normed)[piaa1_idxs, :][:, piaa1_idxs],
    norm=mcolors.CenteredNorm(vcenter=0),
    cmap="RdBu_r",
)
cbar = fig.colorbar(im, ax=axs["piaa_decentres1"], fraction=0.046, pad=0.04)
axs["piaa_decentres1"].set_title("PIAA1 decentres")
axs["piaa_decentres1"].set_xticks(np.arange(len(piaa1_idxs)))
axs["piaa_decentres1"].set_xticklabels([])
axs["piaa_decentres1"].set_yticks(np.arange(len(piaa1_idxs)))
axs["piaa_decentres1"].set_yticklabels([])
format_axs_border(axs["piaa_decentres1"], color=group_colors[2], linewidth=2)
# x labels
nice_label = {
    "params.piaa1_lens1_decentre": "L1 decentre",
    "params.piaa1_lens2_decentre": "L2 decentre",
    "params.piaa2_lens1_decentre": "L1 decentre",
    "params.piaa2_lens2_decentre": "L2 decentre",
}
params_to_plot = [
    "params.piaa1_lens1_decentre",
    "params.piaa1_lens2_decentre",
]
xlabels = []
for param in params_to_plot:
    a = unflatten(np.arange(jax.flatten_util.ravel_pytree(params)[0].size))[param]
    name = param.split(".")[-1]
    if a.size > 1:
        tmp = []
        for i in range(a.size):
            tmp.append(f"{nice_label[param]}[{i}]")
        xlabels += tmp
    else:
        xlabels.append(f"{nice_label[param]}")
axs["piaa_decentres1"].set_xticks(np.arange(len(xlabels)))
axs["piaa_decentres1"].set_xticklabels(xlabels, rotation=90, fontsize=6)


piaa2_idxs = np.arange(rect_positions[3][0], rect_positions[3][1])
im = axs["piaa_decentres2"].imshow(
    (H_normed)[piaa2_idxs, :][:, piaa2_idxs],
    norm=mcolors.CenteredNorm(vcenter=0),
    cmap="RdBu_r",
)
cbar = fig.colorbar(im, ax=axs["piaa_decentres2"], fraction=0.046, pad=0.04)
axs["piaa_decentres2"].set_title("PIAA2 decentres")
axs["piaa_decentres2"].set_xticks(np.arange(len(piaa2_idxs)))
axs["piaa_decentres2"].set_xticklabels([])
axs["piaa_decentres2"].set_yticks(np.arange(len(piaa2_idxs)))
axs["piaa_decentres2"].set_yticklabels([])
format_axs_border(axs["piaa_decentres2"], color=group_colors[3], linewidth=2)
axs["piaa_decentres2"].set_xticks(np.arange(len(xlabels)))
axs["piaa_decentres2"].set_xticklabels(xlabels, rotation=90, fontsize=6)


# add letters to the top left of each subplot
letters = "abcdef"
# letters in the form (a) in top left corner, using axs_array to infer the order
for i, row in enumerate(axs_arr):
    for j, ax in enumerate(row):
        ax.text(
            -0.05,
            1.05,
            f"({letters[i*len(row)+j]})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

# annotate inside the all with the right letter in the top right corner

annotation = (
    (0.82, 0.9, "(b-c)"),
    (0.82, 0.35, "(d)"),
    (0.98, 0.14, "(e)"),
    (0.92, 0.00, "(f)"),
)

for i, a in enumerate(annotation):
    x, y, txt = a
    axs["all"].text(
        x,
        y,
        txt,
        color=f"C{i}",
        fontsize=8,
        ha="center",
        va="center",
        transform=axs["all"].transAxes,
    )

plt.savefig(
    os.path.join(fig_pth, "hessian_all_params.pdf"), dpi=300, bbox_inches="tight"
)
# %%
# save everything and the order of params
import numpy as onp

onp.savez(
    os.path.join(fig_pth, "hessian_all_params.npz"),
    H_reordered=H_reordered,
    H_raw=H,
    H_normed=H_normed,
    new_order=new_order,
    params=params,
    # groups=groups,
    group_colors=group_colors,
)
