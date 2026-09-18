# %%
import os

# # Must be set before JAX is imported!
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# 64 bit jax
os.environ["JAX_ENABLE_X64"] = "False"

import dlWFSoptimize as dlwfs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax.numpy as np
import os
import asdf
import numpy as onp
import zodiax as zdx

# %%

ModelCore = dlwfs.ModelCore


fig_savepth = os.path.join(
    "..",
    "results",
    "figures",
    "different_apertures.pdf",
)

# %%
pth = os.path.join(
    "..",
    "results",
    "dl_optimizations",
    "different_apertures",
)
# find all folders in pth
all_folders = [f for f in os.listdir(pth) if os.path.isdir(os.path.join(pth, f))]
all_folders
# %%
models = {
    f.split("_")[0]: dlwfs.load_model(os.path.join(pth, f))[0] for f in all_folders
}

# %%
# for each model, set detector downsample to 1
# m = m.set("tel.layers.detector.Downsample.kernel_size", 1)
# for name, model in models.items():
#     models[name] = model.set("tel.detector.Downsample.kernel_size", 1)


# %%
m = models["gmt"]

m
# %%
import collections
import Aspheres


def img_at_PIAA(model, after_piaa=True):
    """Return the full-resolution intensity before or after the first PIAA pair."""
    prev_layers = model._update_tel().get("layers")  # ordered dict
    layers_kept = [
        "TransmissiveLayer",
        "BasisLayer",
        "Magnifier",
    ]
    if after_piaa:
        layers_kept.append("PIAAset_0")

    # prev_layers["PIAAset_0"] = prev_layers["PIAAset_0"].set(
    #     "piaa_lens_1.asphere_profile.coeffs", np.ones(8)
    # )

    subset_tel = model._update_tel().set(
        ["layers", "detector.Downsample.kernel_size"],
        [
            collections.OrderedDict(
                {k: v for k, v in prev_layers.items() if k in layers_kept}
            ),
            1,
        ],
    )

    return subset_tel.model()


plt.imshow(img_at_PIAA(m))


# %%

plt.imshow(m.img())

# %%

import jax

perturb_index = -1


def compute_J_single_mode(model, mode_index):
    n_modes = len(model.tel.BasisLayer.coefficients)

    def set_mode_index(model, mode_index, value):
        new_coeffs = model.tel.BasisLayer.coefficients.at[mode_index].set(value)
        return model.set(
            ["params.phase_coefficients"],
            new_coeffs,
        ).img()

    J = jax.jacfwd(lambda val: set_mode_index(model, mode_index, val))(0.0)
    return J


J0 = compute_J_single_mode(m, perturb_index)


# %%
def model_without_inv_PIAA(model):
    prev_layers = model._update_tel().get("layers")  # ordered dict
    layers_rem = [
        "PIAAset_1",
    ]

    subset_tel = model._update_tel().set(
        ["layers"],
        [
            collections.OrderedDict(
                {k: v for k, v in prev_layers.items() if k not in layers_rem}
            )
        ],
    )

    old_params = model.params
    # remove anything that start with piaa2
    subset_params = {
        k: v
        for k, v in old_params.items()
        if not k.startswith("piaa2_") and not k.startswith("inv_PIAA_")
    }
    old_tel_params = model.tel_params_to_params

    subset_tel_params = {
        k: v
        for k, v in old_tel_params.items()
        if not k.startswith("PIAAset_1") and not k.startswith("inv_PIAA_")
    }

    return model.set(
        ["tel", "params", "tel_params_to_params"],
        [subset_tel, subset_params, subset_tel_params],
    )


no_inv_model = model_without_inv_PIAA(m)

J0_no_inv = compute_J_single_mode(no_inv_model, perturb_index)

# %%
J = m.compute_standard_sensitivity(pupil_diameter=6.5, n_freqs=2, return_J=True)


# %%
loss_type = "cov_trace_normed"
no_inv_model.loss(loss_type), m.loss(loss_type)


# %%
ideal_m = dlwfs.generate_idealwfs_optics(m)

J0_ideal = compute_J_single_mode(ideal_m, perturb_index)


# %%
# cbounds are the max of the maxes of each
clim = max(np.max(np.abs(J0)), np.max(np.abs(J0_no_inv)), np.max(np.abs(J0_ideal)))

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(J0, cmap="bwr", vmin=-clim, vmax=clim)
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(
    J0_no_inv,
    cmap="bwr",
    vmin=-clim,
    vmax=clim,
)
plt.colorbar()


plt.subplot(1, 3, 3)
plt.imshow(
    J0_ideal,
    cmap="bwr",
    vmin=-clim,
    vmax=clim,
)
plt.colorbar()

# %%
basis = m.tel.BasisLayer.basis
plt.imshow(basis[-1])

# %%
import matplotlib.gridspec as gridspec

import scienceplots

plt.style.use(["science"])

# text size
plt.rcParams.update({"font.size": 9})

model_names = [
    "magellan",
    "gmt",
    "elt",
]

row_names = [
    "After PIAA",
    "Flat",
    "J no inv",
    "J",
]

n_apterures = len(models)
n_rows = 4

fig = plt.figure(figsize=(3.46457, 4.2), dpi=300)

gs0 = gridspec.GridSpec(n_rows, n_apterures, figure=fig, hspace=0.1, wspace=0.01)

axs = {}
for j, model_name in enumerate(model_names):
    for i, row_name in enumerate(row_names):
        axs[(model_name, row_name)] = fig.add_subplot(gs0[i, j])

for model_name in model_names:
    model = models[model_name]

    # After PIAA
    axs[(model_name, "After PIAA")].imshow(np.sqrt(img_at_PIAA(model)))
    axs[(model_name, "After PIAA")].set_xticks([])
    axs[(model_name, "After PIAA")].set_yticks([])

    # PSF
    img = model.img()
    axs[(model_name, "Flat")].imshow(img, cmap="gray")
    axs[(model_name, "Flat")].set_xticks([])
    axs[(model_name, "Flat")].set_yticks([])

    diams_from_models = {
        "magellan": 6.5,
        "gmt": 24.5,
        "elt": 39.0,
    }

    # mode Jacobian
    # J = compute_J_single_mode(model, perturb_index)
    J = model.compute_standard_sensitivity(
        pupil_diameter=diams_from_models[model_name],
        n_freqs=2,
        return_J=True,
    )[1][:, :, -1]

    no_inv_model = model_without_inv_PIAA(model)
    # J_no_inv = compute_J_single_mode(no_inv_model, perturb_index)
    J_no_inv = no_inv_model.compute_standard_sensitivity(
        pupil_diameter=diams_from_models[model_name],
        n_freqs=2,
        return_J=True,
    )[1][:, :, -1]

    clim = max(np.max(np.abs(J)), np.max(np.abs(J_no_inv)))
    im = axs[(model_name, "J")].imshow(
        J,
        cmap="bwr",
        vmin=-clim,
        vmax=clim,
        interpolation="none",
    )
    axs[(model_name, "J")].set_xticks([])
    axs[(model_name, "J")].set_yticks([])

    # no inv PIAA mode Jacobian
    im = axs[(model_name, "J no inv")].imshow(
        J_no_inv,
        cmap="bwr",
        vmin=-clim,
        vmax=clim,
        interpolation="none",
    )
    axs[(model_name, "J no inv")].set_xticks([])
    axs[(model_name, "J no inv")].set_yticks([])


# plt.tight_layout()

col_titles = [
    "Magellan",
    "GMT",
    "ELT",
]

row_labels = [
    "After PIAA",
    r"$I_0$",
    r"$J_\mathrm{no\ inv}$",
    r"$J$",
]

# labels over each column are the model names
for j, model_name in enumerate(model_names):
    axs[(model_name, "After PIAA")].set_title(col_titles[j], pad=5)

# labels to the left of each row
for i, row_name in enumerate(row_names):
    axs[(model_names[0], row_name)].set_ylabel(row_labels[i], rotation=90, labelpad=5)

plt.savefig(fig_savepth, bbox_inches="tight")

# %%
def radial_intensity_profile(intensity, diameter_over_D):
    """Average intensity in one-pixel annuli, returning their radii in r/D."""
    intensity = onp.asarray(intensity)
    npixels = intensity.shape[0]
    coords = onp.arange(npixels) - (npixels - 1) / 2
    radius = onp.hypot(coords[:, None], coords[None, :])
    bins = radius.astype(int)
    n_bins = npixels // 2
    # Use complete annuli only, excluding the corners of the square image.
    valid = bins < n_bins
    counts = onp.bincount(bins[valid], minlength=n_bins)
    sums = onp.bincount(bins[valid], weights=intensity[valid], minlength=n_bins)
    radii = (onp.arange(n_bins) + 0.5) * float(diameter_over_D) / npixels
    return radii, sums / counts


profile_fig, profile_ax = plt.subplots(figsize=(3.46457, 2.6), dpi=300)

for i, model_name in enumerate(model_names):
    model = models[model_name]
    before_piaa = img_at_PIAA(model, after_piaa=False)
    after_piaa = img_at_PIAA(model)
    # Use the same pre-PIAA peak for both curves to preserve redistribution.
    intensity_scale = float(onp.asarray(before_piaa).max())
    pupil_diameter = dlwfs.aperture_name_to_pupil_diameter(model_name)
    # Magnification scales the grid and pupil diameter equally, so cancels in r/D.
    diameter_over_D = model.tel.optics.diameter / pupil_diameter
    radii, before_profile = radial_intensity_profile(before_piaa, diameter_over_D)
    _, after_profile = radial_intensity_profile(after_piaa, diameter_over_D)

    profile_ax.plot(
        radii,
        before_profile / intensity_scale,
        color=f"C{i}",
        linestyle="--",
    )
    profile_ax.plot(
        radii,
        after_profile / intensity_scale,
        color=f"C{i}",
        linestyle="-",
    )

profile_ax.set_xlabel(r"$r/D$")
profile_ax.set_ylabel(r"$\langle I\rangle_\theta / I_{\mathrm{before,max}}$")
profile_ax.set_xlim(left=0)
profile_ax.set_ylim(bottom=0)
legend_handles = [
    Line2D([], [], color=f"C{i}", label=title) for i, title in enumerate(col_titles)
] + [
    Line2D([], [], color="gray", linestyle="--", label="Before"),
    Line2D([], [], color="gray", linestyle="-", label="After"),
]
profile_ax.legend(handles=legend_handles, ncol=2, fontsize=7, frameon=False)
profile_fig.savefig(
    os.path.join(os.path.dirname(fig_savepth), "different_apertures_radial_profiles.pdf"),
    bbox_inches="tight",
)

# %%
