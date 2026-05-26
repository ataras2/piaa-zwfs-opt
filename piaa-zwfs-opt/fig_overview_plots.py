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
import cmasher as cmr

import scienceplots

plt.style.use(["science"])
plt.rcParams.update({"font.size": 8})

# %%
ModelCore = dlwfs.ModelCore

fig_pth = os.path.join(
    "..",
    "results",
    "figures",
    "overview_subplots",
)

# %%
pth = os.path.join(
    "..",
    "results",
    "dl_optimizations",
    "pre_det_change_opts",
    "bandwidth_change",
    "magellan_Fourier_basis_7_terms_900nm_9.0nm_bw_band_with_PIAA_annular_mask_point_source",
)

model, _ = dlwfs.load_model(pth)

# %%
plt.imshow(model.img())
plt.xticks([])
plt.yticks([])

# %%
n_modes = 9
basis = model.tel.optics.BasisLayer.basis[:n_modes]

clims = np.max(np.abs(basis))

fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio
for i, b in enumerate(basis):
    # subplots, no hspace or vspace
    plt.subplot(3, 3, i + 1)
    plt.imshow(b, cmap="bwr", vmin=-clims, vmax=clims)
    # ticks off
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=0, hspace=0)

# %%
around_params = {
    "phase_coefficients": np.zeros(len(model.params["phase_coefficients"])),
}

J = model.compute_J(around_params)["phase_coefficients"]
# %%
clims = np.max(np.abs(J))
fig = plt.figure(figsize=(8, 8))  # Notice the equal aspect ratio
for i, b in enumerate(basis):
    # subplots, no hspace or vspace
    plt.subplot(3, 3, i + 1)
    plt.imshow(J[..., i], cmap="bwr", vmin=-clims, vmax=clims)
    # ticks off
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace=0, hspace=0)

# %%

FI = model.compute_FI("phase_coefficients")

FI = FI[:n_modes, :n_modes]
clims = np.max(np.abs(FI))


plt.imshow(FI, vmin=-clims, vmax=clims, cmap="bwr")
# remove ticks
plt.xticks([])
plt.yticks([])

plt.savefig(os.path.join(fig_pth, "fisher_information_matrix.svg"), bbox_inches="tight")

# %%


def piaa_profiles(model):
    piaa_set = model.tel.optics.PIAAset_0

    rho_max = piaa_set.piaa_lens_1.asphere_profile.rho_max
    rhos = np.linspace(0, rho_max, 100)

    z1 = piaa_set.piaa_lens_1.asphere_profile.sag(rhos)
    z2 = piaa_set.piaa_lens_2.asphere_profile.sag(rhos)

    return rhos, z1, z2


rhos, z1, z2 = piaa_profiles(model)

plt.figure(figsize=(1.5, 1))

plt.plot(rhos * 1e3, z1 * 1e6, label="PIAA Lens 1")
plt.plot(rhos * 1e3, z2 * 1e6, label="PIAA Lens 2")
plt.xlabel("Radius (mm)", labelpad=0)
plt.ylabel("Sag ($\mu$m)", labelpad=0)

plt.xlim(0, rhos.max() * 1e3)

plt.savefig(os.path.join(fig_pth, "piaa_lens_profiles.svg"), bbox_inches="tight")

# %%


def phase_mask_thickness(model):
    thicc = model.tel.optics.ThreePlaneSystem.mask_at_focus.eval_basis()
    return thicc


plt.figure(figsize=(2, 2))

thicc = phase_mask_thickness(model)
plt.imshow(thicc * 1e6, cmap="plasma", interpolation="none")
plt.xticks([])
plt.yticks([])

total_width = 5  # loD

loD_to_pix = thicc.shape[0] / total_width

# circle at the center with radius 1 loD
d = 2.5
plt.gca().add_artist(
    plt.Circle(
        (thicc.shape[0] / 2, thicc.shape[1] / 2),
        radius=loD_to_pix * d / 2,
        color="w",
        fill=False,
        lw=1,
    )
)
# text label at degrees
plt.text(
    thicc.shape[0] / 2 + loD_to_pix * (d * np.sqrt(2) / 4 - 0.1),
    thicc.shape[1] / 2 + loD_to_pix * (d * np.sqrt(2) / 4 + 0.5),
    f"{d} $\\lambda/D$",
    color="w",
    va="center",
    fontdict={"size": 10},
)

# top left corner mini title
plt.text(
    0.01,
    0.95,
    "Phase mask thickness",
    color="w",
    va="top",
    ha="left",
    transform=plt.gca().transAxes,
    fontdict={"size": 10},
)

plt.savefig(os.path.join(fig_pth, "phase_mask_thickness.svg"), bbox_inches="tight")

# %%
