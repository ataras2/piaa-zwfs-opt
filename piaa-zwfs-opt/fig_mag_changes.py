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

import scienceplots

plt.style.use(["science"])
plt.rcParams.update({"font.size": 8})

# %%

ModelCore = dlwfs.ModelCore


fig_savepth = os.path.join(
    "..",
    "results",
    "figures",
    "mag_change.pdf",
)

# %%
pth = os.path.join(
    "..",
    "results",
    "dl_optimizations",
    "mag_change",
)
# find all folders in pth
all_folders = [f for f in os.listdir(pth) if os.path.isdir(os.path.join(pth, f))]
all_folders
# %%
models = {f: dlwfs.load_model(os.path.join(pth, f))[0] for f in all_folders}

non_zwfs_key = "elt_Fourier_basis_7_terms_Z_band_with_PIAA_annular_mask_point_source_0.0_mag_10.0e_readnoise"
zwfs_key = "elt_Fourier_basis_7_terms_Z_band_no_optimization_2.0_loD_init_eval_diam"
models[zwfs_key] = models[zwfs_key].set("tel.source", models[non_zwfs_key].tel.source)

# add a high read noise case
readnoise = 10.0
models[f"{zwfs_key}_10.0e"] = models[zwfs_key].set("params.read_noise", readnoise)

# %%
# [m.loss("cov_trace_normed") for m in models.values()]

# %%
magnitude = 0.0
waveband = "Z"
exposure_time = 1e-3
system_throughput = 0.3

m = models[all_folders[1]]

aperture = m.tel.optics.TransmissiveLayer
wf_npixels = m.tel.optics.wf_npixels
sim_diameter = m.tel.optics.diameter

wavels = m.tel.source.wavelengths

photons_per_exposure = dlwfs.mag2flux(
    magnitude,
    dlwfs.zero_points[waveband],
    dlwfs.aperture_area(aperture.transmission, wf_npixels, sim_diameter),
    exposure_time,
    wavels.max() - wavels.min(),
)

photons_per_exposure *= system_throughput

m.tel.source.flux / photons_per_exposure

# %%

models_of_interest = [
    "_0.0_",
    "_10.0_",
    "_15.0_",
    "9.0_mag_10.0e",
    "0.0_mag_10.0e",
]

zwfs_models = [
    "2.0_loD_init_eval_diam",
    "2.0_loD_init_eval_diam_10.0e",
]

mods = {}
for val in models_of_interest:
    for k, m in models.items():
        if k.endswith(f"{val}mag") or k.endswith(f"{val}_readnoise"):
            mods[val.strip("_")] = m

for val in zwfs_models:
    for k, m in models.items():
        if k.endswith(val):
            mods[val] = m
# %%

eval_magnitudes = np.linspace(0, 16, 51)

import equinox as eqx


@eqx.filter_jit
def compute_losses(model, magnitude):
    aperture = model.tel.optics.TransmissiveLayer
    wf_npixels = model.tel.optics.wf_npixels
    sim_diameter = model.tel.optics.diameter

    wavels = model.tel.source.wavelengths
    photons_per_exposure = dlwfs.mag2flux(
        magnitude,
        dlwfs.zero_points[waveband],
        dlwfs.aperture_area(aperture.transmission, wf_npixels, sim_diameter),
        exposure_time,
        wavels.max() - wavels.min(),
    )

    model = model.set("tel.source.flux", photons_per_exposure)

    return model.loss("cov_trace_normed")


mods["ideal"] = dlwfs.generate_idealwfs_optics(mods["0.0"])

import jax

losses = {}
for mod_key, mod in mods.items():
    losses[mod_key] = jax.vmap(compute_losses, in_axes=(None, 0))(mod, eval_magnitudes)

# %%

plt.figure(figsize=(3.33, 2.5), dpi=300)
for mod_key, loss_vals in losses.items():
    if "loD_init_eval_diam" in mod_key:
        if "10.0e" in mod_key:
            ls = "--"
            label = f"2.0 ZWFS, $\sigma_R = 10e^{{-}}$"
        else:
            ls = "-"
            label = f"2.0 ZWFS"

    else:
        if mod_key == "10.0":
            ls = "--"
        else:
            ls = "-"

        if mod_key.endswith("10.0e"):
            label = (
                f"{float(mod_key.split('_')[0]):.0f} mag opt. ($\sigma_R = 10e^{{-}}$)"
            )
        else:
            label = f"{float(mod_key):.0f} mag opt."

    plt.plot(
        eval_magnitudes,
        loss_vals * 1e3,
        label=label,
        linestyle=ls,
    )
plt.legend()

plt.xlabel("Source Z Magnitude [mag]")
plt.ylabel("Loss [$10^{-3}$/ph.]")
plt.xlim(eval_magnitudes.min(), eval_magnitudes.max())

plt.ylim(0, losses["0.0"].max() * 1e3 * 1.5)


# %%
m = mods["0.0"]


def phase_mask_thickness(model):
    thicc = model.tel.optics.ThreePlaneSystem.mask_at_focus.eval_basis()
    return thicc


# thicknesses = {k: phase_mask_thickness(m) * 1e6 for k, m in mods.items()}
thicknesses = {}
for k, m in mods.items():
    if k in ["0.0", "15.0"]:
        thicknesses[k] = phase_mask_thickness(m) * 1e6


clims = [
    min([onp.min(t) for t in thicknesses.values()]),
    max([onp.max(t) for t in thicknesses.values()]),
]

plt.figure(figsize=(3.33, 2.5))
plt.imshow(
    onp.array(thicknesses["0.0"]),
    cmap="viridis",
    vmin=clims[0],
    vmax=clims[1],
)

plt.figure(figsize=(3.33, 2.5))
plt.imshow(
    onp.array(thicknesses["15.0"]),
    cmap="viridis",
    vmin=clims[0],
    vmax=clims[1],
)
# %%
# gridspec to put all into one figure

# lazy reorder
losses["9.0_mag_10.0e"] = losses.pop("9.0_mag_10.0e")
losses["2.0_loD_init_eval_diam"] = losses.pop("2.0_loD_init_eval_diam")
losses["2.0_loD_init_eval_diam_10.0e"] = losses.pop("2.0_loD_init_eval_diam_10.0e")

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(3.4, 4.3), dpi=300)
gs0 = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=-0.05)
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[1], wspace=0.1)

axs = {}
axs["loss"] = plt.subplot(gs0[0])
axs["img_bright"] = plt.subplot(gs_bottom[0])
axs["img_faint"] = plt.subplot(gs_bottom[1])
axs["img_diff"] = plt.subplot(gs_bottom[2])

lines = []
# plot losses
for mod_key, loss_vals in losses.items():
    # if mod_key == "10.0":

    kwargs = dict()
    if "ideal" in mod_key:
        ls = "-."
        label = "Ideal WFS"

        kwargs["color"] = "k"

    elif "loD_init_eval_diam" in mod_key:
        if "10.0e" in mod_key:
            ls = "--"
            label = f"2.0 ZWFS ($\sigma_R$=$10e^{{-}}$)"
        else:
            ls = "-"
            label = f"2.0 ZWFS"
            # loss_vals = loss_vals/loss_vals[0]*6.0e-3

        kwargs["color"] = "C6"

    else:

        if mod_key.endswith("10.0e"):
            label = (
                f"{float(mod_key.split('_')[0]):.0f} mag opt. ($\sigma_R$=$10e^{{-}}$)"
            )
            ls = "--"
        else:
            label = f"{float(mod_key):.0f} mag opt."
            # ls = "-"
            if mod_key == "10.0":
                ls = ":"
                continue
            else:
                ls = "-"

    kwargs["label"] = label
    kwargs["linestyle"] = ls
    line = axs["loss"].plot(eval_magnitudes, loss_vals * 1e3, **kwargs)
    lines.append(line[0])

loss_bnd = mods["0.0"].loss_bound_idealwfs("cov_trace_normed")
# add in a hline for the optimal
# axs["loss"].axhline(
#     loss_bnd * 1e3,
#     color="k",
#     ls="-.",
#     label="Ideal WFS",
# )

axs["loss"].set_xlabel("$m_Z$ [mag]", labelpad=0)
axs["loss"].set_ylabel("Loss [$10^{-3}\\mu$m$^2$/ph.]", labelpad=0)
axs["loss"].set_xlim(eval_magnitudes.min(), eval_magnitudes.max())
axs["loss"].set_ylim(0.9 * loss_bnd * 1e3, losses["0.0"].max() * 1e3 * 1.6)

handles, labels = axs["loss"].get_legend_handles_labels()
# desired_order = [
#     "2.0 ZWFS",
#     "2.0 ZWFS ($\\sigma_R$=$10e^{-}$)",
#     "Ideal WFS",
#     "0 mag opt.",
#     # "10 mag opt.",
#     "15 mag opt.",
#     "0 mag opt. ($\\sigma_R$=$10e^{-}$)",
#     "9 mag opt. ($\\sigma_R$=$10e^{-}$)",
# ]
# desired_order = [
#     "0 mag opt.",
#     "0 mag opt. ($\\sigma_R$=$10e^{-}$)",
#     "Ideal WFS",
#     "15 mag opt.",
#     "9 mag opt. ($\\sigma_R$=$10e^{-}$)",
#     "2.0 ZWFS",
#     "2.0 ZWFS ($\\sigma_R$=$10e^{-}$)",
# ]
import matplotlib.patches as mpatches

dummy_handle = mpatches.Patch(color="none")

desired_order = [
    "0 mag opt.",
    "15 mag opt.",
    "0 mag opt. ($\\sigma_R$=$10e^{-}$)",
    "9 mag opt. ($\\sigma_R$=$10e^{-}$)",
    "2.0 ZWFS",
    "2.0 ZWFS ($\\sigma_R$=$10e^{-}$)",
    "Ideal WFS",
]

ordered_handles = [handles[labels.index(label)] for label in desired_order]
grid_handles = [
    ordered_handles[0],
    ordered_handles[1],
    dummy_handle,
    ordered_handles[2],  # Row 1
    ordered_handles[3],
    dummy_handle,
    ordered_handles[4],
    ordered_handles[5],  # Row 2
    ordered_handles[6],  # Row 3 (Bottom-left & mid are empty)
]
labels = [
    "0 mag opt.",
    "15 mag opt.",
    "",
    "0 mag opt. ($\\sigma_R$=$10e^{-}$)",
    "9 mag opt. ($\\sigma_R$=$10e^{-}$)",
    "",
    "2.0 ZWFS",
    "2.0 ZWFS ($\\sigma_R$=$10e^{-}$)",
    "Ideal WFS",
]

axs["loss"].legend(
    handles=grid_handles,
    labels=labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.3),
    ncol=3,
    fontsize=6,
    columnspacing=0.8,
    handletextpad=0.5,
)

# bottom row: normalized images and their difference
img0 = mods["0.0"].img()
img16 = mods["15.0"].img()

# normalize to sum = 1
img0 = img0 / np.sum(img0) * 1e3
img16 = img16 / np.sum(img16) * 1e3

# convert to numpy arrays for plotting
img0_np = onp.array(img0)
img16_np = onp.array(img16)

# common vmin/vmax for first two images
vmin_img = 0.0
vmax_img = float(onp.max([onp.max(img0_np), onp.max(img16_np)]))

im0 = axs["img_bright"].imshow(img0_np, cmap="inferno", vmin=vmin_img, vmax=vmax_img)
im1 = axs["img_faint"].imshow(img16_np, cmap="inferno", vmin=vmin_img, vmax=vmax_img)

# difference image with symmetric limits and seismic colormap
diff_np = img0_np - img16_np
clim_diff = float(onp.max(onp.abs(diff_np)))
imdiff = axs["img_diff"].imshow(
    diff_np, cmap="seismic", vmin=-clim_diff, vmax=clim_diff
)

# turn off ticks for all image axes
for key in ["img_bright", "img_faint", "img_diff"]:
    axs[key].set_xticks([])
    axs[key].set_yticks([])

# axis spline color should be the legend colors
axs["img_bright"].spines[:].set_edgecolor("C0")
axs["img_faint"].spines[:].set_edgecolor("C2")
# and make them thicker
axs["img_bright"].spines[:].set_linewidth(1.5)
axs["img_faint"].spines[:].set_linewidth(1.5)

# axs["img_bright"].set_title("$I_0$ (0 mag)", pad=4)
# axs["img_faint"].set_title("$I_{0}$ (16 mag)", pad=4)
# axs["img_diff"].set_title("Difference", pad=4)

# top left corner text labels
txt_kwargs = dict(fontsize=6, ha="left")
axs["img_bright"].text(
    0.05,
    0.95,
    "Bright",
    **txt_kwargs,
    va="top",
    color="white",
)
axs["img_faint"].text(
    0.05,
    0.95,
    "Faint",
    **txt_kwargs,
    color="white",
    va="top",
)

# bottom left says sigma R
axs["img_bright"].text(
    0.05,
    0.02,
    f"$\sigma_R = 0.5 e^-$",
    color="white",
    **txt_kwargs,
    transform=axs["img_bright"].transAxes,
    va="bottom",
)

axs["img_diff"].text(
    0.05,
    0.95,
    "Bright-Faint",
    color="black",
    **txt_kwargs,
    va="top",
)

# common colorbar for first two images (below the axes)
# fig.colorbar(
#     im0,
#     ax=[axs["img_bright"], axs["img_faint"]],
#     orientation="horizontal",
#     fraction=0.046,
#     pad=0.15,
#     location="bottom",
# )

fig = plt.gcf()
left_pos = axs["img_bright"].get_position().x0
right_pos = axs["img_faint"].get_position().x1
bottom_pos = axs["img_bright"].get_position().y0 - 0.03
top_pos = axs["img_bright"].get_position().y0 - 0.01
width = right_pos - left_pos
height = top_pos - bottom_pos

cbar_ax = plt.gcf().add_axes([left_pos, bottom_pos, width, height])
cbar = plt.colorbar(im0, cax=cbar_ax, orientation="horizontal")
cbar.set_label(r"$I_0$ [$10^{-3}$/ph.]")


left_pos = axs["img_diff"].get_position().x0
right_pos = axs["img_diff"].get_position().x1
bottom_pos = axs["img_diff"].get_position().y0 - 0.03
top_pos = axs["img_diff"].get_position().y0 - 0.01
width = right_pos - left_pos
height = top_pos - bottom_pos

cbar_ax = plt.gcf().add_axes([left_pos, bottom_pos, width, height])
cbar = plt.colorbar(imdiff, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Diff. [$10^{-3}$/ph.]")


# separate colorbar for the difference image (below the axis)
# fig.colorbar(
#     imdiff,
#     ax=axs["img_diff"],
#     orientation="horizontal",
#     fraction=0.046,
#     pad=0.15,
#     location="bottom",
# )

plt.savefig(fig_savepth, bbox_inches="tight")
plt.savefig(
    fig_savepth.replace(".pdf", ".svg"),
    bbox_inches="tight",
    transparent=True,
)
# %%

fig = plt.figure(figsize=(3.4, 2.3), dpi=300)
gs0 = gridspec.GridSpec(1, 1, height_ratios=[1.0], hspace=-0.05)

axs = {}
axs["loss"] = plt.subplot(gs0[0])

lines = []
# plot losses
for mod_key, loss_vals in losses.items():
    # if mod_key == "10.0":

    kwargs = dict()
    if "ideal" in mod_key:
        ls = "-."
        label = "Ideal WFS"

        kwargs["color"] = "k"

    elif "loD_init_eval_diam" in mod_key:
        if "10.0e" in mod_key:
            ls = "--"
            label = f"2.0 ZWFS ($\sigma_R$=$10e^{{-}}$)"
        else:
            ls = "-"
            label = f"2.0 ZWFS"
            # loss_vals = loss_vals/loss_vals[0]*6.0e-3

        kwargs["color"] = "C6"

    else:

        if mod_key.endswith("10.0e"):
            label = (
                f"{float(mod_key.split('_')[0]):.0f} mag opt. ($\sigma_R$=$10e^{{-}}$)"
            )
            ls = "--"
        else:
            label = f"{float(mod_key):.0f} mag opt."
            # ls = "-"
            if mod_key == "10.0":
                ls = ":"
                continue
            else:
                ls = "-"

    kwargs["label"] = label
    kwargs["linestyle"] = ls
    line = axs["loss"].plot(eval_magnitudes, loss_vals / loss_vals[0], **kwargs)
    lines.append(line[0])

loss_bnd = mods["0.0"].loss_bound_idealwfs("cov_trace_normed")
# add in a hline for the optimal
# axs["loss"].axhline(
#     loss_bnd * 1e3,
#     color="k",
#     ls="-.",
#     label="Ideal WFS",
# )

axs["loss"].set_xlabel("$m_Z$ [mag]", labelpad=0)
axs["loss"].set_ylabel("Loss [$10^{-3}\\mu$m$^2$/ph.]", labelpad=0)
axs["loss"].set_xlim(eval_magnitudes.min(), eval_magnitudes.max())
# axs["loss"].set_ylim(0.9 * loss_bnd * 1e3, losses["0.0"].max() * 1e3 * 1.6)
axs["loss"].set_ylim(0.8, 2.5)

handles, labels = axs["loss"].get_legend_handles_labels()
# desired_order = [
#     "2.0 ZWFS",
#     "2.0 ZWFS ($\\sigma_R$=$10e^{-}$)",
#     "Ideal WFS",
#     "0 mag opt.",
#     # "10 mag opt.",
#     "15 mag opt.",
#     "0 mag opt. ($\\sigma_R$=$10e^{-}$)",
#     "9 mag opt. ($\\sigma_R$=$10e^{-}$)",
# ]
# desired_order = [
#     "0 mag opt.",
#     "0 mag opt. ($\\sigma_R$=$10e^{-}$)",
#     "Ideal WFS",
#     "15 mag opt.",
#     "9 mag opt. ($\\sigma_R$=$10e^{-}$)",
#     "2.0 ZWFS",
#     "2.0 ZWFS ($\\sigma_R$=$10e^{-}$)",
# ]
import matplotlib.patches as mpatches

dummy_handle = mpatches.Patch(color="none")

desired_order = [
    "0 mag opt.",
    "15 mag opt.",
    "0 mag opt. ($\\sigma_R$=$10e^{-}$)",
    "9 mag opt. ($\\sigma_R$=$10e^{-}$)",
    "2.0 ZWFS",
    "2.0 ZWFS ($\\sigma_R$=$10e^{-}$)",
    "Ideal WFS",
]

ordered_handles = [handles[labels.index(label)] for label in desired_order]
grid_handles = [
    ordered_handles[0],
    ordered_handles[1],
    dummy_handle,
    ordered_handles[2],  # Row 1
    ordered_handles[3],
    dummy_handle,
    ordered_handles[4],
    ordered_handles[5],  # Row 2
    ordered_handles[6],  # Row 3 (Bottom-left & mid are empty)
]
labels = [
    "0 mag opt.",
    "15 mag opt.",
    "",
    "0 mag opt. ($\\sigma_R$=$10e^{-}$)",
    "9 mag opt. ($\\sigma_R$=$10e^{-}$)",
    "",
    "2.0 ZWFS",
    "2.0 ZWFS ($\\sigma_R$=$10e^{-}$)",
    "Ideal WFS",
]

axs["loss"].legend(
    handles=grid_handles,
    labels=labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.3),
    ncol=3,
    fontsize=6,
    columnspacing=0.8,
    handletextpad=0.5,
)


fig_name = fig_savepth.replace(".pdf", "_normalized.pdf")
plt.savefig(fig_name, bbox_inches="tight")

# %%
plt.figure()
m = mods["0.0"]
plt.imshow(m.img())
plt.figure()
m = mods["16.0"]
plt.imshow(m.img())

plt.figure()
img1 = mods["0.0"].img()
img2 = mods["16.0"].img()

# norm each
img1 /= np.sum(img1)
img2 /= np.sum(img2)

clim_diff = np.max(np.abs(img1 - img2))
plt.imshow(img1 - img2, vmin=-clim_diff, vmax=clim_diff, cmap="bwr")
plt.colorbar()


# %%
def piaa_profiles(model):
    piaa_set = model.tel.optics.PIAAset_0

    rho_max = piaa_set.piaa_lens_1.asphere_profile.rho_max
    rhos = np.linspace(0, rho_max, 100)

    z1 = piaa_set.piaa_lens_1.asphere_profile.sag(rhos)
    z2 = piaa_set.piaa_lens_2.asphere_profile.sag(rhos)

    return rhos, z1, z2


rhos, z1_0, z2_0 = piaa_profiles(mods["0.0"])
rhos, z1_16, z2_16 = piaa_profiles(mods["16.0"])

plt.figure(figsize=(3.33, 2.5), dpi=300)
plt.plot(rhos * 1e3, z1_0 * 1e6, ls="-", color="C0")
plt.plot(rhos * 1e3, z2_0 * 1e6, ls="-", color="C0")
plt.plot(rhos * 1e3, z1_16 * 1e6, ls="--", color="C2")
plt.plot(rhos * 1e3, z2_16 * 1e6, ls="--", color="C2")
# dummy plots for legend
plt.plot([], [], ls="-", color="k", label="0 mag opt.")
plt.plot([], [], ls="--", color="k", label="16 mag opt.")

# text to label which lens is which, at around 3mm radius value
idx = np.argmin(np.abs(rhos - 3e-3))
plt.text(rhos[idx] * 1e3 + 0.1, z1_0[idx] * 1e6, "Lens 1", color="k", va="bottom")
plt.text(rhos[idx] * 1e3 + 0.1, z2_0[idx] * 1e6, "Lens 2", color="k", va="top")

plt.xlabel("Radius [mm]")
plt.ylabel("Sag [$\mu$m]")
plt.legend()
plt.xlim(0, rhos.max() * 1e3)

# %%
