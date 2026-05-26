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
    "bandwidth_dep.pdf",
)

# %%
pth = os.path.join(
    "..",
    "results",
    "dl_optimizations",
    "bandwidth_change",
)
# find all folders in pth
all_folders = [f for f in os.listdir(pth) if os.path.isdir(os.path.join(pth, f))]
all_folders
# %%
models = {f: dlwfs.load_model(os.path.join(pth, f))[0] for f in all_folders}

# %%
import equinox as eqx

centre_wavel = models[all_folders[0]].tel.source.wavelengths.mean()

test_frac_bandwidths = np.linspace(1e-3, 0.3, 20)
loss_values = []


@eqx.filter_jit
def loss_give_wavels(model, test_wavelengths):
    test_model = model.set("tel.source.wavelengths", test_wavelengths)
    loss = test_model.loss("cov_trace_normed")
    return loss


# %%

for model in models.values():
    print("Processing model...")
    model_loss_values = []
    for frac_bw in test_frac_bandwidths:
        test_wavelengths = np.linspace(
            centre_wavel * (1 - frac_bw / 2),
            centre_wavel * (1 + frac_bw / 2),
            len(model.tel.source.wavelengths),
        )
        loss = loss_give_wavels(model, test_wavelengths)
        model_loss_values.append(loss)

    loss_values.append(model_loss_values)
loss_values = np.array(loss_values)

# %%
ideal_model = dlwfs.generate_idealwfs_optics(models[all_folders[0]])

ideal_model_loss_values = []
# only do the endpoints for speed
for frac_bw in [test_frac_bandwidths[0], test_frac_bandwidths[-1]]:
    test_wavelengths = np.linspace(
        centre_wavel * (1 - frac_bw / 2),
        centre_wavel * (1 + frac_bw / 2),
        len(ideal_model.tel.source.wavelengths),
    )
    loss = loss_give_wavels(ideal_model, test_wavelengths)
    ideal_model_loss_values.append(loss)

# %%
plt.figure()

colors = {}

opt_frac_bandwidths = []
for i, model in enumerate(models.values()):
    if "no_optimization" not in list(models.keys())[i]:
        opt_wavels = model.tel.source.wavelengths
        opt_frac_bandwidth = (opt_wavels.max() - opt_wavels.min()) / opt_wavels.mean()
        opt_frac_bandwidths.append(float(opt_frac_bandwidth))

# get the unique values, rounded to nearest percent
opt_frac_bandwidths = list(set([round(fb * 100) / 100 for fb in opt_frac_bandwidths]))

# each color corresponds to an optimized bandwidth, ascending order
for i, fb in enumerate(sorted(opt_frac_bandwidths)):
    colors[f"{fb:.2f}"] = f"C{i+1}"

# %%
plt.figure(figsize=(3.4, 3), dpi=300)

# idx_02_bw_crossing = np.argmin(np.abs(test_frac_bandwidths - 0.2))
idx_02_bw_crossing = -1

for i, (name, model) in enumerate(models.items()):
    pkwargs = {}
    if "no_PIAA" in name:
        pkwargs["linestyle"] = "--"

    opt_wavels = model.tel.source.wavelengths
    opt_frac_bandwidth = (opt_wavels.max() - opt_wavels.min()) / opt_wavels.mean()

    color = colors.get(f"{float(opt_frac_bandwidth):.2f}", "C0")

    if "no_optimization" in name:
        color = "C0"
        if "2.0" in name:
            pkwargs["linestyle"] = "--"

    pkwargs["color"] = color

    plt.plot(
        test_frac_bandwidths,
        loss_values[i],
        **pkwargs,
    )

    if "no_optimization" in name:
        if "2.0" in name:
            plt.plot([], [], c="C0", label="2.0 ZWFS", linestyle="--")
        else:
            plt.plot([], [], c="C0", label="1.06 ZWFS")

    # just outside the plot on the right, at the yvalue of 0.2 bandwidth
    # text of the fraction suboptimality as a percentage
    frac_subopts = loss_values[i][idx_02_bw_crossing] / loss_values[i][0] - 1

    if color == "C0":
        yoffset = 0.0
    # elif color == "C1":
    #     yoffset = +0.001
    # else:
    #     yoffset = -0.001
    elif "no_PIAA" in name and "9.0nm" in name:
        yoffset = +0.0012
    elif "no_PIAA" in name and "180.0nm" in name:
        yoffset = +0.0006
    elif "with_PIAA" in name and "9.0nm" in name:
        yoffset = +0.00
    elif "with_PIAA" in name and "180.0nm" in name:
        yoffset = -0.0006

    plt.text(
        test_frac_bandwidths[-1] + 0.005,
        loss_values[i][-1] + yoffset,
        f"{frac_subopts*100:.0f}\%",
        color=color,
        va="center",
    )


# legend dummy entries for colors
for fb_str, color in colors.items():
    try:
        float(fb_str)
    except:
        continue
    # plt.plot(
    #     [],
    #     [],
    #     c=color,
    #     label=f"Opt. BW: {float(fb_str)*100:.0f}%",
    # )
    plt.axvline(
        float(fb_str),
        c=color,
        ls=":",
        alpha=0.8,
        linewidth=2.0,
    )

# grey lines legend for no PIAA vs PIAA
plt.plot([], [], c="grey", ls="-", label="PIAA")
plt.plot([], [], c="grey", ls="--", label="No PIAA")

# ideal model lines
plt.plot(
    [test_frac_bandwidths[0], test_frac_bandwidths[-1]],
    ideal_model_loss_values,
    c="k",
    # ls=":",
    label="Ideal WFS",
)
plt.xlabel("Fractional bandwidth")
plt.ylabel("Average variance [/frame/ph/mode]")

# ligth box around legend
plt.legend(frameon=True, loc=(0.05, 0.4))
plt.grid()
plt.xlim(test_frac_bandwidths[0], test_frac_bandwidths[-1])
plt.tight_layout()


# plt.savefig(fig_savepth, bbox_inches="tight")

# %%
# gridspec
import matplotlib.gridspec as gridspec

plt.figure(figsize=(3.4, 5.5), dpi=300)

gs0 = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1], hspace=0.2)
gs01 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[1], wspace=-0.2)

axs = {}

axs["loss"] = plt.subplot(gs0[0])
axs["no_piaa_narrow"] = plt.subplot(gs01[0, 0])
axs["no_piaa_broad"] = plt.subplot(gs01[0, 1])
axs["piaa_narrow"] = plt.subplot(gs01[1, 0])
axs["piaa_broad"] = plt.subplot(gs01[1, 1])


# loss plot
# idx_02_bw_crossing = np.argmin(np.abs(test_frac_bandwidths - 0.2))
idx_02_bw_crossing = -1

for i, (name, model) in enumerate(models.items()):
    pkwargs = {}
    if "no_PIAA" in name:
        pkwargs["linestyle"] = "--"

    opt_wavels = model.tel.source.wavelengths
    opt_frac_bandwidth = (opt_wavels.max() - opt_wavels.min()) / opt_wavels.mean()

    color = colors.get(f"{float(opt_frac_bandwidth):.2f}", "C0")

    if "no_optimization" in name:
        color = "C0"
        if "2.0" in name:
            pkwargs["linestyle"] = "--"

    pkwargs["color"] = color

    axs["loss"].plot(
        test_frac_bandwidths,
        loss_values[i] * 1e3,
        **pkwargs,
    )

    if "no_optimization" in name:
        if "2.0" in name:
            axs["loss"].plot([], [], c="C0", label="2.0 ZWFS", linestyle="--")
        else:
            axs["loss"].plot([], [], c="C0", label="1.06 ZWFS")

    # just outside the plot on the right, at the yvalue of 0.2 bandwidth
    # text of the fraction suboptimality as a percentage
    frac_subopts = loss_values[i][idx_02_bw_crossing] / loss_values[i][0] - 1

    offset_scale = 0.0008

    if color == "C0":
        yoffset = 0.0
    elif "no_PIAA" in name and "9.0nm" in name:
        yoffset = +2 * offset_scale
    elif "no_PIAA" in name and "180.0nm" in name:
        yoffset = +1 * offset_scale
    elif "with_PIAA" in name and "9.0nm" in name:
        yoffset = +0.00
    elif "with_PIAA" in name and "180.0nm" in name:
        yoffset = -1 * offset_scale

    print(f"{name}: frac_subopts={frac_subopts*100:.2f}%, yoffset={yoffset}")

    # axs["loss"].text(
    #     test_frac_bandwidths[-1] + 0.005,
    #     loss_values[i][-1] + yoffset,
    #     f"{frac_subopts*100:.0f}\%",
    #     color=color,
    #     va="center",
    # )


# legend dummy entries for colors
for fb_str, color in colors.items():
    try:
        float(fb_str)
    except:
        continue
    # plt.plot(
    #     [],
    #     [],
    #     c=color,
    #     label=f"Opt. BW: {float(fb_str)*100:.0f}%",
    # )
    axs["loss"].axvline(
        float(fb_str),
        c=color,
        ls=":",
        alpha=0.5,
        linewidth=2.0,
    )

    if float(fb_str) == 0.01:
        text = "Narrowband\noptimised"
        ha = "left"
        ha_offset = +0.005
    elif float(fb_str) == 0.20:
        text = "Broadband\noptimised"
        ha = "right"
        ha_offset = -0.005

    axs["loss"].text(
        float(fb_str) + ha_offset,
        axs["loss"].get_ylim()[1]
        - 0.03 * (axs["loss"].get_ylim()[1] - axs["loss"].get_ylim()[0]),
        text,
        color=color,
        ha=ha,
        va="top",
        fontsize=7,
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8
        ),
    )

# grey lines legend for no PIAA vs PIAA
axs["loss"].plot([], [], c="grey", ls="-", label="PIAA")
axs["loss"].plot([], [], c="grey", ls="--", label="No PIAA")

# ideal model lines
axs["loss"].plot(
    [test_frac_bandwidths[0], test_frac_bandwidths[-1]],
    np.array(ideal_model_loss_values) * 1e3,
    c="k",
    # ls=":",
    label="Ideal WFS",
)
axs["loss"].set_xlabel("Fractional bandwidth")
axs["loss"].set_ylabel("Loss [$10^{-3}\,\mu$m$^2$/ph.]")

# ligth box around legend, no alpha on the box
axs["loss"].legend(frameon=True, loc=(0.55, 0.38), framealpha=1.0)
axs["loss"].grid()
axs["loss"].set_xlim(test_frac_bandwidths[0], test_frac_bandwidths[-1])


def phase_mask_thickness(model):
    thicc = model.tel.optics.ThreePlaneSystem.mask_at_focus.eval_basis()
    return thicc


mask_names = {
    "no_piaa_narrow": "magellan_Fourier_basis_7_terms_900nm_9.0nm_bw_band_no_PIAA_annular_mask_point_source",
    "no_piaa_broad": "magellan_Fourier_basis_7_terms_900nm_180.0nm_bw_band_no_PIAA_annular_mask_point_source",
    "piaa_narrow": "magellan_Fourier_basis_7_terms_900nm_9.0nm_bw_band_with_PIAA_annular_mask_point_source",
    "piaa_broad": "magellan_Fourier_basis_7_terms_900nm_180.0nm_bw_band_with_PIAA_annular_mask_point_source",
}

thickness_max = max(
    [phase_mask_thickness(models[mask_name]).max() for mask_name in mask_names.values()]
)
thickness_min = min(
    [phase_mask_thickness(models[mask_name]).min() for mask_name in mask_names.values()]
)


for ax_key, mask_name in mask_names.items():
    model = models[mask_name]
    ax = axs[ax_key]
    im = ax.imshow(
        phase_mask_thickness(model) * 1e6,
        interpolation="none",
        vmin=thickness_min * 1e6,
        vmax=thickness_max * 1e6,
        cmap="plasma",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # axis border should match the loss plot
    # if no piaa, dashed border
    # color C1 if narrow, C2 if broad
    if "no_PIAA" in mask_name:
        ls = "--"
    else:
        ls = "-"

    if "9.0nm" in mask_name:
        color = colors["0.01"]
    else:
        color = colors["0.20"]
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.0)
        spine.set_linestyle(ls)

# common cbar on the right of all thickness plots
cbar_ax = plt.gcf().add_axes([0.87, 0.13, 0.02, 0.3])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label(r"Phase mask thickness [$\mu$m]")


plt.savefig(fig_savepth, bbox_inches="tight")

# %%

mask_of_interest = "magellan_Fourier_basis_7_terms_900nm_180.0nm_bw_band_with_PIAA_annular_mask_point_source"
# mask_of_interest = "magellan_Fourier_basis_7_terms_900nm_9.0nm_bw_band_with_PIAA_annular_mask_point_source"
# mask_of_interest = "magellan_Fourier_basis_7_terms_900nm_9.0nm_bw_band_no_PIAA_annular_mask_point_source"

model = models[mask_of_interest]


plt.figure()
ax = plt.gca()
ax.imshow(phase_mask_thickness(model), interpolation="none")
# rmove axis ticks
ax.set_xticks([])
ax.set_yticks([])

# %%
