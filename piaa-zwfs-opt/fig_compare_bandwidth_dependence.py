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
# Keep optical control flags static when tracing the ideal model under JIT.
ideal_model = ideal_model.set(
    ["tel.TransmissiveLayer.normalise", "tel.BasisLayer.as_phase"],
    [
        bool(ideal_model.tel.TransmissiveLayer.normalise),
        bool(ideal_model.tel.BasisLayer.as_phase),
    ],
)

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

gs0 = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1], hspace=0.4)

axs = {}

axs["loss"] = plt.subplot(gs0[0])
axs["mask"] = plt.subplot(gs0[1])


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


def phase_mask_profile(model, pupil_diameter):
    """Return azimuthally averaged thickness in micrometres versus radius in lambda/D."""
    tel = model._update_tel()
    focal_system = tel.optics.ThreePlaneSystem
    mask = focal_system.mask_at_focus
    if isinstance(mask, dlwfs.GlassDot):
        thickness = mask.generate_thickness()
    else:
        thickness = mask.eval_basis()

    thickness = onp.asarray(thickness, dtype=float)
    npixels = thickness.shape[0]
    coords = onp.arange(npixels) - (npixels - 1) / 2
    radius = onp.hypot(coords[:, None], coords[None, :])
    # One-pixel annuli, restricted to complete circles inside the focal grid.
    edges = onp.arange(npixels // 2 + 1)
    counts, _ = onp.histogram(radius, bins=edges)
    sums, _ = onp.histogram(radius, bins=edges, weights=thickness)
    # The focal grid is angular; D is the pupil diameter after magnification.
    output_diameter = pupil_diameter * tel.Magnifier.magnification
    lambda_over_D = float(tel.source.wavelengths.mean()) / output_diameter
    radii = (edges[:-1] + 0.5) * focal_system.focus_pixel_scale / lambda_over_D
    return radii, sums / counts * 1e6


for name, model in models.items():
    pupil_diameter = dlwfs.aperture_name_to_pupil_diameter(name.split("_")[0])
    radii, thickness = phase_mask_profile(model, pupil_diameter)
    if "no_optimization" in name:
        color = "C0"
        linestyle = "--" if "2.0" in name else "-"
    else:
        opt_wavels = model.tel.source.wavelengths
        opt_frac_bandwidth = (opt_wavels.max() - opt_wavels.min()) / opt_wavels.mean()
        color = colors[f"{float(opt_frac_bandwidth):.2f}"]
        linestyle = "--" if "no_PIAA" in name else "-"
    axs["mask"].plot(radii, thickness, color=color, linestyle=linestyle)

mask_legend = [
    Line2D([], [], color=colors["0.01"], label="Narrowband"),
    Line2D([], [], color=colors["0.20"], label="Broadband"),
    Line2D([], [], color="C0", label="1.06 ZWFS"),
    Line2D([], [], color="C0", linestyle="--", label="2.0 ZWFS"),
    Line2D([], [], color="gray", label="PIAA"),
    Line2D([], [], color="gray", linestyle="--", label="No PIAA"),
]
axs["mask"].set_xlabel(r"Radius [$\lambda/D$]")
axs["mask"].set_ylabel(r"Mask thickness [$\mu$m]")
axs["mask"].set_xlim(left=0, right=2.5)
axs["mask"].legend(handles=mask_legend, ncol=2, fontsize=6, frameon=False)
axs["mask"].grid()

plt.savefig(fig_savepth, bbox_inches="tight")

# %%
