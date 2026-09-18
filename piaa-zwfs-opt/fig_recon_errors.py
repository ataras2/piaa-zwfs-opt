# %%
import scienceplots
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import os
import dlWFSoptimize as dlwfs
import dLux.utils as dlu

import asdf
import jax
import jax.numpy as jnp
import numpy as np

plt.style.use(["science"])
plt.rcParams.update({"font.size": 10})


def rms_opd(opd, mask):
    # return jnp.sqrt(jnp.mean((opd[mask]) ** 2))
    # return jnp.sqrt(jnp.nanmean(jnp.where(mask, opd, jnp.nan)) ** 2)
    masked_opd = jnp.where(mask, opd, 0)
    n_valid = jnp.sum(mask)
    return jnp.sqrt(jnp.sum(masked_opd**2) / n_valid)


# %%
n_terms_errors = 45


PIAA_model = (
    "magellan_Fourier_basis_7_terms_Z_band_with_PIAA_annular_mask_point_source_0.0_mag"
)
exponent = -5 / 3

save_pth = os.path.join(
    "..",
    "results",
    "dl_optimizations",
    "ablation_60",
    "recon_errors_asdf",
    f"{n_terms_errors}_terms_" + PIAA_model + "_recon_errors.asdf",
    # f"{n_terms_errors}_terms_"
    # + PIAA_model
    # + f"_recon_errors_{np.round(exponent, 2)}.asdf",
)
# %%

recon_errors_file = asdf.open(save_pth)

rms_ins = recon_errors_file["rms_ins"]
recon_errors = recon_errors_file["recon_errors"]


# %%


fig = plt.figure(figsize=(3.4, 4.5), dpi=300)
gs0 = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.0)

axs = {}
axs["linear"] = plt.subplot(gs0[0])
axs["nonlinear"] = plt.subplot(gs0[1])


name_to_label = {
    "core_model": "PIAA-ZWFS",
    "ideal_model": "Ideal",
    "nominal_zwfs_model": "1.06 ZWFS",
    "nominal_2zwfs_model": "2.0 ZWFS",
}

plot_kwargs = dict(alpha=0.5, markersize=2)

for name in recon_errors["linear"]:
    axs["linear"].plot(
        rms_ins * 1e9,
        recon_errors["linear"][name] * 1e9,
        "o",
        label=name_to_label.get(name, name),
        **plot_kwargs,
    )

axs["linear"].legend(loc="lower right", fontsize=8, frameon=True)
axs["linear"].set_xscale("log")
axs["linear"].set_yscale("log")
axs["linear"].yaxis.set_minor_locator(
    mticker.LogLocator(base=10, subs=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
)
axs["linear"].yaxis.set_minor_formatter(mticker.NullFormatter())
axs["linear"].minorticks_on()
axs["linear"].tick_params(axis="y", which="minor", left=True, right=False, length=3)

xlim = axs["linear"].get_xlim()
axs["linear"].plot(xlim, xlim, "k--", label="y=x")
axs["linear"].set_xlim(xlim)

title_text_kwargs = dict(
    fontsize=10,
    verticalalignment="top",
    bbox=dict(
        facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.8
    ),
)

axs["linear"].text(
    0.05,
    0.95,
    "Linear Recon.",
    transform=axs["linear"].transAxes,
    **title_text_kwargs,
)


axs["nonlinear"] = plt.subplot(gs0[1])
for name in recon_errors["nonlinear"]:
    axs["nonlinear"].plot(
        rms_ins * 1e9,
        recon_errors["nonlinear"][name] * 1e9,
        "o",
        label=name_to_label.get(name, name),
        **plot_kwargs,
    )

axs["nonlinear"].set_xscale("log")
axs["nonlinear"].set_yscale("log")
xlim = axs["nonlinear"].get_xlim()
axs["nonlinear"].plot(xlim, xlim, "k--", label="y=x")
axs["nonlinear"].set_xlim(xlim)

axs["nonlinear"].set_ylim(axs["linear"].get_ylim())

axs["nonlinear"].text(
    0.05,
    0.95,
    "Non-linear Recon.",
    transform=axs["nonlinear"].transAxes,
    **title_text_kwargs,
)


# turn of xlabels for top plot
axs["linear"].set_xticklabels([])


axs["nonlinear"].set_xlabel("Input OPD [nm RMS]")

# ylabel common to both - find location and put text at midpoint
fig = plt.gcf()
top_pos = axs["linear"].get_position().y1
bottom_pos = axs["nonlinear"].get_position().y0
mid_y = (top_pos + bottom_pos) / 2
fig.text(
    -0.02,
    mid_y,
    "Reconstruction Error [nm RMS]",
    rotation="vertical",
    va="center",
    fontsize=10,
)


plt.savefig(
    "../results/figures/recon_errors.pdf", bbox_inches="tight", transparent=True
)

# %%

# %%
# Binned median and 16–84 percentile plots


def compute_binned_stats(x, y, nbins=20):
    # x, y are 1D arrays (nm units). Returns centers, median, p16, p84.
    x = np.asarray(x)
    y = np.asarray(y)

    # Filter to positive x for log-binning
    pos = x > 0
    x = x[pos]
    y = y[pos]

    if x.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    # Guard against degenerate range
    if xmin <= 0:
        xmin = np.nanmin(x[x > 0])
    if xmax <= xmin:
        xmax = xmin * 1.01

    bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)
    centers = np.sqrt(bins[:-1] * bins[1:])  # geometric mean for log bins

    med = np.full(nbins, np.nan)
    p16 = np.full(nbins, np.nan)
    p84 = np.full(nbins, np.nan)

    for i in range(nbins):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if np.any(m):
            ym = y[m]
            med[i] = np.nanmedian(ym)
            p16[i] = np.nanpercentile(ym, 16)
            p84[i] = np.nanpercentile(ym, 84)

    return centers, med, p16, p84


# Build figure with two panels: linear and non-linear
fig2 = plt.figure(figsize=(3.4, 4.5), dpi=300)
gs1 = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.0)

axs2 = {}
axs2["linear"] = plt.subplot(gs1[0])
axs2["nonlinear"] = plt.subplot(gs1[1])

nbins = 30

# Convert x once to nm
x_nm = np.asarray(rms_ins) * 1e9

# Linear
for name in sorted(recon_errors["linear"], key=lambda name: name != "ideal_model"):
    y_nm = np.asarray(recon_errors["linear"][name]) * 1e9
    centers, med, p16, p84 = compute_binned_stats(x_nm, y_nm, nbins=nbins)
    line = axs2["linear"].plot(
        centers,
        med,
        label=name_to_label.get(name, name),
        lw=1.5,
        color="black" if name == "ideal_model" else None,
    )[0]
    c = line.get_color()
    axs2["linear"].fill_between(centers, p16, p84, color=c, alpha=0.2)

axs2["linear"].legend(loc="lower right", fontsize=8, frameon=True)
axs2["linear"].set_xscale("log")
axs2["linear"].set_yscale("log")
axs2["linear"].yaxis.set_minor_locator(
    mticker.LogLocator(base=10, subs=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
)
axs2["linear"].yaxis.set_minor_formatter(mticker.NullFormatter())
axs2["linear"].minorticks_on()
axs2["linear"].tick_params(axis="y", which="minor", left=True, right=False, length=3)

xlim2 = axs2["linear"].get_xlim()
axs2["linear"].plot(xlim2, xlim2, color="grey", ls="--", label="y=x")
axs2["linear"].set_xlim(xlim2)

axs2["linear"].text(
    0.05,
    0.95,
    "Linear Recon.",
    transform=axs2["linear"].transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(
        facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.8
    ),
)

# Non-linear
for name in sorted(recon_errors["nonlinear"], key=lambda name: name != "ideal_model"):
    y_nm = np.asarray(recon_errors["nonlinear"][name]) * 1e9
    centers, med, p16, p84 = compute_binned_stats(x_nm, y_nm, nbins=nbins)
    line = axs2["nonlinear"].plot(
        centers,
        med,
        label=name_to_label.get(name, name),
        lw=1.5,
        alpha=0.7,
        color="black" if name == "ideal_model" else None,
    )[0]
    c = line.get_color()
    axs2["nonlinear"].fill_between(centers, p16, p84, color=c, alpha=0.2)

axs2["nonlinear"].set_xscale("log")
axs2["nonlinear"].set_yscale("log")
# xlim2b = (rms_ins[0] * 1e9, rms_ins[-1] * 1e9)
xlim2b = (centers[0], centers[-1])
axs2["nonlinear"].plot(xlim2b, xlim2b, color="grey", ls="--", label="y=x")
axs2["nonlinear"].set_xlim(xlim2b)

axs2["linear"].set_xlim(xlim2b)

# Mark relevant wavefront-error scales above the top panel. The x-axis is in nm.
wavelength_nm = 900
for divisor in (1000, 100, 10):
    x_marker = wavelength_nm / divisor
    axs2["linear"].plot(
        x_marker,
        1.02,
        marker="v",
        color="black",
        markersize=4,
        transform=axs2["linear"].get_xaxis_transform(),
        clip_on=False,
    )
    axs2["linear"].annotate(
        rf"$\lambda/{divisor}$",
        xy=(x_marker, 1.02),
        xycoords=axs2["linear"].get_xaxis_transform(),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        annotation_clip=False,
    )


# Match y-limits with the linear panel
axs2["nonlinear"].set_ylim(axs2["linear"].get_ylim())

axs2["nonlinear"].text(
    0.05,
    0.95,
    "Non-linear Recon.",
    transform=axs2["nonlinear"].transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(
        facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.8
    ),
)

# turn off xlabels for top plot
axs2["linear"].set_xticklabels([])

axs2["nonlinear"].set_xlabel("Input OPD [nm RMS]")

# Common y-label between panels
fig2 = plt.gcf()
top_pos2 = axs2["linear"].get_position().y1
bottom_pos2 = axs2["nonlinear"].get_position().y0
mid_y2 = (top_pos2 + bottom_pos2) / 2
fig2.text(
    -0.02,
    mid_y2,
    "Reconstruction Error [nm RMS]",
    rotation="vertical",
    va="center",
    fontsize=10,
)

# in the linear plot, put text just above y=x line indicating that it is y=x

along_line_theta = 0.1  # interpolation parameter along the line
x_text = xlim2[0] * (xlim2[1] / xlim2[0]) ** along_line_theta
y_text = xlim2[0] * (xlim2[1] / xlim2[0]) ** along_line_theta

# compute angle knowing that it is log-log plot - need to use ylim too to get aspect ratio
text_angle = 25

y_offset = 1.3  # multiplicative offset to move text above the line

axs2["linear"].text(
    x_text,
    y_text * y_offset,
    "$y = x$",
    rotation=text_angle,
    fontsize=8,
    color="black",
    horizontalalignment="left",
    verticalalignment="bottom",
)


plt.savefig(
    "../results/figures/recon_errors_binned.pdf", bbox_inches="tight", transparent=True
)
plt.savefig(
    "../results/figures/recon_errors_binned.svg", bbox_inches="tight", transparent=True
)

# %%
