# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import asdf
import scienceplots

plt.style.use(["science"])

# text size
plt.rcParams.update({"font.size": 8})

# %%
pth = os.path.join(
    "..",
    "results",
    "dl_optimizations",
    "src_dependence",
    "source_size_dependence_results.asdf",
)

ds = asdf.open(pth)

fig_savepth = os.path.join(
    "..",
    "results",
    "figures",
    "stellar_size_dep.pdf",
)

# %%
# opposite of this:

# af["r_max_vals_loD"] = onp.array(r_max_vals / loD)
# af["losses"] = losses
# # also save the model names
# af["models_of_interest"] = models_of_interest

r_max_vals_loD = ds["r_max_vals_loD"]
losses = ds["losses"]  # dict of arrays
models_of_interest = ds["models_of_interest"]
qi_bound = ds["QI_bound"]


# %%
models_used = [
    "2.0 ZWFS",
    "PIAA, annular",
    "resolve optimized 0.8",
    "resolve optimized 1.5",
    "ideal WFS",
]

leg_labels = {
    "2.0 ZWFS": "2 ZWFS",
    "PIAA, annular": "PIAA ZWFS (Point opt.)",
    "resolve optimized 0.8": "PIAA ZWFS (0.8 $\lambda/D$ opt.)",
    "resolve optimized 1.5": "PIAA ZWFS (1.5 $\lambda/D$ opt.)",
}

# %%
fig = plt.figure(figsize=(3.46457, 2.6), dpi=300)
for i, model in enumerate(models_of_interest):
    if model not in models_used:
        continue
    kwargs = {}
    if model == "ideal WFS":
        continue

    plt.plot(
        r_max_vals_loD,
        losses[model],
        label=leg_labels.get(model, model),
        **kwargs,
    )

# plto ideal WFS
model = "ideal WFS"
plt.plot(
    r_max_vals_loD,
    losses[model],
    label="Ideal WFS",
    linestyle="--",
    color="k",
)

plt.yscale("log")
plt.xlabel(r"$R_{\star}$ [$\lambda/D$]")
plt.ylabel("Loss [$\mu$m$^2$/ph.]")
plt.scatter([0], [qi_bound], color="k", label="QI Bound")
# plt.grid(True, which="both", ls="--", lw=0.5)
plt.xlim(0, np.max(r_max_vals_loD))
plt.legend()

plt.savefig(fig_savepth, bbox_inches="tight")

# %%
import dlWFSoptimize as dlwfs

ModelCore = dlwfs.ModelCore

model_unresolved = dlwfs.load_model(
    "../results/dl_optimizations/src_dependence/elt_Fourier_basis_7_terms_Z_band_with_PIAA_annular_mask_point_source_0.0_mag"
)[0]

model_resolved = dlwfs.load_model(
    "../results/dl_optimizations/src_dependence/elt_Fourier_basis_7_terms_Z_band_with_PIAA_annular_mask_resolved_source_1.50_loD_0.0_mag"
)[0]


def plot_phase_mask(model, ax):
    thicc = model.tel.optics.ThreePlaneSystem.mask_at_focus.eval_basis()
    ax.imshow(thicc)


plt.figure(figsize=(6, 3), dpi=300)
ax = plt.subplot(1, 2, 1)
plot_phase_mask(model_unresolved, ax)
ax.set_title("Unresolved source optimization")
# add cbar
plt.colorbar(ax.images[0], ax=ax)
ax = plt.subplot(1, 2, 2)
plot_phase_mask(model_resolved, ax)
ax.set_title("Resolved source optimization")
# add cbar
plt.colorbar(ax.images[0], ax=ax)

# %%
FIs = ds["FIs"]["0.0"]
plt.figure()

bound = ds["FIs"]["0.0"]["ideal WFS"][0][0, 0]

freqs = ds["FIs"]["0.0"]["ideal WFS"][2]

for i, model_name in enumerate(models_used):
    sin_FI = FIs[model_name][0] / bound
    cos_FI = FIs[model_name][1] / bound

    sin_sensitivity = np.diag(sin_FI)
    cos_sensitivity = np.diag(cos_FI)

    color = f"C{i}"

    plt.plot(freqs, sin_sensitivity, color=color, ls="-")
    plt.plot(freqs, cos_sensitivity, color=color, ls="--")
    plt.plot(
        [], [], color=color, ls="-", label=f"{leg_labels.get(model_name, model_name)}"
    )

plt.legend()
# %%
FIs = ds["FIs"]["0.4"]
plt.figure()

freqs = ds["FIs"]["0.4"]["ideal WFS"][2] * ds["pupil_diameter"]

for i, model_name in enumerate(models_used):
    sin_FI = FIs[model_name][0] / bound
    cos_FI = FIs[model_name][1] / bound

    sin_sensitivity = np.diag(sin_FI)
    cos_sensitivity = np.diag(cos_FI)

    combined_sensitivity = (
        1 / np.sqrt(2) * np.sqrt(sin_sensitivity**2 + cos_sensitivity**2)
    )

    color = f"C{i}" if not (model_name == "ideal WFS") else "k"

    plt.plot(freqs, sin_sensitivity, color=color, ls=":", alpha=0.5)
    plt.plot(freqs, cos_sensitivity, color=color, ls="--", alpha=0.5)
    plt.plot(freqs, combined_sensitivity, color=color, ls="-")
    plt.plot(
        [], [], color=color, ls="-", label=f"{leg_labels.get(model_name, model_name)}"
    )

plt.legend()
plt.xlim(0, np.max(freqs))
# %%
FIs = ds["FIs"]["0.75"]
plt.figure()

freqs = ds["FIs"]["0.75"]["ideal WFS"][2] * ds["pupil_diameter"]

for i, model_name in enumerate(models_used):
    sin_FI = FIs[model_name][0] / bound
    cos_FI = FIs[model_name][1] / bound

    sin_sensitivity = np.diag(sin_FI)
    cos_sensitivity = np.diag(cos_FI)

    combined_sensitivity = (
        1 / np.sqrt(2) * np.sqrt(sin_sensitivity**2 + cos_sensitivity**2)
    )

    color = f"C{i}" if not (model_name == "ideal WFS") else "k"

    plt.plot(freqs, sin_sensitivity, color=color, ls=":", alpha=0.5)
    plt.plot(freqs, cos_sensitivity, color=color, ls="--", alpha=0.5)
    plt.plot(freqs, combined_sensitivity, color=color, ls="-")
    plt.plot(
        [], [], color=color, ls="-", label=f"{leg_labels.get(model_name, model_name)}"
    )

plt.legend()
plt.xlim(0, np.max(freqs))


# %%

# gridspec
import matplotlib.gridspec as gridspec
import cmasher as cmr

plt.figure(figsize=(3.4, 7.9), dpi=300)
gs0 = gridspec.GridSpec(3, 1, height_ratios=[1.0, 3, 1.0], hspace=0.25)
gs10 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs0[1], hspace=0.0)
gs20 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[2])

axs = {}
axs["losses"] = plt.subplot(gs0[0])
axs["FI_0.0"] = plt.subplot(gs10[0])
axs["FI_0.4"] = plt.subplot(gs10[1])
axs["FI_0.75"] = plt.subplot(gs10[2])
axs["FI_mat_2.0_ZWFS"] = plt.subplot(gs20[0])
axs["FI_mat_PIAA_annular"] = plt.subplot(gs20[1])

axs["FI_mat_resolve_opt_1p5"] = plt.subplot(gs20[2])

# plto ideal WFS
model_name = "ideal WFS"
axs["losses"].plot(
    [],
    [],
    label="Ideal",
    linestyle="--",
    color="k",
)
axs["losses"].scatter([0], [qi_bound], color="k", label="QI Bound", s=10)

# loss plot
for i, model_name in enumerate(models_used):
    if model_name == "ideal WFS":
        continue
    axs["losses"].plot(
        r_max_vals_loD
        * 2,  # everything is doubled to convert from R_star to diameter (for easier comparison to real stars)
        losses[model_name],
        label=leg_labels.get(model_name, model_name),
    )

axs["losses"].plot(
    r_max_vals_loD
    * 2,  # everything is doubled to convert from R_star to diameter (for easier comparison to real stars)
    losses[model_name],
    linestyle="--",
    color="k",
)
axs["losses"].set_yscale("log")
axs["losses"].set_xlabel(r"$D_{\star}$ [$\lambda/D$]")
# axs["losses"].set_ylabel("Average variance [/frame/ph/mode]")
axs["losses"].set_ylabel("Loss [$\mu$m$^2$/ph.]")
# legend with two columns (PIAA_ZWFS models in one column, others in another)
axs["losses"].legend(frameon=True, ncol=2, fontsize=6)
axs["losses"].set_xlim(0, np.max(r_max_vals_loD * 2))


def plot_FI_subplot(ax, FI_dict, freq_scale, clip_yticks=False, rm_xtick_labels=False):
    freqs = FI_dict["ideal WFS"][2] * freq_scale

    for i, model_name in enumerate(models_used):
        sin_FI = FI_dict[model_name][0] / bound
        cos_FI = FI_dict[model_name][1] / bound

        sin_sensitivity = np.diag(sin_FI)
        cos_sensitivity = np.diag(cos_FI)

        combined_sensitivity = (
            1 / np.sqrt(2) * np.sqrt(sin_sensitivity**2 + cos_sensitivity**2)
        )

        color = f"C{i}" if not (model_name == "ideal WFS") else "k"

        ax.plot(freqs, sin_sensitivity, color=color, ls=":", alpha=0.5)
        ax.plot(freqs, cos_sensitivity, color=color, ls="--", alpha=0.5)
        ax.plot(freqs, combined_sensitivity, color=color, ls="-")
    ax.plot([], [], color="k", ls=":", label="sin")
    ax.plot([], [], color="k", ls="--", label="cos")
    ax.plot([], [], color="k", ls="-", label="both")

    ax.set_xlim(np.min(freqs), np.max(freqs))
    ax.set_ylim(0, 1.0)

    if clip_yticks:
        # only do 0.2, 0.4, 0.6, 0.8
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    else:
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    if rm_xtick_labels:
        ax.set_xticklabels([])


plot_FI_subplot(
    axs["FI_0.0"],
    ds["FIs"]["0.0"],
    ds["pupil_diameter"],
    rm_xtick_labels=True,
)
plot_FI_subplot(
    axs["FI_0.4"],
    ds["FIs"]["0.4"],
    ds["pupil_diameter"],
    clip_yticks=True,
    rm_xtick_labels=True,
)
plot_FI_subplot(
    axs["FI_0.75"],
    ds["FIs"]["0.75"],
    ds["pupil_diameter"],
    clip_yticks=True,
)

axs["FI_0.4"].set_ylabel("Fisher information / point source limit")

axs["FI_0.0"].legend()

axs["FI_0.75"].set_xlabel("Spatial frequency [cycles/pupil]", labelpad=0)

# text in top left corner showing R_star value
axs["FI_0.0"].text(
    0.01,
    0.9,
    r"Point src.",
    transform=axs["FI_0.0"].transAxes,
    bbox=dict(
        facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.8
    ),
)
axs["FI_0.4"].text(
    0.01,
    0.9,
    r"$D_{\star} = 0.8 \lambda/D$",
    transform=axs["FI_0.4"].transAxes,
    bbox=dict(
        facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.8
    ),
)
# add a box
axs["FI_0.75"].text(
    0.01,
    0.9,
    r"$D_{\star} = 1.5 \lambda/D$",
    transform=axs["FI_0.75"].transAxes,
    bbox=dict(
        facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.8
    ),
)


# on top of the first plot (above axis), put markers at 0.0, 0.4, 0.75
def add_arrow(
    ax,
    interesting_x,
    text=None,
    text_x_offset=0.0,
    color="black",
    ha="center",
):
    """
    Add an arrow to the top of the plot at the specified x position.
    """
    # Get the current y-limits
    # Note: This should be done after setting axis limits
    # e.g. after ax.set_ylim(...)
    ylim = ax.get_ylim()
    y_arrow = ylim[1] * 1.1  # 10% above the top y-limit

    ax.annotate(
        "",  # No text
        xy=(interesting_x, ylim[1]),  # Arrow tip (at top of plot)
        xytext=(interesting_x, y_arrow),  # Arrow tail (above plot)
        arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle="-|>", lw=1),
        annotation_clip=False,  # Show even if outside axes
        ha="center",
        color=color,
    )

    if text is not None:
        ax.text(
            interesting_x + text_x_offset,
            y_arrow + 0.0 * (ylim[1] - ylim[0]),  # Adjust text position above arrow
            text,
            ha=ha,
            va="bottom",
            color=color,
        )


# add_arrow(axs["losses"], 0.0, text="", text_x_offset=0.05, color="C1")
# add_arrow(axs["losses"], 0.4, text="", text_x_offset=0.05, color="C2")
# add_arrow(axs["losses"], 0.75, text="", text_x_offset=0.05, color="C3")

# arrows on top showing the R_star values for real stars
import astropy.units as u

m = dlwfs.load_model(
    "../results/dl_optimizations/src_dependence_240px/elt_Fourier_basis_7_terms_Z_band_no_optimization_2.0_loD_init_eval_diam"
)[0]

lamb = m.tel.source.wavelengths.mean() * u.m
D = 39 * u.m

loD = (lamb / D * u.rad).to(u.mas)
print(f"lambda/D: {loD:.2f}")

Dstars = {
    "R Scl": 8.9 * u.mas,
    "HR 5171 Aa": 4.10 * u.mas,
    "Altair": 3.2 * u.mas,
}

for name, dstar in Dstars.items():
    r_max = ((dstar).to(u.rad) / loD).to("").value
    print(f"{name}: R_star = {r_max:.2f} lambda/D")
has = {
    "R Scl": "right",
    "HR 5171 Aa": "left",
    "Altair": "right",
}
star_color = "k"
for name, dstar in Dstars.items():
    r_max = ((dstar).to(u.rad) / loD).to("").value
    add_arrow(
        axs["losses"],
        r_max,
        text=name,
        text_x_offset=0.04 if has[name] == "left" else -0.04,
        color=star_color,
        ha=has[name],
    )


pth = os.path.join(
    "..",
    "results",
    "dl_optimizations",
    "src_dependence",
    ds["models_of_interest"]["PIAA, annular"],
)
flux = dlwfs.load_model(pth)[0].tel.source.flux

nom_FIs = ds["nominal_FIs"]["0.75"]

nom_FIs = {name: FI / flux for name, FI in nom_FIs.items()}


def asymmetric_matrix_display(matrix, upper_triangle_scale=5.0):
    """Keep the lower triangle linear and stretch the upper triangle for display."""
    display_matrix = np.tril(matrix).copy()
    display_matrix += np.triu(matrix, k=1) * upper_triangle_scale
    return display_matrix


upper_triangle_scale = 5.0

display_FIs = {
    name: asymmetric_matrix_display(FI, upper_triangle_scale=upper_triangle_scale)
    for name, FI in nom_FIs.items()
}

# shared color limits across three matrices
clims = max(
    np.max(np.abs(display_FIs["2.0 ZWFS"])),
    np.max(np.abs(display_FIs["PIAA, annular"])),
    np.max(np.abs(display_FIs["resolve optimized 1.5"])),
)

# Plot three Fisher information matrices using a loop
fi_axes = [
    ("2.0 ZWFS", "FI_mat_2.0_ZWFS", "C0"),
    ("PIAA, annular", "FI_mat_PIAA_annular", "C1"),
    ("resolve optimized 1.5", "FI_mat_resolve_opt_1p5", "C3"),
]

for model_name, ax_key, color in fi_axes:
    axs[ax_key].imshow(
        display_FIs[model_name],
        vmin=-clims,
        vmax=clims,
        cmap="bwr",
    )
    axs[ax_key].set_xticks([])
    axs[ax_key].set_yticks([])
    for side in ["top", "bottom", "left", "right"]:
        axs[ax_key].spines[side].set_color(color)
        axs[ax_key].spines[side].set_linewidth(1.5)

# add some text in the top right corner that shows the scaling factor for that matrix (e.g. "upper 5x")
for model_name, ax_key, color in fi_axes:
    axs[ax_key].text(
        0.98,
        0.98,
        f"Upper {upper_triangle_scale:.0f}$\\times$",
        color="k",
        ha="right",
        va="top",
        transform=axs[ax_key].transAxes,
        fontsize=6,
    )

# widen the bottom row to use (nearly) full figure width
pos_ref = axs["FI_mat_2.0_ZWFS"].get_position()
new_left, new_right = 0.02, 0.98
gap = 0.02
panel_w = (new_right - new_left - 2 * gap) / 3
y0, height = pos_ref.y0, pos_ref.height
# attempt to make the bottom row a bit taller without overlapping the row above
upper_pos = axs["FI_0.75"].get_position()
safety = 0.005
target_height = height * 1.2
max_height_allowed = max(height, upper_pos.y0 - y0 - safety)
new_height = min(target_height, max_height_allowed)
for i, key in enumerate(
    ["FI_mat_2.0_ZWFS", "FI_mat_PIAA_annular", "FI_mat_resolve_opt_1p5"]
):
    x0 = new_left + i * (panel_w + gap)
    axs[key].set_position([x0, y0, panel_w, new_height])

# set a common title for the row ("FI matrices"), using updated positions
fig = plt.gcf()
left_pos = axs["FI_mat_2.0_ZWFS"].get_position()
right_pos = axs["FI_mat_resolve_opt_1p5"].get_position()
row_center_x = (left_pos.x0 + right_pos.x1) / 2
row_top_y = max(left_pos.y1, right_pos.y1)
fig.text(
    row_center_x,
    row_top_y,
    r"Fisher information matrices ($D_{\star} = 1.5 \lambda/D$)",
    ha="center",
    va="bottom",
    fontsize=10,
)


# print(f"Average FI 2.0 ZWFS: {np.mean(np.diag(nom_FIs['2.0 ZWFS'])):.2e}")
# print(f"Average FI PIAA annular: {np.mean(np.diag(nom_FIs['PIAA, annular'])):.2e}")

# print(
#     f"Average covariance 2.0 ZWFS: {np.mean(np.diag(np.linalg.inv(nom_FIs['2.0 ZWFS']))):.2e}"
# )
# print(
#     f"Average covariance PIAA annular: {np.mean(np.diag(np.linalg.inv(nom_FIs['PIAA, annular']))):.2e}"
# )
# within the plots, bottom left corner, put the average FI and average covariance
avg_fi_2zwfs = np.mean(np.diag(nom_FIs["2.0 ZWFS"]))
avg_fi_piaa = np.mean(np.diag(nom_FIs["PIAA, annular"]))
avg_fi_resopt = np.mean(np.diag(nom_FIs["resolve optimized 1.5"]))
avg_cov_2zwfs = np.mean(np.diag(np.linalg.inv(nom_FIs["2.0 ZWFS"])))
avg_cov_piaa = np.mean(np.diag(np.linalg.inv(nom_FIs["PIAA, annular"])))
avg_cov_resopt = np.mean(np.diag(np.linalg.inv(nom_FIs["resolve optimized 1.5"])))


def sci_fmt(x, precision=1):
    s = f"{x:.{precision}e}"
    m_str, e_str = s.split("e")
    m = float(m_str)
    e = int(e_str)
    return rf"${m:.{precision}f} \times 10^{{{e}}}$"


for model_name, ax_key, color, avg_fi, avg_cov in [
    ("2.0 ZWFS", "FI_mat_2.0_ZWFS", "C0", avg_fi_2zwfs, avg_cov_2zwfs),
    ("PIAA, annular", "FI_mat_PIAA_annular", "C1", avg_fi_piaa, avg_cov_piaa),
    (
        "resolve optimized 1.5",
        "FI_mat_resolve_opt_1p5",
        "C3",
        avg_fi_resopt,
        avg_cov_resopt,
    ),
]:
    axs[ax_key].text(
        0.02,
        0.02,
        r"$\overline{FI}$"
        + f":{sci_fmt(avg_fi)}\n"
        + r"$\overline{\text{Cov}}$"
        + f":{sci_fmt(avg_cov)}",
        color=color,
        ha="left",
        va="bottom",
        transform=axs[ax_key].transAxes,
        fontsize=6,
    )

plt.savefig(fig_savepth, bbox_inches="tight", transparent=True)
plt.savefig(
    fig_savepth.replace(".pdf", ".svg"),
    bbox_inches="tight",
    transparent=True,
)

# %%


plt.imshow(nom_FIs["ideal WFS"], vmin=-clims, vmax=clims, cmap="bwr")
# %%
m = dlwfs.load_model(
    "../results/dl_optimizations/src_dependence/elt_Fourier_basis_7_terms_Z_band_no_optimization_2.0_loD_init_eval_diam"
)[0]

import dLux as dl

m = m.set(
    "tel.source",
    dl.PointSource(
        wavelengths=m.tel.source.wavelengths,
        flux=m.tel.source.flux,
    ),
)

# %%
plt.imshow(m.img())

# %%
around_params = {
    "phase_coefficients": np.zeros(len(m.params["phase_coefficients"])),
}

J = m.compute_J(around_params)

# %%
J["phase_coefficients"].shape

plt.imshow(J["phase_coefficients"][:, :, 0])

# %%

# %%
