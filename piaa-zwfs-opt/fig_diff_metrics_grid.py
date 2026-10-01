"""Compare optimisation metrics across apertures and spectral bandwidths."""

# %% Imports and plotting style
import os
from pathlib import Path

# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# Must be set before JAX is imported.
os.environ["JAX_ENABLE_X64"] = "False"

import dlWFSoptimize as dlwfs
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import scienceplots

plt.style.use(["science"])
plt.rcParams.update({"font.size": 8})

# %% Grid configuration and paths
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results"
MODEL_PATH = RESULTS_PATH / "dl_optimizations" / "diff_metrics" / "generalisations"
FIGURE_PATH = RESULTS_PATH / "figures" / "diff_metrics_grid.svg"

APERTURES = (("magellan", "Magellan"), ("gmt", "GMT"), ("elt", "ELT"))
WAVELENGTH_CENTRE_NM = 1000.0
BANDWIDTHS_NM = (1.0, 200.0, 400.0)
# BANDWIDTHS_NM = (400.0,)
n_freqs = 35
MODEL_VARIANTS = (
    (
        "Single high freq.",
        1,
        "with_PIAA_annular_mask_point_source_neg_FI_trace_normed",
    ),
    (
        "Many freq. (FI opt.)",
        7,
        "with_PIAA_annular_mask_point_source_neg_FI_trace_normed",
    ),
    ("Many freq. (Cov. opt.)", 7, "with_PIAA_annular_mask_point_source"),
    ("1.06 ZWFS", 7, "no_optimization_1.06_loD_init_eval_diam"),
)


# %% Model paths and batched FI evaluation
def model_path(aperture, bandwidth_nm, n_terms, suffix):
    return MODEL_PATH / (
        f"{aperture}_Fourier_basis_{n_terms}_terms_"
        f"{WAVELENGTH_CENTRE_NM:g}nm_{bandwidth_nm:.1f}nm_bw_band_{suffix}"
    )


def rms_fi_curve(model, pupil_diameter, bound, n_freqs, batch_size=3):
    """Evaluate FI diagonals in batches to limit GPU memory use."""
    freqs = onp.linspace(0.5, 8.0, n_freqs)
    combined = []
    for start in range(0, n_freqs, batch_size):
        batch = freqs[start : start + batch_size]
        sin_fi, cos_fi, _ = model.compute_standard_sensitivity(
            pupil_diameter=pupil_diameter,
            n_freqs=len(batch),
            freq_start=float(batch[0]),
            freq_end=float(batch[-1]),
        )
        sin_fi = np.diag(sin_fi) / bound
        cos_fi = np.diag(cos_fi) / bound
        combined.append(onp.asarray(np.sqrt((sin_fi**2 + cos_fi**2) / 2)))
    return freqs, onp.concatenate(combined)


# %% Validate saved models
for aperture, _ in APERTURES:
    for bandwidth_nm in BANDWIDTHS_NM:
        for _, n_terms, suffix in MODEL_VARIANTS:
            path = model_path(aperture, bandwidth_nm, n_terms, suffix)
            if not any(path.glob("core_model_*.model")):
                raise FileNotFoundError(f"No saved core model in {path}")


# %% Compute sensitivity curves
curves = {}
cov_losses = {}
for aperture, aperture_label in APERTURES:
    pupil_diameter = dlwfs.aperture_name_to_pupil_diameter(aperture)
    for bandwidth_nm in BANDWIDTHS_NM:
        curves[aperture, bandwidth_nm] = []
        cov_losses[aperture, bandwidth_nm] = []
        bound = None
        for label, n_terms, suffix in MODEL_VARIANTS:
            model = dlwfs.load_model(
                model_path(aperture, bandwidth_nm, n_terms, suffix)
            )[0]
            if bound is None:
                # Compare all four models on the same orthonormal Fourier basis.
                standard_basis, _ = dlwfs.make_freq_basis(
                    wf_npixels=model.tel.wf_npixels,
                    sim_diam=model.tel.diameter,
                    pupil_diameter=pupil_diameter,
                    pupil=model.tel.TransmissiveLayer.transmission,
                    min_cycles=0.01,
                    max_cycles=8.0,
                    num_freqs=7,
                    trig_type="both",
                )
                standard_basis = np.concatenate(
                    [standard_basis[:1], standard_basis[2:]], axis=0
                )
                standard_basis = dlwfs.orthonormalise_basis(
                    basis=standard_basis,
                    support=model.tel.TransmissiveLayer.transmission,
                )
                standard_basis *= 1e-6

                # Normalise each panel to its own aperture and source spectrum.
                ideal_model = dlwfs.generate_idealwfs_optics(model)
                ideal_fi = ideal_model.compute_standard_sensitivity(
                    pupil_diameter=pupil_diameter, n_freqs=1
                )
                bound = ideal_fi[0][0, 0]

            model_new_basis = model.update_model_basis(standard_basis)
            cov_loss = float(model_new_basis.loss("cov_trace_normed"))
            cov_losses[aperture, bandwidth_nm].append(cov_loss)
            print(
                f"{aperture_label}, {bandwidth_nm:g} nm: {label}, "
                f"cov_trace_normed={cov_loss:.8g}",
                flush=True,
            )

            curves[aperture, bandwidth_nm].append(
                rms_fi_curve(model, pupil_diameter, bound, n_freqs)
            )


# %% Plot grid, shared legend, and labels
fig, axes = plt.subplots(
    3,
    3,
    figsize=(7.2, 5.6),
    dpi=300,
    sharex=True,
    sharey=True,
    gridspec_kw={
        "hspace": 0,
        "wspace": 0,
    },
)
sensitivity_ticks = np.arange(0, 2.1, 0.5)

for row, (aperture, aperture_label) in enumerate(APERTURES):
    for col, bandwidth_nm in enumerate(BANDWIDTHS_NM):
        ax = axes[row, col]
        for index, (label, _, _) in enumerate(MODEL_VARIANTS):
            freqs, combined = curves[aperture, bandwidth_nm][index]
            ax.plot(
                freqs,
                combined,
                color=f"C{index}",
                linestyle="--" if index == 3 else "-",
                label=label,
            )

        ax.set_xlim(0.5, 8.0)
        ax.set_ylim(0, 1.0)
        # A text-only legend chooses a position that avoids the plotted curves.
        ax.legend(
            handles=ax.lines,
            labels=[
                rf"${loss * 1e3:.1f}\times 10^{{-3}}$"
                for loss in cov_losses[aperture, bandwidth_nm]
            ],
            title=r"$\overline{\mathrm{Cov}}$",
            # loc="upper right",
            # set xy position manually
            loc="upper right",
            bbox_to_anchor=(0.95, 0.91),
            bbox_transform=ax.transAxes,
            labelcolor="linecolor",
            handlelength=0,
            handletextpad=0,
            fontsize=6.5,
            title_fontsize=6.5,
            labelspacing=0.3,
            frameon=True,
        )
        if row == 0:
            ax.set_title(
                rf"$\Delta\lambda / \lambda_c = {bandwidth_nm / WAVELENGTH_CENTRE_NM:g}$"
            )
        if col == 0:
            ax.set_ylabel(aperture_label)
        if col == len(BANDWIDTHS_NM) - 1:
            sensitivity_ax = ax.twinx()
            sensitivity_ax.set_ylim(ax.get_ylim())
            sensitivity_ax.set_yticks(sensitivity_ticks**2 / 4)
            sensitivity_ax.set_yticklabels(
                [f"{tick:.2f}" for tick in sensitivity_ticks]
            )
            sensitivity_ax.minorticks_off()
            if row == 1:
                sensitivity_ax.set_ylabel(r"Sensitivity [/ph/rad (at $\lambda_c$)]")

# Hide endpoint labels at internal row boundaries on both y-axes.
for ax in fig.axes:
    row = ax.get_subplotspec().rowspan.start
    lower, upper = ax.get_ylim()
    for tick, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        if (row > 0 and onp.isclose(tick, upper)) or (
            row < len(APERTURES) - 1 and onp.isclose(tick, lower)
        ):
            label.set_visible(False)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=4,
    fontsize=7,
    frameon=False,
)
fig.supxlabel("Spatial frequency [cycles/pupil]", fontsize=8)
fig.supylabel("Fisher Information / bound", fontsize=8, x=0.01)
fig.tight_layout(pad=0.5, rect=(0.0, 0.0, 1.0, 0.96))


# %% Save figure
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE_PATH, bbox_inches="tight", transparent=True)
# replace svg with pdf in the path for printing
fig.savefig(FIGURE_PATH.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
print(f"Saved {FIGURE_PATH}")

# %%
model = dlwfs.load_model(
    model_path(
        "magellan", 400.0, 7, "with_PIAA_annular_mask_point_source_neg_FI_trace_normed"
    )
)[0]

model.loss("cov_trace_normed")

FIM = model.compute_FI("phase_coefficients")
print(f"Max in FIM: {np.max(FIM)}")
plt.imshow(FIM)
plt.colorbar()
plt.figure()

plt.plot(model.tel.mask_at_focus.eval_basis()[80, :])
# %%
model = dlwfs.load_model(
    model_path("magellan", 400.0, 7, "with_PIAA_annular_mask_point_source")
)[0]
model.loss("cov_trace_normed")

FIM = model.compute_FI("phase_coefficients")
print(f"Max in FIM: {np.max(FIM)}")
plt.imshow(FIM)
plt.colorbar()

plt.figure()

plt.plot(model.tel.mask_at_focus.eval_basis()[80, :])
# %%

# %%
