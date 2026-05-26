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
import cmasher as cmr

import scienceplots

plt.style.use(["science"])
plt.rcParams.update({"font.size": 8})

# %%
ModelCore = dlwfs.ModelCore


# %%
pth = os.path.join(
    "..",
    "results",
    "dl_optimizations",
    "diff_metrics",
    "main_compare",
)


models = [
    "magellan_Fourier_basis_1_terms_Z_band_with_PIAA_annular_mask_point_source_0.0_mag_neg_FI_trace_normed",
    "magellan_Fourier_basis_1_terms_Z_band_with_PIAA_annular_mask_point_source_0.0_mag",
    "magellan_Fourier_basis_7_terms_Z_band_with_PIAA_annular_mask_point_source_0.0_mag_neg_FI_trace_normed",
    "magellan_Fourier_basis_7_terms_Z_band_with_PIAA_annular_mask_point_source_0.0_mag",
]

pupil_diameter = 6.5
# %%

models = {
    "Single high freq.": dlwfs.load_model(os.path.join(pth, models[0]))[0],
    # "High freq, neg FI": dlwfs.load_model(os.path.join(pth, models[0]))[0],
    "Many freq. (FI opt.)": dlwfs.load_model(os.path.join(pth, models[2]))[0],
    "Many freq. (Cov. opt.)": dlwfs.load_model(os.path.join(pth, models[3]))[0],
}

default_model_name = "Many freq. (FI opt.)"

ideal_model = dlwfs.generate_idealwfs_optics(models[default_model_name])

# %%
plt.imshow(models[default_model_name].img())
plt.xticks([])
plt.yticks([])


# %%
FI_ideal = ideal_model.compute_standard_sensitivity(
    pupil_diameter=pupil_diameter,
)

# %%
bound = FI_ideal[0][0, 0]


standard_basis, _ = dlwfs.make_freq_basis(
    wf_npixels=models[default_model_name].tel.wf_npixels,
    sim_diam=models[default_model_name].tel.diameter,
    pupil_diameter=6.5,
    pupil=models[default_model_name].tel.TransmissiveLayer.transmission,
    min_cycles=0.01,
    max_cycles=8.0,
    num_freqs=7,
    trig_type="both",
)
standard_basis = np.concatenate(
    [standard_basis[0][None, :, :], standard_basis[2:]],
    axis=0,
)
standard_basis = dlwfs.orthonormalise_basis(
    basis=standard_basis,
    support=models[default_model_name].tel.TransmissiveLayer.transmission,
)
standard_basis *= 1e-6

# %%
plt.figure(figsize=(3.33, 1.5), dpi=300)
for i, (name, model) in enumerate(models.items()):
    FI = model.compute_standard_sensitivity(
        pupil_diameter=pupil_diameter,
        n_freqs=35,
        freq_start=0.5,
        freq_end=8.0,
    )

    flux = model.tel.source.flux
    sin_FI = np.diag(FI[0]) / bound
    cos_FI = np.diag(FI[1]) / bound
    freqs = FI[2] * pupil_diameter
    color = f"C{i}"

    combined = np.sqrt(sin_FI**2 + cos_FI**2) / np.sqrt(2)

    plt.plot(freqs, sin_FI, alpha=0.3, linestyle="--", color=color)
    plt.plot(freqs, cos_FI, alpha=0.3, linestyle=":", color=color)

    model_new_basis = model.update_model_basis(standard_basis)

    print(name, model_new_basis.loss("cov_trace_normed"))

    plt.plot(
        freqs,
        combined,
        label=f"{name}, "
        + r"$\overline{\text{Cov}}="
        + f"{model_new_basis.loss('cov_trace_normed')*1e3:.2f}"
        + r"\times 10^{-3}$",
        color=color,
    )
plt.legend(fontsize=7)
plt.ylim(0, 1.00)
plt.xlim(freqs.min(), freqs.max())
plt.xlabel("Spatial frequency [cycles/pupil]")
plt.ylabel("Fisher Information / bound")

# Add second y-axis with transformed values
ax1 = plt.gca()
ax2 = ax1.twinx()

# Get ticks from left axis and transform them
# left_ticks = ax1.get_yticks()
# right_ticks = np.sqrt(4 * left_ticks)
# ax2.set_yticklabels([f"{tick:.2f}" for tick in right_ticks])
new_tick_values = np.arange(0, 2.1, 0.5)
new_tick_positions = new_tick_values**2 / 4
ax2.set_yticks(new_tick_positions)
ax2.set_yticklabels([f"{tick:.2f}" for tick in new_tick_values])
# remove minor ticks
ax2.minorticks_off()


ax2.set_ylabel(r"Sensitivity [/ph/rad (at $\lambda_c$)]")

fig_savepth = os.path.join(
    "..",
    "results",
    "figures",
    "max_diff_freqs.pdf",
)

# plt.savefig(fig_savepth, bbox_inches="tight", transparent=True)
# also save as svg
plt.savefig(
    fig_savepth.replace(".pdf", ".svg"),
    bbox_inches="tight",
    transparent=True,
)

# %%


standard_basis, _ = dlwfs.make_freq_basis(
    wf_npixels=models[default_model_name].tel.wf_npixels,
    sim_diam=models[default_model_name].tel.diameter,
    pupil_diameter=6.5,
    pupil=models[default_model_name].tel.TransmissiveLayer.transmission,
    min_cycles=0.01,
    max_cycles=8.0,
    num_freqs=7,
    trig_type="both",
)
standard_basis = np.concatenate(
    [standard_basis[0][None, :, :], standard_basis[2:]],
    axis=0,
)
standard_basis = dlwfs.orthonormalise_basis(
    basis=standard_basis,
    support=models[default_model_name].tel.TransmissiveLayer.transmission,
)

for i, (name, model) in enumerate(models.items()):
    plt.figure()
    model_new_basis = model.update_model_basis(standard_basis)

    FI = model_new_basis.compute_FI("phase_coefficients")

    clims = np.abs(FI).max() * np.array([-1, 1])

    plt.imshow(FI, vmin=clims[0], vmax=clims[1], cmap=cmr.prinsenvlag)
    plt.colorbar()

# %%
