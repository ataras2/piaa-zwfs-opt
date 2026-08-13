# %%
import jax.numpy as np
import jax
import matplotlib.pyplot as plt
import dlWFSoptimize as dlwfs

# %%
pupil_diameter = 8.0
model = dlwfs.make_system(
    wavels=dlwfs.get_wavels(10, "Z"),
    waveband="Z",
    exposure_time=1e-3,
    system_throughput=0.3,
    pupil_diameter=pupil_diameter,
    min_sim_diameter_frac=1.4,
    det_pix_over_pupil=60,
    num_airy=2.5,
    q=32,
    target_wf_npixels=512,
    aperture_name="magellan",
    oversample_aperture=4,
    basis_type="Fourier",
    phase_rms=1e-6,
    n_terms=5,
    basis_kwargs={
        "min_cycles": 0.01,
        "max_cycles": 8,
        "trig_type": "both",
    },
    has_PIAA=False,
    phase_mask_basis_name="circle",
    phase_mask_material="sio2",
    n_zwfs_rings=None,
    final_D=10e-3,
    piaa_material="sio2",
    n_sag_terms=None,
    d=None,
    init_PIAA_eval_dot_diam=1.06,
    read_noise=0.5,
    init_zernike_dot_diam=2.0,  # loD
    Nph_per_sec=None,
    magnitude=0.0,
    debug_plots=False,
)

# %%
model=model.set("tel.detector", dl.LayeredDetector([]))

plt.imshow(model.img())

# %%
FI = model.compute_FI("phase_coefficients")
# %%
import matplotlib.colors as mcolors

plt.imshow(FI, cmap="bwr", norm=mcolors.CenteredNorm())

# %%
# same plot but 5x everything above the diagonal
FI_scaled = FI.copy()
for i in range(FI_scaled.shape[0]):
    for j in range(FI_scaled.shape[1]):
        if i < j:
            FI_scaled = FI_scaled.at[i, j].set(FI_scaled[i, j] * 5)

plt.imshow(FI_scaled, cmap="bwr", norm=mcolors.CenteredNorm())
# %%
B = model.tel.basis

B = B.reshape((B.shape[0], -1))

# aperture = model.tel.transmission

XB = B @ B.T
plt.imshow(XB, cmap="bwr", norm=mcolors.CenteredNorm())

XB_scaled = XB.copy()
for i in range(XB_scaled.shape[0]):
    for j in range(XB_scaled.shape[1]):
        if i < j:
            XB_scaled = XB_scaled.at[i, j].set(XB_scaled[i, j] * 5)

plt.figure()
plt.imshow(XB_scaled, cmap="bwr", norm=mcolors.CenteredNorm())


# %%
import dLux as dl
import dLux.utils as dlu

aperture = model.tel.transmission


# zern = dlu.zernikes.zernike_basis(3)

zern = dlwfs.make_zernike_basis(
    wf_npixels=model.tel.optics.wf_npixels,
    sim_diam=model.tel.optics.diameter,
    pupil_diameter=pupil_diameter,
    n_zernikes=1,
    starting_index=4,
    orthonormalise=True,
    support=aperture,
)

plt.imshow(zern[0], cmap="bwr", norm=mcolors.CenteredNorm())
plt.colorbar()

# %%

lod = model.tel.wavelengths.mean() / pupil_diameter

optics = dl.LayeredOpticalSystem(
    wf_npixels=model.tel.optics.wf_npixels,
    diameter=model.tel.optics.diameter,
    layers=(
        ("TransmissiveLayer", model.tel.TransmissiveLayer),
        ("BasisLayer", model.tel.BasisLayer),
        ("Defocus", dl.AberratedLayer(opd=0.12 * zern[0] * 1e-6)),
        # ("Magnifier", model.tel.Magnifier),
        (
            "MFT",
            dl.MFT(256, lod / 8),
        ),
    ),
)

defocus_model = model.set(
    {
        "tel.optics": optics,
        "tel_params_to_params": {"BasisLayer.coefficients": "phase_coefficients"},
        "tel.detector": dl.LayeredDetector([]),
    }
)

# %%
n_B = len(B)

plt.imshow(
    defocus_model.set("params.phase_coefficients", np.zeros(n_B).at[-1].set(0.0)).img()
)
plt.colorbar()


# %%
defocus_FI, J = defocus_model.compute_FI("phase_coefficients", return_J=True)

plt.imshow(defocus_FI, cmap="bwr", norm=mcolors.CenteredNorm())

plt.figure()
plt.imshow(J[:, -1].reshape(256, 256), cmap="bwr", norm=mcolors.CenteredNorm())
plt.colorbar()
# %%
