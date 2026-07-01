# %%
import hcipy
import jax.numpy as np
import matplotlib.pyplot as plt
import dlWFSoptimize as dlwfs
import dlMaterials
import dLux as dl
import dLux.utils as dlu
import Aspheres
import WFSoptimize

# %%

##### Simulation parameters ##########
target_wf_npixels = 512
det_pix_over_pupil = 30
num_airy = 2.5  # number of airy rings to cover in the focal plane (radius units)
q = 32  # pixels per loD

n_wavels = 3
waveband = "Z"
wavels = dlwfs.get_wavels(n_wavels, waveband)
# wavels = np.array([wavels[len(wavels) // 2]])


##### aperture parameters ##########
oversample_aperture = 1
aperture_name = "magellan"
pupil_diameter = 6.5  # metres
min_sim_diameter_frac = 1.35  # fraction of pupil diameter that must be simulated

##### PIAA-ZWFS parameters ##########
has_PIAA = True
# has_PIAA = False
# phase_mask_basis_name = "circle"  # annular or circle
phase_mask_basis_name = "annular"  # annular or circle
n_zwfs_rings = 64
final_D = 10e-3  # final beam diameter
piaa_material = "sio2"
phase_mask_material = "sio2"
d = 100e-3  # distance between PIAA lenses
init_PIAA_eval_dot_diam = 2.0  # loD
n_sag_terms = 8
init_zernike_dot_diam = 2.0  # loD

debug_plots = True

# %%
central_wavel = wavels[len(wavels) // 2]
loD = central_wavel / (final_D)  # radians

magnifier = dlwfs.Magnifier(final_D / pupil_diameter)
n_init_fit = dlMaterials.material_factory(piaa_material).n(wavels[len(wavels) // 2])

#### Aperture stuff ####
min_sim_diameter = min_sim_diameter_frac * pupil_diameter  # metres
sim_diameter, wf_npixels, det_downsample = dlwfs.sim_dims(
    pupil_diameter, det_pix_over_pupil, min_sim_diameter, target_wf_npixels
)

if debug_plots:
    sample_wf = dl.Wavefront(
        npixels=wf_npixels,
        diameter=sim_diameter,
        wavelength=central_wavel,
    )

aperture, hcaperture = dlwfs.make_aperture(
    aperture_name, sim_diameter, wf_npixels, oversample_aperture
)


# %%
hc_pupil_grid = hcipy.make_pupil_grid(wf_npixels, diameter=sim_diameter)
m = hcipy.Magnifier(final_D / pupil_diameter)
scaled_pupil_grid = m.get_output_grid(hc_pupil_grid, central_wavel)
wf = WFSoptimize.solve_mask_eigenvalue_problem(
    pupil_grid=hc_pupil_grid,
    mask_diameter=init_PIAA_eval_dot_diam,
    wavelength=float(central_wavel),
    telescope_pupil=hcaperture,
    telescope_diameter=pupil_diameter,
    magnifier=m,
)
popt, pcov, gaussian = WFSoptimize.fit_gaussian(wf, scaled_pupil_grid)
z, Z, h, h_inv = WFSoptimize.solve_PIAA_sag_functions(n_init_fit, d, final_D, popt)

sample_wf = dl.Wavefront(
    npixels=wf_npixels,
    diameter=sim_diameter,
    wavelength=central_wavel,
)
sample_wf = sample_wf.set(
    ["phasor"], [hcaperture.reshape(hc_pupil_grid.shape).astype(np.complex128)]
)
sample_wf = sample_wf.normalise()

sample_wf = magnifier.apply(sample_wf)

init_wf = sample_wf

rhos = np.linspace(0, final_D / 2, 500)
if debug_plots:
    r_vals = dlu.cart2polar(sample_wf.coordinates())[0][:, wf_npixels // 2]

    plt.figure()
    plt.plot(r_vals * 1e3, z(r_vals) * 1e6)
    plt.plot(r_vals * 1e3, Z(h(r_vals)) * 1e6)
    plt.xlabel("Radius (mm)")
    plt.ylabel("Surface sag (um)")
    plt.title("PIAA surface profiles")

sag1_profile = -z(rhos)
sag2_profile = Z(h(rhos))

sag1_asphere = Aspheres.MildAsphere.from_target_sag(
    target_rhos=rhos,
    target_sags=sag1_profile,
    n_terms=n_sag_terms,
    scale=1.0e-6,  # scale to microns
)

sag2_asphere = Aspheres.MildAsphere.from_target_sag(
    target_rhos=rhos,
    target_sags=sag2_profile,
    n_terms=n_sag_terms,
    scale=1.0e-6,  # scale to microns
)

piaa_lens_1 = Aspheres.AsphericLens(
    asphere_profile=sag1_asphere,
    material=dlMaterials.material_factory(piaa_material),
    coords=sample_wf.coordinates(),
    decentre_scale=1e-6,
)

piaa_lens_2 = Aspheres.AsphericLens(
    asphere_profile=sag2_asphere,
    material=dlMaterials.material_factory(piaa_material),
    coords=sample_wf.coordinates(),
    decentre_scale=1e-6,
)

inv_piaa_lens_1 = Aspheres.AsphericLens(
    asphere_profile=sag1_asphere,
    material=dlMaterials.material_factory(piaa_material),
    coords=sample_wf.coordinates(),
    is_inverse=True,
    decentre_scale=1e-6,
)

inv_piaa_lens_2 = Aspheres.AsphericLens(
    asphere_profile=sag2_asphere,
    material=dlMaterials.material_factory(piaa_material),
    coords=sample_wf.coordinates(),
    is_inverse=True,
    decentre_scale=1e-6,
)

piaa_set = Aspheres.PIAAset(
    piaa_lens_1=piaa_lens_1,
    piaa_lens_2=piaa_lens_2,
    distance=d,
)
inv_piaa_set = Aspheres.PIAAset(
    piaa_lens_1=inv_piaa_lens_1,
    piaa_lens_2=inv_piaa_lens_2,
    distance=d,
    is_inverse=True,
)

if debug_plots:
    # show the PIAA profiles after prop
    plt.figure()
    dlwfs.imshow_wavefront(sample_wf)
    sample_wf = piaa_set.apply(sample_wf)
    plt.figure()
    dlwfs.imshow_wavefront(sample_wf)
    plt.figure()
    sample_wf = inv_piaa_set.apply(sample_wf)
    dlwfs.imshow_wavefront(sample_wf)

# %%
piaa_set = piaa_set.set("piaa_lens_1.decentre", np.array([1000, 0]))
piaa_set = piaa_set.set("piaa_lens_2.decentre", np.array([1000, 0]))

sample_wf = init_wf
plt.figure()
dlwfs.imshow_wavefront(sample_wf)
sample_wf = piaa_set.apply(sample_wf)
plt.figure()
dlwfs.imshow_wavefront(sample_wf)
plt.figure()
sample_wf = inv_piaa_set.apply(sample_wf)
dlwfs.imshow_wavefront(sample_wf)
# %%
