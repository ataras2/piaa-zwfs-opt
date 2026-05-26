# %%

import os
import argparse

# # Must be set before JAX is imported!
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# # 64 bit jax
# os.environ["JAX_ENABLE_X64"] = "True"
import jax.numpy as np
import dlWFSoptimize as dlwfs
import matplotlib.pyplot as plt
import zodiax as zdx
import numpy as onp
import asdf
import optimistix as optx

# get the date for the save name
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="DL WFS optimization and save script")

    # General/debug/save
    parser.add_argument(
        "--debug_plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable debug plotting",
    )
    parser.add_argument(
        "--save_pth",
        type=str,
        default=os.path.join("..", "results", "dl_optimizations"),
        help="Root folder to save results and model",
    )

    # Simulation parameters
    parser.add_argument("--target_wf_npixels", type=int, default=512)
    parser.add_argument("--det_pix_over_pupil", type=int, default=30)
    parser.add_argument("--num_airy", type=float, default=2.5)
    parser.add_argument("--q", type=int, default=32, help="Pixels per loD")
    parser.add_argument("--n_wavels", type=int, default=3)
    parser.add_argument(
        "--waveband", type=lambda s: None if s.lower() == "none" else s, default="Z"
    )
    parser.add_argument(
        "--central_wavel",
        type=lambda s: None if s.lower() == "none" else float(s),
        default=None,
    )
    parser.add_argument(
        "--bandwidth",
        type=lambda s: None if s.lower() == "none" else float(s),
        default=None,
    )
    parser.add_argument("--Nph_per_sec", type=float, default=None)
    parser.add_argument(
        "--magnitude",
        type=lambda s: None if s.lower() == "none" else float(s),
        default=0.0,
    )
    parser.add_argument("--exposure_time", type=float, default=1.0e-3)
    parser.add_argument("--system_throughput", type=float, default=0.3)
    parser.add_argument("--read_noise", type=float, default=0.5)

    # Aberration basis parameters
    parser.add_argument(
        "--basis_type",
        type=str,
        default="Fourier",
        choices=["Fourier", "Zernike"],
    )
    parser.add_argument("--phase_rms", type=float, default=1e-6)
    parser.add_argument("--amp_rms", type=float, default=None)
    parser.add_argument("--n_terms", type=int, default=7)
    parser.add_argument("--basis_min_cycles", type=float, default=1.0e-2)
    parser.add_argument("--basis_max_cycles", type=float, default=8.0)
    parser.add_argument("--basis_trig_type", type=str, default="both")

    # Aperture parameters
    parser.add_argument("--oversample_aperture", type=int, default=4)
    parser.add_argument("--aperture_name", type=str, default="magellan")
    parser.add_argument("--min_sim_diameter_frac", type=float, default=1.35)

    # PIAA-ZWFS parameters
    parser.add_argument(
        "--has_PIAA",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable PIAA optics",
    )
    parser.add_argument(
        "--phase_mask_basis_name",
        type=str,
        default="circle",
        choices=["annular", "circle"],
    )
    parser.add_argument("--n_zwfs_rings", type=int, default=64)
    parser.add_argument("--final_D", type=float, default=10e-3)
    parser.add_argument("--piaa_material", type=str, default="sio2")
    parser.add_argument("--phase_mask_material", type=str, default="sio2")
    parser.add_argument("--d", type=float, default=100e-3)
    parser.add_argument("--init_PIAA_eval_dot_diam", type=float, default=2.0)
    parser.add_argument("--n_sag_terms", type=int, default=8)
    parser.add_argument("--init_zernike_dot_diam", type=float, default=2.0)

    # Optimization parameters
    parser.add_argument(
        "--optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run optimization or just build the system",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cov_trace_normed",
        choices=["cov_trace_normed", "neg_FI_trace_normed"],
    )
    parser.add_argument("--n_iter", type=int, default=int(3e2))

    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)

    parser.add_argument(
        "--source_diameter_loD",
        type=float,
        default=0.0,
        help="Diameter of resolved source in loD",
    )

    return parser.parse_args()


# %%
args = parse_args()

debug_plots = args.debug_plots

save_pth = args.save_pth

date_str = datetime.now().strftime("%Y%m%d_%H%M")
save_fname = f"optimization_{date_str}.asdf"

##### Simulation parameters ##########
target_wf_npixels = args.target_wf_npixels
det_pix_over_pupil = args.det_pix_over_pupil
num_airy = (
    args.num_airy
)  # number of airy rings to cover in the focal plane (radius units)
q = args.q  # pixels per loD

n_wavels = args.n_wavels
waveband = args.waveband
central_wavel = args.central_wavel
bandwidth = args.bandwidth
if (central_wavel is not None) and (bandwidth is not None):
    wavels = dlwfs.get_wavels(
        n_wavels, central_wavel=central_wavel, bandwidth=bandwidth
    )
elif waveband is not None:
    wavels = dlwfs.get_wavels(n_wavels, waveband)
else:
    raise ValueError("Must specify either waveband or (central_wavel and bandwidth).")
# wavels = np.array([wavels[len(wavels) // 2]])

print("Using wavelengths (nm):", wavels * 1e9)

# Nph_per_sec = 1e20  # total number of photons
# magnitude = None
# OR:
Nph_per_sec = args.Nph_per_sec
magnitude = args.magnitude  # stellar magnitude
exposure_time = args.exposure_time
system_throughput = args.system_throughput

read_noise = args.read_noise  # e- rms

##### Aberration basis parameters ##########
basis_type = args.basis_type  # "Zernike" or "Fourier"
phase_rms = args.phase_rms
amp_rms = args.amp_rms  # 0.01  # 1%
n_terms = args.n_terms
basis_kwargs = {
    "min_cycles": args.basis_min_cycles,
    "max_cycles": args.basis_max_cycles,
    "trig_type": args.basis_trig_type,
}

##### aperture parameters ##########
oversample_aperture = args.oversample_aperture
aperture_name = args.aperture_name
pupil_diameter = dlwfs.aperture_name_to_pupil_diameter(aperture_name)
min_sim_diameter_frac = (
    args.min_sim_diameter_frac
)  # fraction of pupil diameter that must be simulated

##### PIAA-ZWFS parameters ##########
has_PIAA = args.has_PIAA
phase_mask_basis_name = args.phase_mask_basis_name  # annular or circle
n_zwfs_rings = args.n_zwfs_rings
final_D = args.final_D  # final beam diameter
piaa_material = args.piaa_material
phase_mask_material = args.phase_mask_material
d = args.d  # distance between PIAA lenses
init_PIAA_eval_dot_diam = args.init_PIAA_eval_dot_diam  # loD
n_sag_terms = args.n_sag_terms
init_zernike_dot_diam = args.init_zernike_dot_diam  # loD

##### Optimization parameters ##########
optimize = args.optimize
loss_type = args.loss_type
n_iter = args.n_iter

### source option ###
source_diameter_loD = args.source_diameter_loD  # diameter in loD

# %%

if waveband is not None:
    waveband_str = waveband
else:
    waveband_str = f"{central_wavel*1e9:.0f}nm_{bandwidth*1e9:.1f}nm_bw"

# add some more information to the save path
if optimize:
    if np.isclose(source_diameter_loD, 0.0):
        source_str = "point_source"
    else:
        source_str = f"resolved_source_{source_diameter_loD:.2f}_loD"

    fname = f"{aperture_name}_{basis_type}_basis_{n_terms}_terms_{waveband_str}_band_{'with' if has_PIAA else 'no'}_PIAA_{phase_mask_basis_name}_mask_{source_str}"

    if magnitude is not None:
        fname += f"_{magnitude}_mag"

    if read_noise > 0.5:
        fname += f"_{read_noise:.1f}e_readnoise"

    if loss_type != "cov_trace_normed":
        fname += f"_{loss_type}"

    save_pth = os.path.join(
        save_pth,
        fname,
    )
else:
    save_pth = os.path.join(
        save_pth,
        f"{aperture_name}_{basis_type}_basis_{n_terms}_terms_{waveband_str}_band_no_optimization_{init_zernike_dot_diam}_loD_init_eval_diam",
    )

print("Starting simulation:", save_pth)

core_model = dlwfs.make_system(
    wavels=wavels,
    waveband=waveband,
    target_wf_npixels=target_wf_npixels,
    det_pix_over_pupil=det_pix_over_pupil,
    num_airy=num_airy,
    q=q,
    basis_type=basis_type,
    phase_rms=phase_rms,
    n_terms=n_terms,
    basis_kwargs=basis_kwargs,
    oversample_aperture=oversample_aperture,
    aperture_name=aperture_name,
    pupil_diameter=pupil_diameter,
    min_sim_diameter_frac=min_sim_diameter_frac,
    has_PIAA=has_PIAA,
    phase_mask_basis_name=phase_mask_basis_name,
    n_zwfs_rings=n_zwfs_rings,
    final_D=final_D,
    piaa_material=piaa_material,
    phase_mask_material=phase_mask_material,
    d=d,
    init_PIAA_eval_dot_diam=init_PIAA_eval_dot_diam,
    n_sag_terms=n_sag_terms,
    exposure_time=exposure_time,
    system_throughput=system_throughput,
    read_noise=read_noise,
    Nph_per_sec=Nph_per_sec,
    magnitude=magnitude,
    init_zernike_dot_diam=init_zernike_dot_diam,
)

loD = core_model.tel.wavelengths.mean() / pupil_diameter

if source_diameter_loD > 0.0:
    core_model = core_model.set(
        "tel.source",
        dlwfs.ResolvedUniformDiskSource(
            r_max=0.5 * source_diameter_loD * loD,
            wavelengths=core_model.get("tel.source.wavelengths"),
            position=core_model.get("tel.source.position"),
            flux=core_model.get("tel.source.flux"),
            n_r_integral=3,
            n_theta_integral=6,
        ),
    )

img = core_model.tel.model()
if debug_plots:
    plt.imshow(img)

# %%
if optimize:
    opt_params = [f"params.{k}" for k in core_model.params.keys()]
    # remove phase coefficients for now
    opt_params.remove("params.phase_coefficients")
    opt_params.remove("params.read_noise")
FI = core_model.compute_FI("phase_coefficients")

if debug_plots:
    plt.figure()
    plt.imshow(FI)
    plt.colorbar()

# %%
if optimize:
    atol = args.atol
    rtol = args.rtol

    solver = optx.LBFGS(
        rtol=rtol,
        atol=atol,
        verbose=frozenset({"step", "accepted", "loss", "step_size"}),
        norm=optx.rms_norm,
    )

    # parameters = [
    #     # "params.piaa1_coeffs",
    #     # "params.piaa2_coeffs",
    #     # "params.piaa1_f_rho_max",
    #     # "params.piaa2_f_rho_max",
    #     # "params.phase_mask_coeffs",
    #     "params.phase_mask_thickness",
    #     "params.phase_mask_diameter",
    # ]
    parameters = []

    if has_PIAA:
        parameters += [
            "params.piaa1_coeffs",
            "params.piaa2_coeffs",
            "params.piaa1_f_rho_max",
            "params.piaa2_f_rho_max",
        ]
    if phase_mask_basis_name == "annular":
        parameters += [
            "params.phase_mask_coeffs",
        ]
    elif phase_mask_basis_name == "circle":
        parameters += [
            "params.phase_mask_thickness",
            "params.phase_mask_diameter",
        ]

    core_model, res = dlwfs.optx_minimize(
        core_model,
        parameters,
        solver,
        loss_type,
        max_steps=5000,
    )


# %%
rhos = np.linspace(0, final_D / 2, 500)
if has_PIAA:
    tscope = core_model._update_tel()

    sag1 = tscope.get("PIAAset_0.piaa_lens_1.asphere_profile").sag(rhos)
    sag2 = -tscope.get("PIAAset_0.piaa_lens_2.asphere_profile").sag(rhos)

    plt.figure()
    plt.plot(rhos * 1e3, sag1 * 1e6, label="PIAA Lens 1")
    plt.plot(rhos * 1e3, sag2 * 1e6, label="PIAA Lens 2")
    plt.xlabel("Radius (mm)")
    plt.ylabel("Surface sag (um)")

# %%
if optimize:
    optim_telescope = core_model._update_tel()
    core_model = core_model.set("tel", optim_telescope)

# %%
FI = core_model.compute_FI("phase_coefficients")
if debug_plots:
    plt.figure()
    plt.imshow(FI)
    plt.colorbar()

# %%
# Create a dictionary of parameters to save
# We save all the configuration variables defined at the top of the script
config_tree = {
    "simulation": {
        "target_wf_npixels": target_wf_npixels,
        "det_pix_over_pupil": det_pix_over_pupil,
        "num_airy": num_airy,
        "q": q,
        "n_wavels": n_wavels,
        "waveband": waveband,
        "magnitude": magnitude,
        "exposure_time": exposure_time,
        "system_throughput": system_throughput,
        "read_noise": read_noise,
    },
    "aberration_basis": {
        "basis_type": basis_type,
        "phase_rms": phase_rms,
        "amp_rms": amp_rms,
        "n_terms": n_terms,
        "basis_kwargs": basis_kwargs,
    },
    "aperture": {
        "oversample_aperture": oversample_aperture,
        "aperture_name": aperture_name,
        "pupil_diameter": pupil_diameter,
        "min_sim_diameter_frac": min_sim_diameter_frac,
    },
    "piaa_zwfs": {
        "has_PIAA": has_PIAA,
        "phase_mask_basis_name": phase_mask_basis_name,
        "n_zwfs_rings": n_zwfs_rings,
        "final_D": final_D,
        "piaa_material": piaa_material,
        "phase_mask_material": phase_mask_material,
        "d": d,
        "init_PIAA_eval_dot_diam": init_PIAA_eval_dot_diam,
        "n_sag_terms": n_sag_terms,
    },
}

if optimize:
    config_tree["optimization"] = {
        "optimize": optimize,
        "loss_type": loss_type,
        "n_iter": n_iter,
    }


# Helper to convert JAX arrays to numpy for ASDF
def to_cpu_numpy(obj):
    if hasattr(obj, "device_buffer") or hasattr(obj, "sharding"):  # JAX array
        return onp.array(obj)
    elif isinstance(obj, dict):
        return {k: to_cpu_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu_numpy(v) for v in obj]
    return obj


# Also save the results
results_tree = {
    "final_params": core_model.params,
    "FI": core_model.compute_FI("phase_coefficients"),
    "loss": core_model.loss(loss_type=loss_type),
}

# Combine into one tree and convert to numpy
tree = {"config": config_tree, "results": results_tree}
tree = to_cpu_numpy(tree)

# Save to ASDF file
if not os.path.exists(save_pth):
    os.makedirs(save_pth)

filename = os.path.join(save_pth, save_fname)
with asdf.AsdfFile(tree) as af:
    af.write_to(filename)
    print(f"Saved configuration and results to {filename}")

# also save the model file using zdx.experimental.serialise
# zdx.experimental.serialise(
#     os.path.join(save_pth, f"core_model_{date_str}.model"),
#     core_model,
# )

import pickle

with open(os.path.join(save_pth, f"core_model_{date_str}.pkl"), "wb") as f:
    pickle.dump(core_model, f)


print("Saved core model.")
# %%

# %%
