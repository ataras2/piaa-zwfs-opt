"""Shared configuration, optimization, and output helpers for DL WFS runs."""

import os

import asdf
import jax.numpy as np
import numpy as onp
import optimistix as optx
import zodiax as zdx

import dlWFSoptimize as dlwfs


def add_basis_arguments(parser):
    parser.add_argument(
        "--basis_type", choices=["Fourier", "Zernike"], default="Fourier"
    )
    parser.add_argument("--phase_rms", type=float, default=1e-6)
    parser.add_argument("--amp_rms", type=float, default=None)
    parser.add_argument("--n_terms", type=int, default=7)
    parser.add_argument("--basis_min_cycles", type=float, default=1.0e-2)
    parser.add_argument("--basis_max_cycles", type=float, default=8.0)
    parser.add_argument("--basis_trig_type", type=str, default="both")


def basis_config_from_args(args):
    return {
        "basis_type": args.basis_type,
        "phase_rms": args.phase_rms,
        "amp_rms": args.amp_rms,
        "n_terms": args.n_terms,
        "basis_kwargs": {
            "min_cycles": args.basis_min_cycles,
            "max_cycles": args.basis_max_cycles,
            "trig_type": args.basis_trig_type,
        } if args.basis_type == "Fourier" else {},
    }


def optimization_folder_name(
    aperture_name, basis_type, n_terms, waveband_str, has_PIAA,
    phase_mask_basis_name, source_diameter_loD, magnitude, read_noise, loss_type,
):
    if np.isclose(source_diameter_loD, 0.0):
        source_str = "point_source"
    else:
        source_str = f"resolved_source_{source_diameter_loD:.2f}_loD"
    name = f"{aperture_name}_{basis_type}_basis_{n_terms}_terms_{waveband_str}_band_{'with' if has_PIAA else 'no'}_PIAA_{phase_mask_basis_name}_mask_{source_str}"
    if magnitude is not None:
        name += f"_{magnitude}_mag"
    if read_noise > 0.5:
        name += f"_{read_noise:.1f}e_readnoise"
    if loss_type != "cov_trace_normed":
        name += f"_{loss_type}"
    return name


def optimize_model(core_model, has_PIAA, phase_mask_basis_name, loss_type, atol, rtol, n_iter):
    solver = optx.LBFGS(
        rtol=rtol,
        atol=atol,
        verbose=frozenset({"step", "accepted", "loss", "step_size"}),
        norm=optx.rms_norm,
    )
    parameters = []
    if has_PIAA:
        parameters += [
            "params.piaa1_coeffs",
            "params.piaa2_coeffs",
            "params.piaa1_f_rho_max",
            "params.piaa2_f_rho_max",
        ]
    if phase_mask_basis_name == "annular":
        parameters += ["params.phase_mask_coeffs"]
    elif phase_mask_basis_name == "circle":
        parameters += ["params.phase_mask_thickness", "params.phase_mask_diameter"]
    else:
        raise ValueError(f"Unknown phase mask basis {phase_mask_basis_name}")
    return dlwfs.optx_minimize(
        core_model, parameters, solver, loss_type, max_steps=n_iter
    )


def to_cpu_numpy(obj):
    if hasattr(obj, "device_buffer") or hasattr(obj, "sharding"):
        return onp.array(obj)
    if isinstance(obj, dict):
        return {k: to_cpu_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu_numpy(v) for v in obj]
    return obj


def save_optimization(core_model, config_tree, save_pth, loss_type, date_str):
    results_tree = {
        "final_params": core_model.params,
        "FI": core_model.compute_FI("phase_coefficients"),
        "loss": core_model.loss(loss_type=loss_type),
    }
    tree = to_cpu_numpy({"config": config_tree, "results": results_tree})
    os.makedirs(save_pth, exist_ok=True)
    filename = os.path.join(save_pth, f"optimization_{date_str}.asdf")
    with asdf.AsdfFile(tree) as af:
        af.write_to(filename)
    print(f"Saved configuration and results to {filename}")
    zdx.experimental.serialise(
        os.path.join(save_pth, f"core_model_{date_str}.model"), core_model
    )
    print("Saved core model.")
