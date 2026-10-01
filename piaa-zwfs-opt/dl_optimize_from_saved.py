"""Optimize the newest saved model, using the standard aberration basis defaults."""

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import asdf

import dlWFSoptimize as dlwfs
from dl_optimization_common import (
    add_basis_arguments,
    basis_config_from_args,
    optimization_folder_name,
    optimize_model,
    save_optimization,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_pth", type=Path,
        help="Folder containing the saved .model files and matching .asdf files",
    )
    parser.add_argument(
        "--save_pth", type=Path,
        help="Root folder for results (default: parent of the input model folder)",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument(
        "--loss_type", choices=["cov_trace_normed", "neg_FI_trace_normed"],
        default="cov_trace_normed",
    )
    parser.add_argument("--n_iter", type=int, default=300)
    add_basis_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_pth = args.model_pth.resolve()
    if not model_pth.is_dir():
        raise NotADirectoryError(f"Model folder does not exist: {model_pth}")
    if not any(model_pth.glob("*.model")):
        raise FileNotFoundError(f"No .model files found in {model_pth}")

    core_model, asdf_pth = dlwfs.load_model(str(model_pth))
    with asdf.open(asdf_pth, lazy_load=False) as af:
        config = deepcopy(af["config"])
    print(f"Loaded model with configuration: {asdf_pth}")

    basis_config = basis_config_from_args(args)
    basis = dlwfs.make_aberration_basis(
        wf_npixels=core_model.tel.optics.wf_npixels,
        sim_diam=core_model.tel.optics.diameter,
        pupil_diameter=config["aperture"]["pupil_diameter"],
        pupil=core_model.get_aperture(),
        **{k: v for k, v in basis_config.items() if k != "amp_rms"},
    )
    core_model = core_model.update_model_basis(basis)
    config["aberration_basis"] = basis_config
    config["optimization"] = {
        "optimize": True,
        "loss_type": args.loss_type,
        "n_iter": args.n_iter,
        "atol": args.atol,
        "rtol": args.rtol,
        "starting_model": str(model_pth / (
            Path(asdf_pth).stem.replace("optimization_", "core_model_", 1) + ".model"
        )),
    }

    simulation = config["simulation"]
    piaa = config["piaa_zwfs"]
    waveband_str = simulation["waveband"]
    if waveband_str is None:
        wavels = core_model.tel.wavelengths
        central_wavel = simulation.get("central_wavel")
        bandwidth = simulation.get("bandwidth")
        if central_wavel is None:
            central_wavel = float(wavels[len(wavels) // 2])
        if bandwidth is None:
            bandwidth = float(wavels.max() - wavels.min())
        waveband_str = f"{central_wavel*1e9:.0f}nm_{bandwidth*1e9:.1f}nm_bw"
    source_diameter = simulation.get("source_diameter_loD")
    if source_diameter is None:
        loD = core_model.tel.wavelengths.mean() / config["aperture"]["pupil_diameter"]
        source_diameter = float(2 * getattr(core_model.tel.source, "r_max", 0.0) / loD)
    folder = optimization_folder_name(
        config["aperture"]["aperture_name"], args.basis_type, args.n_terms,
        waveband_str, piaa["has_PIAA"], piaa["phase_mask_basis_name"],
        source_diameter, simulation["magnitude"], simulation["read_noise"], args.loss_type,
    )
    save_pth = (args.save_pth or model_pth.parent) / folder
    print(f"Starting optimization: {save_pth}")
    core_model, _ = optimize_model(
        core_model, piaa["has_PIAA"], piaa["phase_mask_basis_name"],
        args.loss_type, args.atol, args.rtol, args.n_iter,
    )
    core_model = core_model.set("tel", core_model._update_tel())
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_optimization(core_model, config, save_pth, args.loss_type, date_str)


if __name__ == "__main__":
    main()
