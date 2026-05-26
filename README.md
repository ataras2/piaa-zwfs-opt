# PIAA-ZWFS Optimization

This repository contains the code used to simulate, optimize, and analyse a PIAA-ZWFS based wavefront sensing system.

The project is intentionally small and script-driven: the main Python package lives in `piaa-zwfs-opt/`, while generated optimization outputs are written to `results/`.

## Repository Structure

### Top level

- `pyproject.toml` - project metadata and Python dependencies.
- `README.md` - project overview and repository structure.
- `piaa-zwfs-opt/` - Python source code and analysis scripts.
- `results/` - saved optimization runs, trained models, and other generated artifacts.

### Python package and scripts

The `piaa-zwfs-opt/` directory contains the code used for the simulation and analysis workflow:

- `WFSoptimize.py` - utilities for basis generation, wavelength handling, and Fisher-information calculations.
- `dlWFSoptimize.py` - core model and optimization logic for the differentiable wavefront-sensing pipeline.
- `dl_optimize_and_save.py` - command-line entry point that builds a model, runs optimization, and saves outputs to disk.
- `serialisation.py` - helpers for saving and restoring nested model structures.
- `Aspheres.py` - aspheric surface and lens-shape utilities.
- `dlMaterials.py` - material definitions and optical properties used by the model.
- `t_load_model.py` - helper script for loading a saved model or optimization result.

### Analysis and figure scripts

The remaining `fig_*.py` files are analysis scripts used to reproduce individual figures or tables from the paper:

- `fig_overview_plots.py`
- `fig_recon_errors.py`
- `fig_compare_bandwidth_dependence.py`
- `fig_source_size_dep.py`
- `fig_diff_apertures.py`
- `fig_mag_changes.py`
- `fig_maximise_high_freqs.py`
- `fig_ablation_table.py`

### Data and outputs

- `results/dl_optimizations/` contains saved optimization runs.
- Each run directory typically includes an `.asdf` file with the optimization state and a `.model` file with the saved core model.

## Notes

- The repository targets Python 3.11 or newer.  
- Dependencies are declared in `pyproject.toml`. Use `uv` to install dependencies
- The ELT pupil was generated using an internal PCS coronagraph evaluation pipeline. If you do not have access to this, you should use the `pcs_aperture_with_spiders_N2048.fits` file (and resample to the right number of pupil pixels as needed). 

