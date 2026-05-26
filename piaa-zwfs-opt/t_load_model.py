# %%
import serialisation
from dlWFSoptimize import ModelCore

# pth = "results/core_model_2024-10-01.pkl"
pth = "results/dl_optimizations/ablation/magellan_Fourier_basis_7_terms_Z_band_no_optimization_1.06_loD_init_eval_diam/core_model_20260526_1340.model"

print(f"Loading model from {pth}...")
core_model: ModelCore = serialisation.deserialise(pth)
print("Model loaded successfully.")

print(core_model.loss())

# %%

pth = "results/dl_optimizations/ablation/magellan_Fourier_basis_7_terms_Z_band_no_optimization_1.06_loD_init_eval_diam/"

import dlWFSoptimize

print(f"Loading model from {pth}...")
core_model: ModelCore = dlWFSoptimize.load_model(pth)[0]
print("Model loaded successfully.")

print(core_model.loss())
