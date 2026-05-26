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
import asdf
import numpy as onp
import zodiax as zdx

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
    "ablation_60",
)
# find all folders in pth
all_folders = [f for f in os.listdir(pth) if os.path.isdir(os.path.join(pth, f))]
all_folders
# %%
models = {f: dlwfs.load_model(os.path.join(pth, f))[0] for f in all_folders}

# %%
m = models[all_folders[0]]

loss = "cov_trace_normed"

ideal_wfs = dlwfs.generate_idealwfs_optics(m)

optimal_loss = ideal_wfs.loss(loss)

print(f"Ideal WFS loss: {optimal_loss*1e3:.3g}")

order = [
    "1.06",
    "2.0",
    "no_PIAA_circle",
    "with_PIAA_circle",
    "no_PIAA_annular",
    "with_PIAA_annular",
]

# print the loss of each mode in order, and the sub-optimality percentage

for key in order:
    # search for the model with key in its name
    model = None
    for fname, m in models.items():
        if key in fname:
            model = m
            break
    if model is None:
        print(f"Model with key {key} not found.")
        continue
    model_loss = model.loss(loss)
    suboptimality = (model_loss - optimal_loss) / optimal_loss * 100
    # print loss to 3 s.f. and suboptimality as percentage to 0 decimal places
    print(f"Model: {key}, {model_loss*1e3:.3g} & {suboptimality:.0f}")

# %%
