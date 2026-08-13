# %%
import os

# cpu
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import dlWFSoptimize as dlwfs
import matplotlib.pyplot as plt
import jax.numpy as jnp
import asdf
import zodiax as zdx


import jax

jax.config.update("jax_debug_nans", True)
jax.devices()

# %%
cmn_pth = os.path.join(
    "..",
    "results",
    "dl_optimizations",
    "ablation_60",
)

pth = os.path.join(cmn_pth)

PIAA_model = (
    "magellan_Fourier_basis_7_terms_Z_band_with_PIAA_annular_mask_point_source_0.0_mag"
)

core_model, asdf_pth = dlwfs.load_model(os.path.join(pth, PIAA_model))
nominal_zwfs_model, _ = dlwfs.load_model(
    os.path.join(
        pth,
        "magellan_Fourier_basis_7_terms_Z_band_no_optimization_1.06_loD_init_eval_diam",
    )
)
nominal_2zwfs_model, _ = dlwfs.load_model(
    os.path.join(
        pth,
        "magellan_Fourier_basis_7_terms_Z_band_no_optimization_2.0_loD_init_eval_diam",
    )
)

with asdf.open(asdf_pth) as opt_notes:
    loss_type = opt_notes["config"]["optimization"]["loss_type"]
    pupil_diameter = opt_notes["config"]["aperture"]["pupil_diameter"]
    o = opt_notes.tree


# %%
# n_terms = 45
n_terms = 15

basis = dlwfs.make_zernike_basis(
    wf_npixels=int(core_model.get("tel.wf_npixels")),
    sim_diam=core_model.get("tel.diameter"),
    pupil_diameter=pupil_diameter,
    n_zernikes=n_terms,
)

core_model = core_model.set(
    [
        "tel.BasisLayer.basis",
        "tel.BasisLayer.coefficients",
        "params.phase_coefficients",
    ],
    [basis, jnp.zeros(len(basis)), jnp.zeros(len(basis))],
)

nominal_zwfs_model = nominal_zwfs_model.set(
    [
        "tel.BasisLayer.basis",
        "tel.BasisLayer.coefficients",
        "params.phase_coefficients",
    ],
    [basis, jnp.zeros(len(basis)), jnp.zeros(len(basis))],
)

nominal_2zwfs_model = nominal_2zwfs_model.set(
    [
        "tel.BasisLayer.basis",
        "tel.BasisLayer.coefficients",
        "params.phase_coefficients",
    ],
    [basis, jnp.zeros(len(basis)), jnp.zeros(len(basis))],
)

aperture = core_model.get("tel.TransmissiveLayer.transmission")
# %%
ideal_model = dlwfs.generate_idealwfs_optics(core_model)
plt.imshow(ideal_model.img())

# %%
from abc import ABC, abstractmethod
import optimistix as optx
import equinox as eqx

around_params = {
    "phase_coefficients": jnp.zeros(len(basis)),
}


def rms_opd(opd, mask):
    # return jnp.sqrt(jnp.mean((opd[mask]) ** 2))
    # return jnp.sqrt(jnp.nanmean(jnp.where(mask, opd, jnp.nan)) ** 2)
    masked_opd = jnp.where(mask, opd, 0)
    n_valid = jnp.sum(mask)
    return jnp.sqrt(jnp.sum(masked_opd**2) / n_valid)


class Reconstructor(ABC):
    def __init__(self, core_model):
        self.core_model = core_model

    @abstractmethod
    def reconstruct(self, data):
        pass

    @staticmethod
    def plot_actual_and_diff(
        actual_opd,
        reconstructed_opd,
        aperture,
        to_rad=1 / core_model.get("tel.wavelengths")[1] * 2 * jnp.pi,
    ):
        bounds = max([jnp.abs(actual_opd).max(), jnp.abs(reconstructed_opd).max()])

        if to_rad is not None:
            bounds *= to_rad
            reconstructed_opd *= to_rad
            actual_opd *= to_rad

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(actual_opd, vmin=-bounds, vmax=bounds, cmap="bwr")
        plt.title(f"Actual OPD ({rms_opd(actual_opd, aperture):.2e} rad RMS)")
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(reconstructed_opd, vmin=-bounds, vmax=bounds, cmap="bwr")
        plt.title(
            f"Reconstructed OPD ({rms_opd(reconstructed_opd, aperture):.2e} rad RMS)"
        )
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(
            actual_opd - reconstructed_opd, vmin=-bounds, vmax=bounds, cmap="bwr"
        )
        plt.title(
            f"Difference ({rms_opd(actual_opd - reconstructed_opd, aperture):.2e} rad RMS)"
        )
        plt.colorbar()


class LinearReconstructor(Reconstructor):
    def __init__(self, core_model):
        super().__init__(core_model)
        J = core_model.compute_J(around_params)
        self.interaction_matrix = J["phase_coefficients"].reshape(
            -1, J["phase_coefficients"].shape[-1]
        )

        self.ref_img = core_model.img()
        self._pinv_interaction_matrix = jnp.linalg.pinv(self.interaction_matrix)

    def reconstruct(self, data):
        residual_img = data - self.ref_img
        reconstructed_opd_coeffs = (
            self._pinv_interaction_matrix @ residual_img.flatten()
        )
        return reconstructed_opd_coeffs


class NonLinearReconstructor(Reconstructor):
    def __init__(self, core_model, solver):
        super().__init__(core_model)
        self.solver = solver

    def reconstruct(self, data):
        def log_like(core_model, data):
            theory = core_model.img()
            theory = jnp.clip(theory, 1e-12, None)
            return -jax.scipy.stats.poisson.logpmf(
                data,
                theory,
            ).sum()

        init_guess = jnp.zeros_like(self.core_model.get("params.phase_coefficients"))
        init_model = self.core_model.set(
            "params.phase_coefficients",
            init_guess,
        )

        core_model, res = dlwfs.optx_minimize_general(
            init_model,
            ["params.phase_coefficients"],
            self.solver,
            eqx.filter_jit(lambda model: log_like(model, data)),
            max_steps=2048,
        )

        # # also start from linear reconstructor?
        # init_guess = LinearReconstructor(self.core_model).reconstruct(data)
        # init_model = self.core_model.set(
        #     "params.phase_coefficients",
        #     init_guess,
        # )

        # core_model_lin, res = dlwfs.optx_minimize_general(
        #     init_model,
        #     ["params.phase_coefficients"],
        #     self.solver,
        #     eqx.filter_jit(lambda model: log_like(model, data)),
        #     max_steps=2048,
        # )

        # reconstructed_opd_coeffs = jax.lax.cond(
        #     log_like(core_model, data) < log_like(core_model_lin, data),
        #     lambda: core_model.get("params.phase_coefficients"),
        #     lambda: core_model_lin.get("params.phase_coefficients"),
        # )

        reconstructed_opd_coeffs = core_model.get("params.phase_coefficients")

        return reconstructed_opd_coeffs

# some got reset to floats for some reason
bool_params = [
    "tel.TransmissiveLayer.normalise",
    "tel.BasisLayer.as_phase",
]

for param in bool_params:
    ideal_model = ideal_model.set({param: bool(ideal_model.get(param))})

models = {
    "core_model": core_model,
    "ideal_model": ideal_model,
    "nominal_zwfs_model": nominal_zwfs_model,
    "nominal_2zwfs_model": nominal_2zwfs_model,
}

# rtors = {name: LinearReconstructor(m) for name, m in models.items()}

# rtors_nonlin = {
#     name: NonLinearReconstructor(
#         m,
#         optx.LBFGS(
#             atol=1e-5,
#             rtol=1e-5,
#         ),
#     )
#     for name, m in models.items()
# }

# %%
import hcipy
import dLux.utils as dlu


batch_size = 3

n_screens_per_ptv = 10
# ptvs = [100e-9, 500e-9, 1_000e-9]
ptvs = jnp.logspace(-9, -5.4, batch_size * 10)
hc_pup_grid = hcipy.make_pupil_grid(
    core_model.get("tel.wf_npixels"), core_model.get("tel.diameter")
)

screens = []

for ptv in ptvs:
    for i in range(n_screens_per_ptv):
        err = hcipy.make_power_law_error(
            hc_pup_grid,
            ptv,
            diameter=pupil_diameter,
            exponent=-3.0,
        )
        screens.append(err.reshape(hc_pup_grid.shape))
screens = jnp.array(screens)

batched_screens = screens.reshape(-1, batch_size, screens.shape[-2], screens.shape[-1])

basis = core_model.get("tel.BasisLayer.basis")

# 1. Define and JIT the base projection function
basis_project = lambda x: dlwfs.project_onto_basis(x, basis)
basis_project = jax.jit(basis_project)

# 2. Create a vmapped version that processes a single batch of size `batch_size`
batch_project_vmap = jax.vmap(basis_project)

# 3. Use jax.lax.map to sequentially map over the leading axis of batched_screens
# batched_screens has shape (num_batches, batch_size, H, W)
# lax.map yields shape (num_batches, batch_size, num_modes)
proj_screens_batched = jax.lax.map(batch_project_vmap, batched_screens)

# 4. Flatten the batches back to a single 2D array of shape (total_screens, num_modes)
# Using .reshape() is cleaner and faster than jnp.concatenate here!
proj_screens = proj_screens_batched.reshape(-1, proj_screens_batched.shape[-1])

# %%
idx = 20

plt.subplot(131)
plt.imshow(screens[idx])
plt.title("Original Screen")
plt.colorbar()

reproj_screen = dlu.eval_basis(basis, proj_screens[idx])
plt.subplot(132)
plt.imshow(reproj_screen)
plt.title("Reprojected Screen")
plt.colorbar()

plt.subplot(133)
plt.imshow(screens[idx] - reproj_screen)
plt.title("Difference")
plt.colorbar()

# %%
import jax.random as jr
from tqdm.auto import tqdm

estimated_coeffs = {
    "linear": {},
    "nonlinear": {},
}

for name, model in models.items():
    # model = models[name]
    print(f"Reconstructing with {name} model...")

    rtor = LinearReconstructor(model)
    rtor_nonlin = NonLinearReconstructor(
        model,
        optx.LBFGS(
            atol=1e-5,
            rtol=1e-5,
        ),
    )

    lin_recon_coeffs = []
    nonlin_recon_coeffs = []

    # for batch in proj_screens_batched:
    for batch in tqdm(proj_screens_batched):
        imgs = jax.vmap(
            lambda coeffs: model.set("params.phase_coefficients", coeffs).img()
        )(batch)
        imgs = jr.poisson(jr.key(42), imgs)
        recon_coeffs_batch = jax.vmap(rtor.reconstruct)(imgs)
        lin_recon_coeffs.append(recon_coeffs_batch)

        recon_coeffs_batch_nonlin = jax.vmap(rtor_nonlin.reconstruct)(imgs)
        nonlin_recon_coeffs.append(recon_coeffs_batch_nonlin)

    lin_recon_coeffs = jnp.concatenate(lin_recon_coeffs, axis=0)
    nonlin_recon_coeffs = jnp.concatenate(nonlin_recon_coeffs, axis=0)

    estimated_coeffs["linear"][name] = lin_recon_coeffs
    estimated_coeffs["nonlinear"][name] = nonlin_recon_coeffs


# %%
def error(actual_coeffs, estimated_coeffs, basis):
    actual_opd = dlu.eval_basis(basis, actual_coeffs)
    estimated_opd = dlu.eval_basis(basis, estimated_coeffs)
    return rms_opd(actual_opd - estimated_opd, mask=aperture > 0)


error(proj_screens[0], estimated_coeffs["linear"]["core_model"][0], basis)

# %%
error_jitted = jax.jit(lambda x, y: error(x, y, basis))

recon_errors = {
    "linear": {},
    "nonlinear": {},
}

for name in models.keys():
    # errors = jax.vmap(error_jitted, in_axes=(0, 0))(
    #     proj_screens, jnp.array(estimated_coeffs[name])
    # )
    # recon_errors[name] = errors

    for recon_type in recon_errors.keys():
        errors = jax.vmap(error_jitted, in_axes=(0, 0))(
            proj_screens, jnp.array(estimated_coeffs[recon_type][name])
        )
        recon_errors[recon_type][name] = errors

# %%

rms_ins = jax.vmap(lambda x: rms_opd(dlu.eval_basis(basis, x), aperture > 0))(
    proj_screens
)
# %%

save_pth = os.path.join(
    cmn_pth,
    "recon_errors_asdf",
    f"{n_terms}_terms_" + PIAA_model + "_recon_errors.asdf",
)

import numpy as np

# convert everything from jax numpy to regular numpy for asdf saving
recon_errors = {
    "linear": {k: np.array(v) for k, v in recon_errors["linear"].items()},
    "nonlinear": {k: np.array(v) for k, v in recon_errors["nonlinear"].items()},
}
estimated_coeffs = {
    "linear": {k: np.array(v) for k, v in estimated_coeffs["linear"].items()},
    "nonlinear": {k: np.array(v) for k, v in estimated_coeffs["nonlinear"].items()},
}
proj_screens = np.array(proj_screens)
basis = [np.array(b) for b in basis]

with asdf.AsdfFile(
    {
        "recon_errors": recon_errors,
        "estimated_coeffs": estimated_coeffs,
        "proj_screens": proj_screens,
        "basis": basis,
        "rms_ins": np.array(rms_ins),
    }
) as af:
    af.write_to(save_pth)

# %%
