import numpy.typing
import hcipy
import numpy as np
from typing import Any
from scipy.special import jn_zeros, j0
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.integrate import quad


################################################################
# Basis utility functions
################################################################
def get_mode(
    basis: hcipy.ModeBasis,
    coeffs: np.ndarray,
):
    mode = basis.linear_combination(coeffs)
    # If the result is a list or tuple (e.g. from make_sine_basis),
    # return the first element (horizontal modes)
    if isinstance(mode, (list, tuple)):
        return mode[0]
    if mode.ndim == 2:
        return mode[0]
    return mode


def freq_basis(
    pupil_grid,
    pupil_diameter: float,
    min_cycles: float = 1e-2,
    max_cycles: float = 15.0,
    num_freqs: int = 21,
    osc_type: str = "sine",
):
    freqs = np.linspace(min_cycles, max_cycles, num_freqs) / pupil_diameter
    # basis = make_fourier_basis(pupil_grid, Grid(freqs), sort_by_energy=False)
    if osc_type == "sine":
        return freqs, hcipy.make_sine_basis(
            pupil_grid, hcipy.Grid(2 * np.pi * freqs), sort_by_energy=False
        )
    elif osc_type == "cosine":
        return freqs, hcipy.make_cosine_basis(
            pupil_grid, hcipy.Grid(2 * np.pi * freqs), sort_by_energy=False
        )
    else:
        raise ValueError(f"Unknown osc_type: {osc_type}")


def inner(a, b):
    return np.sum(a.conj() * b)


################################################################
# Wavelength stuff
################################################################

band_to_center_and_bandwidth = {
    # All values in meters
    "U": (365e-9, 66e-9),
    "B": (445e-9, 94e-9),
    "G": (464e-9, 128e-9),
    "V": (551e-9, 88e-9),
    "R": (658e-9, 138e-9),
    "I": (806e-9, 149e-9),
    "Z": (900e-9, 152e-9),
    "Y": (1020e-9, 120e-9),
    "J": (1220e-9, 213e-9),
    "H": (1630e-9, 307e-9),
    "K": (2190e-9, 390e-9),
    "L": (3450e-9, 472e-9),
    "M": (4750e-9, 460e-9),
    "N": (10500e-9, 2500e-9),
    "Q": (21000e-9, 5800e-9),
}


def get_wavels(
    num_wavels: int,
    band=None,
    min_wavel=None,
    max_wavel=None,
    central_wavel=None,
    bandwith=None,
):
    """
    Get an array of wavelengths either from a band name, min/max values or centre + bandwidth.
    Only one method should be used.
    Parameters
    ----------
    num_wavels : int
        Number of wavelengths to generate.
    band : str, optional
        Name of the band to use. If provided, min_wavel and max_wavel are ignored.
    min_wavel : float, optional
        Minimum wavelength in meters. Used if band is None.
    max_wavel : float, optional
        Maximum wavelength in meters. Used if band is None.
    central_wavel : float, optional
        Central wavelength in meters. Used if band is None.
    bandwith : float, optional
        Bandwidth in meters. Used if band is None.
    Returns
    -------
    np.ndarray
        Array of wavelengths in meters.
    """

    if band is not None:
        central_wavel, bandwith = band_to_center_and_bandwidth[band]
        min_wavel = central_wavel - bandwith / 2
        max_wavel = central_wavel + bandwith / 2
    elif central_wavel is not None and bandwith is not None:
        min_wavel = central_wavel - bandwith / 2
        max_wavel = central_wavel + bandwith / 2
    elif min_wavel is None or max_wavel is None:
        raise ValueError(
            "Either band or min/max wavelengths or central wavelength and bandwidth must be provided."
        )

    wavels = np.linspace(min_wavel, max_wavel, num_wavels)
    return wavels


################################################################
# FI calculation functions
################################################################
def compute_dEout_dalpha(
    basis: hcipy.ModeBasis,
    prop: callable,
    is_phase: bool,
    aperture_arr: hcipy.field.field.OldStyleField,
    wavelength: float,
    normalise_modes: bool = True,
    coeffs: np.ndarray | None = None,
):
    """
    Compute the derivative of the output electric field with respect to the mode coefficients.
    Parameters
    ----------
    basis : hcipy.ModeBasis
        The mode basis used to generate the perturbations.
    prop : callable
        The propagation function that takes a Wavefront and returns the propagated electric field.
    is_phase : bool
        Whether the modes represent phase perturbations (True) or amplitude perturbations (False).
    aperture_arr : hcipy.field.field.OldStyleField
        The aperture array defining the pupil.
    coeffs : np.ndarray, optional
        Coefficients for the ideal wavefront mode. If None, defaults to zeros.

    Returns
    -------
    np.ndarray
        A 2D array where each column corresponds to the derivative of the output electric field
        with respect to a mode coefficient.
    """
    if coeffs is None:
        coeffs = np.zeros(basis.num_modes)

    input_E = aperture_arr * np.exp(1j * get_mode(basis, coeffs))
    input_E = input_E / np.sum(np.abs(input_E) ** 2) ** 0.5  # Normalize input power
    ideal_wf = hcipy.Wavefront(input_E, wavelength=wavelength)

    ideal_out_E = prop(ideal_wf.electric_field, wavelength=wavelength)

    dEout_dalpha = np.zeros((basis.num_modes, ideal_out_E.size), dtype=np.complex128)

    for i in range(0, basis.num_modes):
        # coeffs = np.zeros(basis.num_modes)
        # coeffs[i] = 1.0

        # mode = get_mode(basis, coeffs)
        if basis.transformation_matrix.ndim == 3:
            mode = basis.transformation_matrix[0, :, i]
        else:
            mode = basis.transformation_matrix[:, i]

        if normalise_modes:
            normalised_mode = mode / np.std(mode[aperture_arr > 0])
        else:
            normalised_mode = mode

        if is_phase:
            diff_wf = hcipy.Wavefront(
                1j * normalised_mode * ideal_wf.electric_field, wavelength=wavelength
            )
        else:
            diff_wf = hcipy.Wavefront(
                -normalised_mode * ideal_wf.electric_field, wavelength=wavelength
            )

        diff_out_Es = prop(diff_wf.electric_field, wavelength=wavelength)

        dEout_dalpha[i] = diff_out_Es

    return ideal_out_E, dEout_dalpha


def compute_diag_sensitivity(dEout_dalpha, ideal_out):
    top = 2 * np.real(dEout_dalpha.conj() * ideal_out[:, np.newaxis])
    bottom = np.abs(ideal_out)

    J = ((top / bottom[:, np.newaxis]) ** 2).sum(axis=0)
    J = np.sqrt(J)  # per unit rms phase
    return J


def compute_fisher_information(
    dEout_all: np.ndarray,
    ideal_out: np.ndarray,
    limited="photon",
    final_npix=None,
):
    """
    Compute the Fisher Information Matrix for a given set of mode derivatives and ideal output field.
    The units are per unit rms phase, and per photon (or per read noise sigma per pixel).

    Parameters
    ----------
    dEout_all : np.ndarray
        A 2D array where each column corresponds to the derivative of the output electric field
        with respect to a mode coefficient. Shape: (n_modes, n_pixels)
    ideal_out : np.ndarray
        The ideal output electric field. Shape: (n_pixels,)
    limited : str
        The noise limitation type. Either "photon" for photon-limited or "read" for read-noise-limited.
    final_npix : int, optional
        If provided, the output pixels will be downsampled to this number of pixels
    Returns
    -------
    np.ndarray
        The Fisher Information Matrix. Shape: (n_modes, n_modes)
    """
    # dEout_all: shape (n_modes, n_pixels)
    # ideal_out: shape (n_pixels,)
    top = 2 * np.real(dEout_all.conj() * ideal_out)

    if final_npix is not None:
        # check that final_npix divides dEout_all.shape[0]
        if dEout_all.shape[-1] % final_npix != 0:
            raise ValueError("final_npix must divide the number of pixels in dEout_all")
        current_npix = ideal_out.shape[0]

        subsampled_top = np.zeros((top.shape[0], final_npix**2))
        subsample_fact = int(np.sqrt(current_npix) / final_npix)

        # downsample frac
        for i in range(top.shape[0]):
            subsampled_top[i] = hcipy.field.subsample_field(
                top[i],
                subsample_fact,
                statistic="sum",
            )

    if limited == "photon":
        bottom = np.abs(ideal_out)

        # shape: (n_pixels, n_modes)
        if final_npix is not None:
            subsampled_bottom = hcipy.field.subsample_field(
                bottom,
                subsample_fact,
                statistic="sum",
            )
            frac = subsampled_top / subsampled_bottom * subsample_fact
        else:
            frac = top / bottom

    elif limited == "read":
        frac = top

    # Fisher information matrix:
    complete_cross_terms = frac @ frac.T

    return complete_cross_terms


################################################################
# Propagators/ optical system designs
################################################################
def baseline_ZWFS_propagator(
    pupil_grid,
    dot_diam,
    pupil_diameter=1,
    reference_wavelength=1,
    phase_step=np.pi / 2,
):
    ZWFS_ideal = hcipy.ZernikeWavefrontSensorOptics(
        pupil_grid,
        phase_step=phase_step,
        phase_dot_diameter=dot_diam,
        num_pix=256,
        pupil_diameter=pupil_diameter,
        reference_wavelength=reference_wavelength,
    )

    def prop(e_in, wavelength):
        wf_in = hcipy.Wavefront(e_in, wavelength=wavelength)
        wf_out = ZWFS_ideal.forward(wf_in)
        return wf_out.electric_field

    return prop


# Base class for radial basis superpositions
class RadialBasisSuperposition:
    def __init__(
        self,
        n_modes,
        phase_dot_diameter,
        npix=128,
        loD_linear=1,
    ):
        phase_dot_diameter *= loD_linear
        self.n_modes = n_modes
        self.phase_dot_diameter = phase_dot_diameter
        self.grid = hcipy.make_uniform_grid([npix, npix], phase_dot_diameter)
        self.modes = None
        self.num_modes = None

    def linear_combination(self, coeffs):
        return np.tensordot(coeffs, self.modes, axes=1)


class RadialZernikeSuperposition(RadialBasisSuperposition):
    def __init__(
        self,
        n_modes,
        phase_dot_diameter,
        npix=128,
        loD_linear=1,
    ):
        super().__init__(n_modes, phase_dot_diameter, npix, loD_linear)
        self.modes = []
        for n in range(0, 2 * n_modes, 2):
            self.modes.append(
                hcipy.zernike(
                    n, 0, self.phase_dot_diameter, self.grid, radial_cutoff=True
                )
            )
        self.modes = np.array(self.modes)
        self.num_modes = self.modes.shape[0]


class PolynomialBasisSuperposition(RadialBasisSuperposition):
    """
    basis f(r) = r^n for n in 0, 1, ..., N-1
    """

    def __init__(
        self,
        n_modes,
        phase_dot_diameter,
        npix=128,
        loD_linear=1,
    ):
        super().__init__(n_modes, phase_dot_diameter, npix, loD_linear)
        rr = np.sqrt(self.grid.x**2 + self.grid.y**2) / (self.phase_dot_diameter / 2)
        self.modes = []
        for n in range(n_modes):
            self.modes.append(rr**n * (rr <= 1))
        self.modes = np.array(self.modes)
        self.num_modes = self.modes.shape[0]


class ChebyshevBasisSuperposition(RadialBasisSuperposition):
    """
    Basis of Chebyshev polynomials T_n(r) of the first kind on the unit disk
    """

    def __init__(
        self,
        n_modes,
        phase_dot_diameter,
        npix=128,
        loD_linear=1,
    ):
        super().__init__(n_modes, phase_dot_diameter, npix, loD_linear)
        rr = np.sqrt(self.grid.x**2 + self.grid.y**2) / (self.phase_dot_diameter / 2)
        self.modes = []
        for n in range(n_modes):
            Tn = np.cos(n * np.arccos(rr))
            Tn[rr > 1] = 0.0
            self.modes.append(Tn)
        self.modes = np.array(self.modes)
        self.num_modes = self.modes.shape[0]


class BesselBasisSuperposition(RadialBasisSuperposition):
    """
    Bessel function basis J_0(k_n r) on the unit disk, where k_n is the n-th root of J_0
    """

    def __init__(
        self,
        n_modes,
        phase_dot_diameter,
        npix=128,
        loD_linear=1,
    ):

        super().__init__(n_modes, phase_dot_diameter, npix, loD_linear)
        rr = np.sqrt(self.grid.x**2 + self.grid.y**2) / (self.phase_dot_diameter / 2)
        self.modes = []
        roots = jn_zeros(0, n_modes)
        for n in range(n_modes):
            kn = roots[n]
            J0 = j0(kn * rr)
            J0[rr > 1] = 0.0
            self.modes.append(J0)
        self.modes = np.array(self.modes)
        self.num_modes = self.modes.shape[0]


class AnnularBasisSuperposition(RadialBasisSuperposition):
    """
    Each annular ring is a constant value
    """

    def __init__(
        self,
        n_modes,
        phase_dot_diameter,
        npix=128,
        loD_linear=1,
    ):
        super().__init__(n_modes, phase_dot_diameter, npix, loD_linear)
        rr = np.sqrt(self.grid.x**2 + self.grid.y**2) / (self.phase_dot_diameter / 2)
        self.modes = []
        radial_bounds = np.linspace(0, 1, n_modes + 1)
        for n in range(n_modes):
            ring = ((rr >= radial_bounds[n]) & (rr < radial_bounds[n + 1])).astype(
                float
            )
            self.modes.append(ring)
        self.modes = np.array(self.modes)
        self.num_modes = self.modes.shape[0]


class BasisZWFSOptics:
    def __init__(
        self,
        input_grid,
        radial_basis: Any,
        basis_coeffs: np.ndarray,
        phase_dot_diameter=1.06,
        pupil_diameter=1,
        reference_wavelength=1,
    ):
        self.input_grid = input_grid
        self.output_grid = input_grid

        # Make the phase dot
        phase_dot_diameter *= reference_wavelength / pupil_diameter
        focal_grid = radial_basis.grid
        self.phase_dot = hcipy.PhaseApodizer(
            radial_basis.linear_combination(basis_coeffs) * np.pi / 2
        )

        # Make the propagator
        self.prop = hcipy.FraunhoferPropagator(input_grid, focal_grid)

    def set_phase_mask(self, phase_mask_arr):
        self.phase_dot = hcipy.PhaseApodizer(phase_mask_arr)

    def forward(self, wavefront):
        """Propagates a wavefront through the wavefront sensor.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf : Wavefront
            The output wavefront.
        """
        wf_foc = self.prop.forward(wavefront)
        wf_foc.electric_field -= self.phase_dot.forward(wf_foc).electric_field

        pup = self.prop.backward(wf_foc)
        pup.electric_field[:] = wavefront.electric_field - pup.electric_field

        return pup

    def backward(self, wavefront):
        """Propagates a wavefront backwards through the wavefront sensor.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf : Wavefront
            The output wavefront.
        """
        wf_foc = self.prop.forward(wavefront)
        wf_foc.electric_field -= self.phase_dot.backward(wf_foc).electric_field

        pup = self.prop.backward(wf_foc)
        pup.electric_field[:] = wavefront.electric_field - pup.electric_field

        return pup


################################################################
# PIAA stuff
################################################################


def solve_mask_eigenvalue_problem(
    pupil_grid,
    mask_diameter,
    wavelength,
    telescope_pupil,
    telescope_diameter,
    magnifier,
):
    """
    Solve the eigenvalue problem for the focal plane mask to find mode through the telescope
    that matches a mask of given diameter.

    Parameters
    ----------
    pupil_grid : hcipy.Field
        The pupil grid.
    mask_diameter : float
        The diameter of the focal plane mask in units of lambda/D.
    wavelength : float
        The wavelength in meters.
    telescope_pupil : hcipy.Field
        The telescope pupil function.
    telescope_diameter : float
        The diameter of the telescope in meters.
    magnifier : hcipy.optics.magnifer.Magnifier
        The magnifier used in the system.
    """
    f_number = 69  # it actually doesnt matter, get the same answer either way

    final_D = telescope_diameter * magnifier.magnification

    focal_length = f_number * final_D  # metres
    loD = wavelength / final_D  # radians
    focal_grid = hcipy.make_focal_grid(
        q=32,
        num_airy=mask_diameter * 1.2,
        reference_wavelength=wavelength,
        f_number=f_number,
    )

    mask = hcipy.make_circular_aperture(mask_diameter * loD * focal_length)(focal_grid)
    field_stop_ideal = hcipy.OccultedLyotCoronagraph(
        pupil_grid, mask, focal_plane_mask_grid=focal_grid, focal_length=focal_length
    )

    ideal_wf = telescope_pupil * np.exp(1j * np.zeros(pupil_grid.size))
    ideal_wf = hcipy.Wavefront(ideal_wf, wavelength=wavelength)
    ideal_wf.total_power = 1.0

    ideal_wf = magnifier.forward(ideal_wf)

    wf = ideal_wf.copy()
    # wf_diffs = []
    # wf_neighbour_diffs = []

    for i in range(int(2e1)):
        wf = field_stop_ideal.forward(wf)
        wf.electric_field *= telescope_pupil
        wf.total_power = 1.0
        # prev_wf = wf.copy()

        # wf_diffs.append(
        #     np.sum(np.abs(wf.electric_field - ideal_wf.electric_field) ** 2)
        # )
        # if i > 0:
        #     wf_neighbour_diffs.append(
        #         np.sum(np.abs(wf.electric_field - prev_wf.electric_field) ** 2)
        #     )

    return wf


def fit_gaussian(wf, scaled_pupil_grid):
    output_row_middle = wf.intensity.reshape(scaled_pupil_grid.shape)[
        scaled_pupil_grid.shape[0] // 2, :
    ]
    radii = scaled_pupil_grid.x.reshape(scaled_pupil_grid.shape)[
        scaled_pupil_grid.shape[0] // 2, :
    ]

    mask = output_row_middle > 0
    radii = radii[mask]
    output_row_middle = output_row_middle[mask]

    def gaussian(x, a, c):
        return a * np.exp(-((x) ** 2) / (2 * c**2))

    popt, pcov = curve_fit(
        gaussian,
        radii,
        output_row_middle,
        p0=[np.max(output_row_middle), np.mean(np.abs(radii))],
    )

    return popt, pcov, lambda x: gaussian(x, *popt)


def solve_PIAA_sag_functions(n, d, final_D, gaussian_params):
    popt = gaussian_params

    def int_fr(r):
        r_max = final_D / 2
        # analytic integral of f(r) * r dr from 0 to r
        if r <= final_D / 2:
            return (r**2) / (r_max**2) / (2 * np.pi)
        else:
            return 1.0 / (2 * np.pi)

    max_R = final_D / 2 - 0.05e-3

    def int_gr(R):
        # anlytic integral of a gaussian with standard deviation c
        a, c = popt
        norm = 2 * np.pi * c**2 * (1 - np.exp(-(max_R**2) / (2 * c**2)))
        if R >= 0 and R <= max_R:
            return (c**2 / norm) * (1 - np.exp(-(R**2) / (2 * c**2)))
        elif R > max_R:
            return 1.0 / (2 * np.pi)
        else:
            return 0.0

    # verify normalization
    integral_f = int_fr(np.inf) * 2 * np.pi
    # print(f"Integral of f(r): {integral_f}")
    integral_g = int_gr(np.inf) * 2 * np.pi
    # print(f"Integral of g(r): {integral_g}")

    integral_f = int_fr(final_D / 2) * 2 * np.pi
    # print(f"Integral of f(r): {integral_f}")
    integral_g = int_gr(max_R) * 2 * np.pi
    # print(f"Integral of g(r): {integral_g}")

    def find_R(f_int, g_int, r):
        # eq 17 from Hoffnagle and Jefferson
        # find R such that
        # int_0^r f(x) x dx = int_0^R g(x) x dx
        integral_f = f_int(r)

        def to_solve(R):
            integral_g = g_int(R)
            return integral_f - integral_g

        R = fsolve(to_solve, r)[0]
        return R

    r_vals = np.linspace(0.0, final_D / 2, 100)
    R_vals = [find_R(int_fr, int_gr, r) for r in r_vals]

    def h(r):
        return np.interp(r, r_vals, R_vals)

    def h_inv(R):
        return np.interp(R, R_vals, r_vals)

    def z(r):
        # eq 18a
        # z(r) = int_0^r  ((n^2-1) + ((n-1)d / (h(x) - x))^2)^(-1/2)  dx
        # where h(x) is the mapping function found above
        if r > final_D / 2:
            return z(final_D / 2)
        integral, error = quad(
            lambda x: ((n**2 - 1) + ((n - 1) * d / (h(x) - x)) ** 2) ** (-1 / 2), 0, r
        )
        return integral

    def Z(R):
        integral, error = quad(
            lambda x: ((n**2 - 1) + ((n - 1) * d / (h_inv(x) - x)) ** 2) ** (-1 / 2),
            0,
            R,
        )
        return integral

    z_vals = [z(r) for r in r_vals]

    R_vals_plot = np.linspace(0.0, max_R, 100)
    Z_vals = [Z(R) for R in R_vals_plot]

    # now see if we can make this PIAA lens using 2 phase screens and a fresnel propagator

    # convert z(r) to a 2D phase screen, where phase = (n-1) * k * z(r), k=2pi/1
    # 1D array of coordinates (length = pupil_grid.size)

    z_approx = lambda r: np.interp(r, r_vals, z_vals)
    Z_approx = lambda R: np.interp(R, R_vals, Z_vals)

    return z_approx, Z_approx, h, h_inv


def compute_phase_screens(
    z: callable,
    Z: callable,
    h: callable,
    n: float,
    wavel: float,
    scaled_pupil_grid: hcipy.Field,
):
    r_1d = scaled_pupil_grid.as_("polar").r

    phase_1 = (n - 1) * (2 * np.pi / wavel) * z(r_1d)
    phase_2 = (n - 1) * (2 * np.pi / wavel) * Z(h(r_1d))

    for i, phase in enumerate([phase_1, phase_2]):
        max_phase_diff = np.max(
            np.diff(
                phase.reshape(scaled_pupil_grid.shape)[
                    :, scaled_pupil_grid.shape[0] // 2
                ]
            )
        )
        print(
            f"Max phase difference between adjacent pixels in phase {i + 1}: {max_phase_diff} rad"
        )
        if max_phase_diff > np.pi:
            print(f"Warning: phase screen {i + 1} may be undersampled!")

    return phase_1, phase_2


def compute_sag_screens(
    z: callable,
    Z: callable,
    h: callable,
    scaled_pupil_grid: hcipy.Field,
):
    r_1d = scaled_pupil_grid.as_("polar").r

    sag_1 = z(r_1d)
    sag_2 = Z(h(r_1d))

    return sag_1, sag_2


class PIAA_Optics:
    def __init__(
        self,
        pupil_grid,
        sag_1,
        sag_2,
        dist,
        refractive_index,
    ):
        self.pupil_grid = pupil_grid
        self.lens_1 = hcipy.SurfaceApodizer(sag_1, refractive_index=refractive_index)
        self.lens_2 = hcipy.SurfaceApodizer(sag_2, refractive_index=refractive_index)
        self.dist = dist
        self.prop = hcipy.FresnelPropagator(
            pupil_grid, dist, num_oversampling=4, zero_padding=2
        )

    def apply_PIAA(self, wavefront):
        """Propagates a wavefront through the PIAA optics.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf : Wavefront
            The output wavefront.
        """
        wf = self.lens_1.forward(wavefront)
        wf = self.prop.forward(wf)
        wf = self.lens_2.forward(wf)
        wf = self.prop.backward(wf)  # return back to pupil plane

        return wf

    def apply_inverse_PIAA(self, wavefront):
        """Propagates a wavefront backwards through the PIAA optics.

        Parameters
        ----------
        wavefront : Wavefront
            The input wavefront that will propagate through the system.

        Returns
        -------
        wf : Wavefront
            The output wavefront.
        """
        pia_wf_inv = self.lens_2.backward(wavefront)
        pia_wf_inv = self.prop.backward(pia_wf_inv)
        pia_wf_inv = self.lens_1.backward(pia_wf_inv)
        pia_wf_inv = self.prop.forward(pia_wf_inv)

        return pia_wf_inv


###################################################################
# FI metrics
###################################################################


def FI_metric(FI_matrix, criterion="trace"):
    """
    A metric to evaluate the Fisher Information matrix.

    Parameters
    ----------
    FI_matrix : np.ndarray
        Fisher Information matrix.
    criterion : str, optional
        The criterion to use for the metric. Default is "trace".
        Different metrics have the following interpretations:
        - "trace": S-optimality, sum of variances.
        - "determinant": D-optimality, determinant of covariance (minimize volume).
        - "max_eigenvalue": E-optimality, largest eigenvalue of covariance (worst-case variance).
        - "minus min_eigenvalue": S-optimality (alternative), smallest eigenvalue of FI (best constrained direction).
    """
    cov_matrix = np.linalg.inv(FI_matrix)
    if criterion == "trace":
        # S-optimality: sum of variances (trace of covariance)
        return np.trace(cov_matrix)
    elif criterion == "determinant":
        # D-optimality: determinant of covariance (minimize volume)
        return np.linalg.det(cov_matrix)
    elif criterion == "log_determinant":
        # D-optimality: log-determinant of covariance (minimize volume)
        return np.log10(np.linalg.det(cov_matrix))
    elif criterion == "max_eigenvalue":
        # E-optimality: largest eigenvalue of covariance (worst-case variance)
        return np.max(np.linalg.eigvalsh(cov_matrix))
    elif criterion == "minus min_eigenvalue":
        # S-optimality (alternative): smallest eigenvalue of FI (best constrained direction)
        return -np.min(np.linalg.eigvalsh(FI_matrix))
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


all_criteria = [
    "trace",
    "determinant",
    "log_determinant",
    "max_eigenvalue",
    "minus min_eigenvalue",
]
