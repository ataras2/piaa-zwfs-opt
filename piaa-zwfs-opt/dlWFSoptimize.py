import cmocean
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
import matplotlib.pyplot as plt
import dlMaterials
import zodiax as zdx
import equinox as eqx
import pcscep
import os
import hcipy
import jax
import optimistix as optx
import jax.lax as lax
from jax import grad, jit, vmap, linearize
import Aspheres
import WFSoptimize
import collections
import serialisation


def sim_dims(
    pupil_diameter: float,
    det_pix_over_pupil: int,
    min_sim_diameter: float,
    target_wf_npixels: int,
):
    sim_upsample = int(
        (target_wf_npixels * pupil_diameter / min_sim_diameter) // det_pix_over_pupil
    )

    det_projected_pixel_size = pupil_diameter / det_pix_over_pupil

    final_detector_pixels = int(np.ceil(min_sim_diameter / det_projected_pixel_size))

    actual_sim_diameter = final_detector_pixels * det_projected_pixel_size
    actual_wf_npixels = final_detector_pixels * sim_upsample

    # make sure actual_wf_npixels is even but also a multiple of sim_upsample
    if actual_wf_npixels % 2 != 0:
        actual_wf_npixels += sim_upsample
        actual_sim_diameter = (
            actual_wf_npixels / sim_upsample * det_projected_pixel_size
        )

    return actual_sim_diameter, actual_wf_npixels, sim_upsample


def apply_complex_map(field, cmap, power=1.0, normalise=True, shift=0.0):
    """
    Maps a complex-valued 2D array to an RGB image using a colormap.

    The hue is determined by the phase (angle) of each complex value, and the brightness
    is scaled by the magnitude (absolute value) raised to the given power. The result is
    normalized to the [0, 1] range for display.

    Parameters
    ----------
    field : np.ndarray
        2D array of complex values to visualize.
    cmap : callable
        A matplotlib colormap function (e.g., plt.cm.hsv).
    power : float, optional
        Exponent to apply to the magnitude for brightness scaling/gamma (default is 1.0).

    Returns
    -------
    np.ndarray
        3D array representing the RGB image (shape: field.shape + (3,)).
    """
    angles = np.angle(field)
    angles_norm = np.mod(angles + shift, 2 * np.pi) / (2 * np.pi)

    if normalise:
        amp = np.abs(field) ** power / np.max(np.abs(field) ** power)
    else:
        amp = np.abs(field) ** power

    img = cmap(angles_norm) * (amp)[:, :, None]

    if normalise:
        img = img[..., :3] / np.max(img[..., 0:3])
    else:
        img = img[..., :3]

    return img


def imshow_wavefront(
    wavefront: dl.Wavefront,
    show="complex",
    ax=None,
):
    """
    Display a wavefront on a Matplotlib axis.

    Parameters
    ----------
    wavefront : dl.Wavefront
        Wavefront object containing `phasor` and `coordinates` to display.
    show : {'complex', 'intensity', 'phase'}, optional
        How to visualise the wavefront. 'complex' shows phase mapped to hue
        and amplitude to brightness, 'intensity' shows |E|^2, 'phase' shows
        the wrapped phase. Default is 'complex'.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. If None, uses the current axis from Matplotlib.
    """
    if ax is None:
        ax = plt.gca()

    kwargs = {}
    kwargs["extent"] = [
        wavefront.coordinates[0].min(),
        wavefront.coordinates[0].max(),
        wavefront.coordinates[1].min(),
        wavefront.coordinates[1].max(),
    ]

    if show == "complex":
        img = apply_complex_map(wavefront.phasor, cmocean.cm.phase)
        # take extent from wavefront.coordinates
        ax.imshow(
            img,
            **kwargs,
        )
    elif show == "intensity":
        ax.imshow(
            np.abs(wavefront.phasor) ** 2,
            **kwargs,
        )
    elif show == "phase":
        ax.imshow(
            np.angle(wavefront.phasor),
            **kwargs,
        )
    else:
        raise ValueError("show must be 'complex', 'intensity' or 'phase'")


def make_freq_basis(
    wf_npixels,
    sim_diam,
    pupil_diameter,
    pupil,
    min_cycles: float = 1e-2,
    max_cycles: float = 15.0,
    num_freqs: int = 21,
    trig_type="sin",
    direction="h",
):
    """
    Make a horizontal frequency basis set.

    Parameters
    ----------
    wf_npixels : int
        The number of pixels in the wavefront.
    sim_diam : float
        The diameter of the simulation in metres.
    pupil_diameter : float
        The diameter of the pupil in metres.
    pupil : Array
        The pupil mask to normalise the basis functions.
    min_cycles : float, optional
        The minimum number of cycles across the pupil diameter, by default 1e-2.
        Units are cycles per pupil diameter.
    max_cycles : float, optional
        The maximum number of cycles across the pupil diameter, by default 15.0.
        Units are cycles per pupil diameter.
    num_freqs : int, optional
        The number of frequency basis functions to generate, by default 21.
    trig_type : str, optional
        The type of trigonometric function to use ('sin', 'cos', or 'both' to include both), by default "sin".
    direction : str, optional
        The direction of the frequency basis ('h' for horizontal, 'v' for vertical, "b" for both - in which case 2*num_freqs
        basis functions are returned), by default "h".
    Returns
    -------
    freq_basis : List[Array]
        The list of frequency basis functions.
    freqs : Array
        The array of frequencies in cycles (physical units: cycles per metre).
    """
    if trig_type not in ["sin", "cos", "both"]:
        raise ValueError("trig_type must be 'sin', 'cos', or 'both'")

    if trig_type == "sin":
        trig_fns = [np.sin]
    elif trig_type == "cos":
        trig_fns = [np.cos]
    else:  # both
        trig_fns = [np.sin, np.cos]

    x, y = dlu.coordinates.pixel_coords(wf_npixels, sim_diam)
    freq_basis = []
    freqs = np.linspace(min_cycles, max_cycles, num_freqs) / pupil_diameter

    def _generate(axis_values):
        out = []
        for f in freqs:
            for tf in trig_fns:
                bf = 1.0 * tf(2 * np.pi * f * axis_values)
                bf /= np.std(bf[pupil > 0])
                bf *= pupil
                out.append(bf)
        return out

    # number of repeats per axis due to trig functions
    trig_repeat = len(trig_fns)
    freqs_repeated = np.tile(freqs, trig_repeat)

    if direction == "h":
        freq_basis = np.array(_generate(x))
        return freq_basis, freqs_repeated
    elif direction == "v":
        freq_basis = np.array(_generate(y))
        return freq_basis, freqs_repeated
    elif direction == "b":
        # both directions: horizontal then vertical
        freq_basis = np.concatenate([np.array(_generate(x)), np.array(_generate(y))])
        return freq_basis, np.concatenate([freqs_repeated, freqs_repeated])
    else:
        raise ValueError("direction must be 'h', 'v', or 'b'")


def orthonormalise_basis(basis: np.ndarray, support: np.ndarray) -> np.ndarray:
    """
    Orthonormalise a basis set on a given support using a QR decomposition on the flattened basis.
    Parameters
    ----------
    basis : Array
        The basis set to orthonormalise. Shape (n_basis, ny, nx).
    support : Array
        The support mask to orthonormalise the basis on. Shape (ny, nx) - should be 1 where the basis is defined, 0 elsewhere.
    Returns
    -------
    orthonormal_basis : Array
        The orthonormalised basis set. Shape (n_basis, ny, nx).
    """
    n_basis, ny, nx = basis.shape
    basis_reshaped = basis.reshape(n_basis, ny * nx) * support.flatten()
    q, r = np.linalg.qr(basis_reshaped.T)
    orthonormal_basis = q.T.reshape(n_basis, ny, nx)
    # normalise
    orthonormal_basis = (
        orthonormal_basis / orthonormal_basis[:, support > 0].std(axis=1)[:, None, None]
    )
    return orthonormal_basis


def make_zernike_basis(
    wf_npixels,
    sim_diam,
    pupil_diameter,
    n_zernikes,
    starting_index: int = 2,
    orthonormalise: bool = False,
    support=None,
):
    """
    Make a zernike basis set on the pupil.
    If the pupil is not a pure circle, provide the support mask to orthonormalise the basis.

    Parameters
    ----------
    wf_npixels : int
        The number of pixels in the wavefront.
    sim_diam : float
        The diameter of the simulation in metres.
    pupil_diameter : float
        The diameter of the pupil in metres.
    n_zernikes : int
        The number of zernike modes to generate.
    starting_index: int, optional
        The starting index for the zernike modes, by default 1 (excludes piston).
    orthonormalise : bool, optional
        Whether to orthonormalise the basis on the support, by default False.
    support : Array, optional
        The support mask to orthonormalise the basis, by default None.
    Returns
    -------
    zernike_basis : Array
        The array of zernike basis functions.
    """
    if orthonormalise and support is None:
        raise ValueError("Support must be provided to orthonormalise the basis.")

    pupil_coords = dlu.coordinates.pixel_coords(wf_npixels, sim_diam)
    zernike_basis = []
    for n in range(starting_index, n_zernikes + starting_index):
        zernike_mode = dlu.zernike(n, pupil_coords, pupil_diameter)
        # zernike_mode /= np.std(zernike_mode)
        zernike_basis.append(zernike_mode)

    zernike_basis = np.array(zernike_basis)

    if orthonormalise:
        zernike_basis = orthonormalise_basis(zernike_basis, support)

    return zernike_basis


class BasisGlassLayer(dl.optical_layers.OpticalLayer):
    basis: np.ndarray
    coefficients: np.ndarray
    refractive_index: float | dlMaterials.Material

    def __init__(
        self,
        basis: np.ndarray,
        coefficients: np.ndarray,
        refractive_index: float,
    ):
        """
        A layer that applies a basis set of glass thicknesses to the wavefront.
        Note that when used in the ThreePlaneSystem, this layer should have zero thickness
        where you want zero phase change (e.g. at the outer edges)

        Parameters
        ----------
        basis : Array
            The basis set of glass thicknesses to be applied to the wavefront.
        coefficients : Array
            The coefficients to be applied to each basis vector.
        refractive_index : float | callable
            The refractive index of the glass material.
        """
        basis = np.asarray(basis, dtype=float)
        coefficients = np.asarray(coefficients, dtype=float)
        if basis.shape[:-2] != coefficients.shape:
            raise ValueError(
                "The number of basis vectors must be equal to "
                "the number of coefficients."
            )

        self.basis = basis
        self.coefficients = coefficients
        self.refractive_index = refractive_index

    """A glass layer constructed from a linear basis of thickness patterns.

    The layer's optical path difference is the dot-product of `basis` and
    `coefficients`, scaled by the refractive index. Can be used as a
    focal-plane or pupil-plane thickness element.
    """

    def eval_basis(self) -> np.ndarray:
        """
        Calculates the dot product of the basis vectors and coefficients.

        Returns
        -------
        output : Array
            The output of the dot product between the basis vectors and coefficients.
        """
        return dlu.eval_basis(self.basis, self.coefficients)

    def get_index(self, wavelength: float) -> float:
        """
        Get the refractive index at a given wavelength.

        Parameters
        ----------
        wavelength : float
            The wavelength to get the refractive index at.

        Returns
        -------
        n : float
            The refractive index at the given wavelength.
        """
        if not isinstance(self.refractive_index, float):
            n = self.refractive_index.n(wavelength)
        else:
            n = self.refractive_index
        return n

    def apply(self, wavefront: dl.Wavefront) -> dl.Wavefront:
        """
        Applies the layer to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The transformed wavefront.
        """
        output = self.eval_basis()  # TODO dlu thickness to opd

        n = self.get_index(wavefront.wavelength)

        return wavefront.add_opd(output * (n - 1.0))

    def phase_at_wavel(self, wavelength: float, wrap=False) -> np.ndarray:
        """Show the phase applied by the layer at a given wavelength."""
        opd = self.eval_basis()
        n = self.get_index(wavelength)
        phase = (2 * np.pi / wavelength) * opd * (n - 1.0)

        if wrap:
            phase = np.angle(np.exp(1j * phase))

        return phase

    def __call__(self, wavefront: dl.Wavefront) -> dl.Wavefront:
        return self.apply(wavefront)


def annulus(coords, inner_radius, outer_radius):
    """
    Create an annulus function.

    Parameters
    ----------
    coords : Array
        The pixel coordinates.
    inner_radius : float
        The inner radius of the annulus.
    outer_radius : float
        The outer radius of the annulus.

    Returns
    -------
    annulus : Array
        The annulus function.
    """
    r = np.sqrt(coords[0] ** 2 + coords[1] ** 2)
    annulus = np.logical_and(r >= inner_radius, r <= outer_radius).astype(float)
    return annulus


def make_annular_basis(
    wf_npixels,
    sim_diam,
    n_rings: int,
    scale=1e-6,
):
    """
    Make an annular basis set.

    Parameters
    ----------
    wf_npixels : int
        The number of pixels in the wavefront.
    sim_diam : float
        The diameter of the simulation in metres.
    n_rings : int
        The number of annular rings to generate.

    Returns
    -------
    annular_basis : List[Array]
        The list of annular basis functions.
    """
    coords = dlu.coordinates.pixel_coords(wf_npixels, sim_diam)
    max_radius = sim_diam / 2
    annular_basis = []
    for i in range(n_rings):
        inner_radius = (i / n_rings) * max_radius
        outer_radius = ((i + 1) / n_rings) * max_radius
        ring = annulus(
            coords=coords, inner_radius=inner_radius, outer_radius=outer_radius
        )
        annular_basis.append(ring * scale)
    return annular_basis


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
    bandwidth=None,
    fractional_bandwidth=None,
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
    bandwidth : float, optional
        Bandwidth in meters. Used if band is None.
    Returns
    -------
    np.ndarray
        Array of wavelengths in meters.
    """

    if fractional_bandwidth is not None:
        if central_wavel is None:
            raise ValueError(
                "central_wavel must be provided if fractional_bandwidth is used."
            )
        if bandwidth is not None:
            raise ValueError(
                "bandwidth should not be provided if fractional_bandwidth is used."
            )
        bandwidth = central_wavel * fractional_bandwidth

    if band is not None:
        central_wavel, bandwidth = band_to_center_and_bandwidth[band]
        min_wavel = central_wavel - bandwidth / 2
        max_wavel = central_wavel + bandwidth / 2
    elif central_wavel is not None and bandwidth is not None:
        min_wavel = central_wavel - bandwidth / 2
        max_wavel = central_wavel + bandwidth / 2
    elif min_wavel is None or max_wavel is None:
        raise ValueError(
            "Either band or min/max wavelengths or central wavelength and bandwidth must be provided."
        )

    wavels = np.linspace(min_wavel, max_wavel, num_wavels)
    return wavels


class ThreePlaneSystem(dl.optical_layers.OpticalLayer):
    """
    A propagator for a three-plane system: pupil -> focus (with mask) -> pupil.
    Only simulates the part of the beam that is changed by the mask, as in
    Soummer 2007 (https://arxiv.org/pdf/0711.0368)
    Assumes the rest of the focal plane mask is perfectly transmissive.

    Parameters
    ----------
    focus_npixels : int
        The number of pixels in the focal plane.
    focus_pixel_scale : float
        The pixel scale in the focal plane (radians per pixel).
    mask_at_focus : callable
        A function that takes a wavefront at the focus and applies the mask.
    Returns
    -------
    dl.Wavefront
        The output wavefront at the pupil plane after propagation.
    """

    focus_npixels: int
    focus_pixel_scale: float
    mask_at_focus: dl.BaseLayer

    def __init__(
        self,
        focus_npixels: int,
        focus_pixel_scale: float,
        mask_at_focus: dl.BaseLayer,
    ):
        self.focus_npixels = focus_npixels
        self.focus_pixel_scale = focus_pixel_scale
        self.mask_at_focus = mask_at_focus

    def apply(
        self,
        wavefront: dl.Wavefront,
    ) -> dl.Wavefront:

        # Propagate to focus
        # foc_wf = dl.MFT(focus_npixels, focus_pixel_scale).apply(wavefront)
        foc_wf = dl.Wavefront(
            npixels=self.focus_npixels,
            diameter=self.focus_pixel_scale * self.focus_npixels,
            wavelength=wavefront.wavelength,
        )
        foc_phasor = dlu.MFT(
            phasor=wavefront.phasor,
            pixel_scale_in=wavefront.pixel_scale,
            wavelength=wavefront.wavelength,
            pixel_scale_out=self.focus_pixel_scale,
            npixels_out=self.focus_npixels,
        )
        foc_wf = foc_wf.set(["phasor"], [foc_phasor])
        foc_wf -= self.mask_at_focus.apply(foc_wf)

        # Propagate back to pupil
        # pup_wf = dl.MFT(wavefront.npixels, wavefront.pixel_scale).apply(foc_wf)
        pup_wf_phasor = dlu.MFT(
            phasor=foc_wf.phasor,
            pixel_scale_in=self.focus_pixel_scale,
            wavelength=wavefront.wavelength,
            pixel_scale_out=wavefront.pixel_scale,
            npixels_out=wavefront.npixels,
            inverse=True,
        )

        output_phasor = wavefront.phasor - pup_wf_phasor

        output_wavefront = wavefront.set(
            ["phasor"],
            [output_phasor],
        )
        return output_wavefront

    def __call__(
        self,
        wavefront: dl.Wavefront,
    ) -> dl.Wavefront:
        return self.apply(wavefront)


class Magnifier(dl.OpticalLayer):
    magnification: float

    def __init__(self, magnification: float):
        self.magnification = magnification

    def apply(self, wavefront: dl.Wavefront) -> dl.Wavefront:
        new_diameter = wavefront.diameter * self.magnification
        return wavefront.set(
            ["pixel_scale"],
            [new_diameter / wavefront.npixels],
        )

    def __call__(self, wavefront: dl.Wavefront) -> dl.Wavefront:
        return self.apply(wavefront)


def load_model(pth):
    """Load the latest core model and associated ASDF optimization file.

    Parameters
    ----------
    pth : str
        Path to a directory containing `core_model_*.model` and
        `optimization_*.asdf` files. The latest core model is selected by
        file creation time.

    Returns
    -------
    core_model, asdf_pth
        The deserialised core model object and the path to the matching
        optimization ASDF file.
    """
    # find the latest file
    all_files = os.listdir(pth)
    all_asdf_files = [f for f in all_files if f.endswith(".model")]
    latest_file = max(
        all_asdf_files, key=lambda f: os.path.getctime(os.path.join(pth, f))
    )
    load_pth = os.path.join(pth, latest_file)

    file_time_info = latest_file.replace("core_model_", "").replace(".model", "")

    asdf_fname = f"optimization_{file_time_info}.asdf"
    asdf_pth = os.path.join(pth, asdf_fname)

    core_model = serialisation.deserialise(load_pth)

    return core_model, asdf_pth


def aperture_name_to_pupil_diameter(aperture_name: str) -> float:
    """Map a known aperture name to its physical pupil diameter (m).

    Parameters
    ----------
    aperture_name : str
        Known aperture identifier (e.g. 'magellan', 'elt', 'gmt').

    Returns
    -------
    float
        Pupil diameter in metres.
    """
    if aperture_name == "magellan" or aperture_name == "magellan_clear":
        return 6.5  # meters
    elif aperture_name == "elt":
        return 39.0  # meters
    elif aperture_name == "gmt":
        return 25.448  # meters
    else:
        raise ValueError(f"Unknown aperture: {aperture_name}")


import pcscep


def make_aperture(aperture_name, diameter, wf_npixels, aperture_oversample):
    """Create an `dl.TransmissiveLayer` aperture and its high-level HCIPy mask.

    Parameters
    ----------
    aperture_name : str
        Identifier for the aperture ('magellan', 'magellan_clear', 'elt', 'gmt').
    diameter : float
        Physical diameter of the simulated pupil (metres).
    wf_npixels : int
        Number of wavefront pixels across the simulation.
    aperture_oversample : int
        Supersampling factor for aperture evaluation.

    Returns
    -------
    dl.TransmissiveLayer, numpy.ndarray
        A `dl.TransmissiveLayer` containing the sampled transmission array
        and the raw (flattened) aperture mask returned by HCIPy evaluation.
    """
    if aperture_name == "magellan":
        pupil_grid = hcipy.make_pupil_grid(wf_npixels, diameter=diameter)
        aperture = hcipy.make_magellan_aperture(normalized=False, with_spiders=True)
    elif aperture_name == "magellan_clear":
        pupil_grid = hcipy.make_pupil_grid(wf_npixels, diameter=diameter)
        aperture = hcipy.make_obstructed_circular_aperture(
            6.5, central_obscuration_ratio=0.0
        )
    elif aperture_name == "elt":
        pupil_grid = hcipy.make_pupil_grid(wf_npixels, diameter=diameter)
        aperture = pcscep.make_elt_aperture(normalized=False)
    elif aperture_name == "gmt":
        pupil_grid = hcipy.make_pupil_grid(wf_npixels, diameter=diameter)
        aperture = hcipy.make_gmt_aperture(normalized=False)
    else:
        raise ValueError(f"Unknown aperture: {aperture_name}")

    aperture = hcipy.evaluate_supersampled(aperture, pupil_grid, aperture_oversample)
    aperture_sqr = aperture.reshape((wf_npixels, wf_npixels))
    return (
        dl.TransmissiveLayer(
            aperture_sqr,
            normalise=True,
        ),
        aperture,
    )


def project_onto_basis(instance, basis):
    """Project an aberration instance onto a basis with least-squares.

    Parameters
    ----------
    instance : array_like
        Flattenable aberration map to project (shape matching basis pixels).
    basis : ndarray
        Basis array of shape (n_terms, ny, nx).

    Returns
    -------
    ndarray
        Coefficients for the basis that best fit `instance` in least-squares.
    """
    n_terms = basis.shape[0]

    A = basis.reshape((n_terms, -1)).T  # shape (npixels^2, n_terms)
    b = instance.reshape((-1,))  # shape (npixels^2,)

    # Solve the least squares problem
    return np.linalg.lstsq(A, b, rcond=None)[0]


class IdealWFS(dl.optical_layers.OpticalLayer):
    """
    An ideal wavefront sensor changes the piston mode i.e. aperture by pi/2,
    and leaves all other modes unchanged.
    """

    aperture: np.ndarray

    def __init__(self, aperture):
        self.aperture = aperture / np.sqrt(np.sum(aperture**2))

    def apply(self, wavefront: dl.Wavefront) -> dl.Wavefront:
        Ei = wavefront.phasor

        overlap = np.sum(Ei * self.aperture)

        Eo = 1j * overlap * self.aperture + (Ei - overlap * self.aperture)
        return wavefront.set(
            "phasor",
            Eo,
        )


class GlassDot(Aspheres.ParametricRefractiveOptic):
    diameter: np.ndarray
    thickness: np.ndarray
    material: dlMaterials.Material

    coords: eqx.field(static=True)
    diameter_scale: eqx.field(static=True)
    thickness_scale: eqx.field(static=True)

    def __init__(
        self,
        diameter: np.ndarray,
        thickness: np.ndarray,
        material: dlMaterials.Material,
        coords: np.ndarray,
        diameter_scale: float = 1.0,
        thickness_scale: float = 1.0,
    ):
        """Create a circular glass dot parameterised by diameter and thickness.

        The dot is represented as a softened circle on the provided coordinate
        grid. Returned thickness maps are in the same units as `thickness`.
        """
        if isinstance(diameter, (float, int)):
            diameter = np.array([diameter])
        if isinstance(thickness, (float, int)):
            thickness = np.array([thickness])

        self.diameter = diameter
        self.thickness = thickness
        self.material = material
        self.coords = coords

        self.diameter_scale = diameter_scale
        self.thickness_scale = thickness_scale

    def generate_thickness(self):
        """Generate the thickness map for the glass dot on `coords`.

        Returns
        -------
        ndarray
            2D thickness map matching the `coords` grid.
        """
        diameter = self.diameter * self.diameter_scale
        thickness = self.thickness * self.thickness_scale
        circle = dlu.soft_circle(
            coords=self.coords,
            radius=diameter / 2,
            clip_dist=0.01 * diameter,
        )
        return circle * thickness

    def __call__(self):
        return self.generate_thickness()


def mag2flux(z_mag, zero_point_flux, pupil_area, exposure_time, bandwidth):
    """Convert magnitude to photon counts for an exposure.

    Parameters
    ----------
    z_mag : float
        Source magnitude in the chosen zero-point system.
    zero_point_flux : float
        Zero-point in photons cm^-2 s^-1 A^-1 (from `zero_points`).
    pupil_area : float
        Collecting area (m^2).
    exposure_time : float
        Exposure time in seconds.
    bandwidth : float
        Wavelength bandwidth in metres.

    Returns
    -------
    float
        Number of photons detected in the exposure (photons).
    """
    flux_per_m2 = zero_point_flux * 1e4 * 1e10 * 10 ** (-0.4 * z_mag)  # photons/m2/s/m
    source_photons_per_s = flux_per_m2 * pupil_area * bandwidth

    return source_photons_per_s * exposure_time


def flux2mag(flux, zero_point_flux, pupil_area, exposure_time, bandwidth):
    """Convert detected flux back to magnitude.

    Parameters
    ----------
    flux : float
        Detected photons in the exposure.
    zero_point_flux : float
        Zero-point in photons cm^-2 s^-1 A^-1.
    pupil_area : float
        Collecting area (m^2).
    exposure_time : float
        Exposure time in seconds.
    bandwidth : float
        Wavelength bandwidth in metres.

    Returns
    -------
    float
        Computed magnitude.
    """
    flux_per_m2 = flux / (pupil_area * exposure_time) / bandwidth  # photons/m2/s/m
    z_mag = -2.5 * np.log10(flux_per_m2 / (zero_point_flux * 1e4 * 1e10))
    return z_mag


class ModelCore(zdx.base.Base):
    """Container representing the core forward model and its parameters.

    `ModelCore` wraps a telescope (`dl.Telescope`) and a parameter dictionary
    used to construct, evaluate and differentiate model outputs (images,
    Fisher information, losses) with respect to instrument or aberration
    parameters.
    """

    params: dict
    tel: dl.Telescope

    tel_params_to_params: eqx.field(static=True)

    def __init__(
        self,
        params: dict,
        tel: dl.Telescope,
    ):
        self.params = params
        self.tel = tel

        tel_params_to_params = {
            "BasisLayer.coefficients": "phase_coefficients",
            "AmplitudeBasisLayer.coefficients": "amp_coefficients",
            # case of annular basis mask
            "ThreePlaneSystem.mask_at_focus.coefficients": "phase_mask_coeffs",
            # case of phase dot
            "ThreePlaneSystem.mask_at_focus.diameter": "phase_mask_diameter",
            "ThreePlaneSystem.mask_at_focus.thickness": "phase_mask_thickness",
            # piaas
            "PIAAset_0.piaa_lens_1.asphere_profile.coeffs": "piaa1_coeffs",
            "PIAAset_0.piaa_lens_2.asphere_profile.coeffs": "piaa2_coeffs",
            "PIAAset_0.piaa_lens_1.asphere_profile.f_rho_max": "piaa1_f_rho_max",
            "PIAAset_0.piaa_lens_2.asphere_profile.f_rho_max": "piaa2_f_rho_max",
            "PIAAset_0.distance": "piaa_distance",
            "PIAAset_1.piaa_lens_1.asphere_profile.coeffs": "piaa1_coeffs",
            "PIAAset_1.piaa_lens_2.asphere_profile.coeffs": "piaa2_coeffs",
            "PIAAset_1.piaa_lens_1.asphere_profile.f_rho_max": "piaa1_f_rho_max",
            "PIAAset_1.piaa_lens_2.asphere_profile.f_rho_max": "piaa2_f_rho_max",
            "PIAAset_1.distance": "piaa_distance",
            "PIAAset_0.piaa_lens_1.decentre": "piaa1_lens1_decentre",
            "PIAAset_0.piaa_lens_2.decentre": "piaa1_lens2_decentre",
            "PIAAset_1.piaa_lens_1.decentre": "piaa2_lens1_decentre",
            "PIAAset_1.piaa_lens_2.decentre": "piaa2_lens2_decentre",
        }

        # for each entry in tel_params_to_params, check if the param exists in tel
        # if not, remove it from the dict
        tel_params_to_params_final = {}
        for tel_param, param in tel_params_to_params.items():
            try:
                _ = tel.get(tel_param)
                tel_params_to_params_final[tel_param] = param
            except ValueError:
                pass
            except KeyError:
                pass

        self.tel_params_to_params = tel_params_to_params_final

    def _update_tel(self):
        tel_params_to_params = self.tel_params_to_params
        tel = self.tel

        params_used = []
        new_vals = []
        for tel_param, param in tel_params_to_params.items():
            value = self.params.get(param, None)
            if value is not None:
                params_used.append(tel_param)
                new_vals.append(value)
        return tel.set(params_used, new_vals)

    def img(self):
        self = self.set("tel", self._update_tel())
        return self.tel.model()

    def compute_J(self, around_params):
        """Compute Jacobian of the model image with respect to parameters.

        Parameters
        ----------
        around_params : dict
            Mapping of parameter names to values around which to linearise.

        Returns
        -------
        array-like
            Jacobian of the image with respect to the parameters provided.
        """

        def new_intensity(params_dict, model):
            # update each entry in core_model.params
            for key, value in params_dict.items():
                model = model.set(f"params.{key}", value)
            img = model.img()
            return img

        fn = eqx.filter_jacfwd(lambda x: new_intensity(x, self))

        return fn(around_params)

    def compute_FI(self, abberation_name: str):
        """Compute the Fisher information matrix for a named aberration set.

        Parameters
        ----------
        abberation_name : str
            Name of the parameter in `params` that contains the aberration
            coefficients (e.g. 'phase_coefficients').

        Returns
        -------
        ndarray
            Fisher information matrix for the requested aberration coefficients.
        """
        I = self.img()
        aberration_dict = {abberation_name: self.get(f"params.{abberation_name}")}
        J = self.compute_J(aberration_dict)[abberation_name]

        read_noise = self.get("params.read_noise")

        J = J.reshape((-1, len(aberration_dict[abberation_name])))

        return J.T @ (J / (I.flatten()[:, None] + read_noise**2))

    @staticmethod
    def calculate_loss(FI, flux, n_coefficients, loss_type):
        """Calculate a scalar loss from a Fisher information matrix.

        Parameters
        ----------
        FI : ndarray
            Fisher information matrix.
        flux : float
            Photon flux used for normalisation variants.
        n_coefficients : int
            Number of coefficients represented by FI (for normalisations).
        loss_type : str
            One of the supported loss identifiers.
        """
        if loss_type == "cov_trace_normed":
            return np.trace(np.linalg.inv(FI)) * flux / n_coefficients
        elif loss_type == "cov_trace":
            return np.trace(np.linalg.inv(FI))
        elif loss_type == "neg_FI_trace":
            return -np.trace(FI)
        elif loss_type == "neg_FI_trace_normed":
            return -np.trace(FI) / flux / n_coefficients
        elif loss_type == "cov_logdet_normed":
            return np.linalg.slogdet(np.linalg.inv(FI))[1] * flux / n_coefficients
        else:
            raise ValueError(
                "loss_type must be one of 'cov_trace_normed', 'cov_trace', "
                "'neg_FI_trace', 'neg_FI_trace_normed', or 'cov_logdet_normed'"
            )

    def loss(self, loss_type: str = "neg_FI_trace_normed"):
        """Compute the scalar loss for the current model parameters.

        Parameters
        ----------
        loss_type : str
            Loss variant to compute (see `calculate_loss`).
        """
        aberration_coefficients = self.get("params.phase_coefficients")
        FI = self.compute_FI("phase_coefficients")
        flux = self.get("tel.source.flux")
        return self.calculate_loss(FI, flux, len(aberration_coefficients), loss_type)

    def FI_normalise(self, FI_matrix, inverse=False):
        """Normalise or un-normalise a Fisher information matrix.

        This function scales FI matrices by photon flux and wavelength factors
        used elsewhere in the code to compare different sensors on a common
        scale.
        """
        central_wavel = self.tel.wavelengths.mean()
        flux = self.get("tel.source.flux")

        norm_factor = flux * (2 * np.pi / (central_wavel * 1e6)) ** 2

        if inverse:
            return FI_matrix / norm_factor
        else:
            return FI_matrix * norm_factor

    def loss_bound_idealwfs(self, loss_type: str = "neg_FI_trace_normed"):
        """Compute the loss for an ideal wavefront sensor bound.

        Returns the scalar loss assuming an ideal WFS sensitivity for each
        coefficient (used as a benchmark).
        """
        # an ideal WFS has FI = 4 on the diagional,
        # an example conversion from FI normalized sensitivty is:
        # FI[3,3]/actual_flux*(central_wavel*1e6/(2*np.pi))**2
        # i.e. the above =4 for an ideal WFS, assuming the opd_rms = 1 micron
        flux = self.get("tel.source.flux")
        central_wavel = self.tel.wavelengths.mean()

        aberration_coefficients = self.get("params.phase_coefficients")

        FI = (
            np.eye(len(self.get("params.phase_coefficients")))
            * 4
            * flux
            * (2 * np.pi / (central_wavel * 1e6)) ** 2
        )

        return self.calculate_loss(FI, flux, len(aberration_coefficients), loss_type)

    def optics_to_hcipy(self):
        """Placeholder: convert internal optics representation to HCIPy objects.

        Not yet implemented.
        """
        pass

    def get_aperture(self):
        """Return the current pupil transmission array from the telescope.

        Returns
        -------
        ndarray
            Transmission array used as the pupil support.
        """
        tel = self._update_tel()
        return tel.get("TransmissiveLayer").transmission

    def compute_standard_sensitivity(
        self, pupil_diameter, n_freqs=10, freq_start=1, freq_end=8, return_J=False
    ):
        """Compute standard sensitivity (Fisher or Jacobian) for sinusoidal bases.

        Parameters
        ----------
        pupil_diameter : float
            Physical pupil diameter in metres used to define cycles per meter.
        n_freqs : int
            Number of frequency samples.
        freq_start, freq_end : float
            Range of cycles-per-pupil values to evaluate.
        return_J : bool
            If True, return Jacobians instead of Fisher matrices.
        """
        model_copy = self.set("tel", self._update_tel())

        wfnpix = model_copy.tel.get("optics.wf_npixels")
        sim_diam = model_copy.tel.get("optics.diameter")

        args = {
            "wf_npixels": wfnpix,
            "sim_diam": sim_diam,
            "pupil_diameter": pupil_diameter,
            "pupil": model_copy.get_aperture(),
            "min_cycles": freq_start,
            "max_cycles": freq_end,
            "num_freqs": n_freqs,
        }

        sine_basis, freqs = make_freq_basis(
            **args,
            trig_type="sin",
        )

        cos_basis = make_freq_basis(
            **args,
            trig_type="cos",
        )[0]

        sin_model = model_copy.update_model_basis(sine_basis)
        cos_model = model_copy.update_model_basis(cos_basis)

        if not return_J:
            sin_FI = sin_model.compute_FI("phase_coefficients")
            cos_FI = cos_model.compute_FI("phase_coefficients")

            return sin_FI, cos_FI, freqs

        else:
            sin_J = sin_model.compute_J(
                {"phase_coefficients": sin_model.get("params.phase_coefficients")}
            )["phase_coefficients"]
            cos_J = cos_model.compute_J(
                {"phase_coefficients": cos_model.get("params.phase_coefficients")}
            )["phase_coefficients"]

            return sin_J, cos_J, freqs

    def update_model_basis(self, new_basis):
        """Replace the current BasisLayer in the telescope with `new_basis`.

        Parameters
        ----------
        new_basis : ndarray
            New basis array (n_terms, ny, nx) to install in the model.

        Returns
        -------
        ModelCore
            A new `ModelCore` instance with the updated telescope and
            zeroed coefficients.
        """
        tel = self._update_tel()

        new_coeffs = np.zeros(new_basis.shape[0])

        tel = tel.set(
            ["BasisLayer.basis", "BasisLayer.coefficients"], [new_basis, new_coeffs]
        )

        return self.set(["tel", "params.phase_coefficients"], [tel, new_coeffs])


def _perturb(X, pytree, parameters, shapes, lengths):
    """Apply a flat perturbation vector `X` to selected leaf parameters.

    This builds a new pytree where the listed `parameters` are replaced by
    segments taken from the flat vector `X`, using `shapes` and `lengths` to
    reconstruct array shapes.
    """
    n, xs = 0, []
    if isinstance(parameters, str):
        parameters = [parameters]
    indexes = range(len(parameters))

    for i, param, shape, length in zip(indexes, parameters, shapes, lengths):
        if length == 1:
            xs.append(X[i + n])
        else:
            xs.append(lax.dynamic_slice(X, (i + n,), (length,)).reshape(shape))
            n += length - 1

    return pytree.add(parameters, xs)


def hessian(f, x, has_aux=False, batch_size=1):
    """Compute the full Hessian of scalar function `f` at `x` using JAX.

    This function forms the Hessian by applying `linearize(grad(f))` to the
    standard basis. `batch_size` can be used to reduce peak RAM usage.
    """
    # Jit the sub-function here since it is called many times
    if has_aux:
        _, hvp, aux = linearize(grad(f, has_aux=has_aux), x, has_aux=has_aux)
    else:
        _, hvp = linearize(grad(f), x)
    hvp = jit(hvp)

    # Build the basis
    basis = np.eye(x.size).reshape(-1, *x.shape)

    if batch_size == 1:
        return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)

    hvp = vmap(hvp)

    # Break it into batches
    n_batch = np.maximum(1, len(basis) // batch_size)
    basis = np.array_split(basis, n_batch)
    return np.concatenate([hvp(batch) for batch in basis])


def smart_hessian(
    pytree,
    parameters,
    fn,
    *fn_args,
    has_aux=False,
    reduce_ram=False,
    batch_size=1,
    **fn_kwargs,
):
    """Convenience wrapper to compute a Hessian for selected pytree leaves.

    Parameters
    ----------
    pytree : zodiax-style pytree
        Container providing `.get` and `.add` for parameter leaves.
    parameters : list[str]
        Names of leaves to include in the Hessian.
    fn : callable
        Function accepting the perturbed pytree and additional args that
        returns a scalar to differentiate.

    Returns
    -------
    ndarray
        Hessian matrix for the flattened selected parameters.
    """
    # Build X vec
    pytree = zdx.tree.set_array(pytree, parameters)

    if len(parameters) == 1:
        parameters = [parameters]

    leaves = [pytree.get(p) for p in parameters]
    shapes = [leaf.shape for leaf in leaves]
    lengths = [leaf.size for leaf in leaves]
    N = np.array(lengths).sum()
    X = np.zeros(N)

    # Build function to calculate FIM and calculate
    def fn_vec(X):
        parametric_pytree = _perturb(X, pytree, parameters, shapes, lengths)
        return fn(parametric_pytree, *fn_args, **fn_kwargs)

    # Note reduce ram is removed until has_aux is implemented
    if reduce_ram:
        return hessian(fn_vec, X, has_aux=has_aux, batch_size=batch_size)
    else:
        return hessian(fn_vec, X, has_aux=has_aux)


def optx_minimize(
    pytree,
    parameters,
    solver,
    loss_type,
    **minimize_kwargs,
):
    """Minimise a `pytree.loss` using `optimistix.minimise` over selected leaves.

    Returns the optimised pytree and the solver result structure.
    """
    leaves = [pytree.get(p) for p in parameters]
    shapes = [leaf.shape for leaf in leaves]
    lengths = [leaf.size for leaf in leaves]

    X = np.zeros(np.array(lengths).sum())

    print("Initial loss:", pytree.loss(loss_type=loss_type))

    def loss_fn_vec(X, args):
        parametric_pytree = _perturb(X, pytree, parameters, shapes, lengths)
        return parametric_pytree.loss(loss_type=loss_type)

    res = optx.minimise(
        loss_fn_vec,
        solver,
        X,
        **minimize_kwargs,
    )
    pytree_optim = _perturb(res.value, pytree, parameters, shapes, lengths)
    return pytree_optim, res  # note that res.value is relative change in x


def optx_minimize_general(
    pytree,
    parameters,
    solver,
    fn,
    **minimize_kwargs,
):
    """General optimiser wrapper that minimises an arbitrary function `fn`.

    Parameters mirror `optx_minimize` but accept an explicit `fn` which
    takes a perturbed pytree and returns a scalar loss.
    """
    leaves = [pytree.get(p) for p in parameters]
    shapes = [leaf.shape for leaf in leaves]
    lengths = [leaf.size for leaf in leaves]

    X = np.zeros(np.array(lengths).sum())

    # print("Initial loss:", fn(pytree))

    def loss_fn_vec(X, args):
        parametric_pytree = _perturb(X, pytree, parameters, shapes, lengths)
        return fn(parametric_pytree)

    res = optx.minimise(
        loss_fn_vec,
        solver,
        X,
        **minimize_kwargs,
    )
    pytree_optim = _perturb(res.value, pytree, parameters, shapes, lengths)
    return pytree_optim, res  # note that res.value is relative change in x


# runs a big with optimization and saves the result in an asdf file
def aperture_area(aperture, wf_npixels, sim_diameter):
    """Compute the physical area of the aperture transmission > 0.5.

    Parameters
    ----------
    aperture : ndarray
        Transmission array (ny, nx) with values in [0, 1].
    wf_npixels : int
        Number of pixels across the simulation.
    sim_diameter : float
        Simulation diameter in metres.

    Returns
    -------
    float
        Physical area in square metres where aperture > 0.5.
    """
    dx = sim_diameter / wf_npixels
    return np.sum(aperture > 0.5) * dx * dx


def generate_idealwfs_optics(core_model):
    """Replace the telescope optics after the `BasisLayer` with an ideal WFS.

    This constructs a minimal telescope that terminates at a `BasisLayer`
    and appends an `IdealWFS` layer for benchmarking.
    """
    tel = core_model.get("tel")

    # delete every layer after "BasisLayer"
    new_layers = collections.OrderedDict()
    for layer_name, layer in tel.layers.items():
        new_layers[layer_name] = layer
        if layer_name == "BasisLayer":
            new_layers[layer_name] = layer
            break

    # add ideal wfs
    aperture = tel.get("TransmissiveLayer").transmission
    ideal_wfs = IdealWFS(aperture)
    new_layers["IdealWFS"] = ideal_wfs

    new_tel = tel.set(
        "layers",
        new_layers,
    )

    new_params = collections.OrderedDict()
    new_params["phase_coefficients"] = core_model.get("params.phase_coefficients")
    new_params["read_noise"] = core_model.get("params.read_noise")

    return core_model.set("tel", new_tel).set("params", new_params)


"""
https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
Quantity 	u 	g 	r 	i 	z 	Notes and units
λeff 	0.356 	0.483 	0.626 	0.767 	0.910 	microns, from Fukugita et al. (1996)
Δλ 	0.0463 	0.0988 	0.0955 	0.1064 	0.1248 	microns, from Fukugita et al. (1996)
fv 	3631 	3631 	3631 	3.631 	3631 	Jy or x10-23 erg cm-2 s-1 Hz-1
fλ 	859.5 	466.9 	278.0 	185.2 	131.5 	x10-11 erg cm-2 s-1 A-1, calculated from above quantities
Φλ 	1539.3 	1134.6 	875.4 	714.5 	602.2 	photons cm-2 s-1 A-1, calculated from above quantities


Quantity 	U 	B 	V 	R 	I 	J 	H 	K 	Notes and units
λeff 	0.36 	0.438 	0.545 	0.641 	0.798 	1.22 	1.63 	2.19 	microns
Δλ 	0.06 	0.09 	0.085 	0.15 	0.15 	0.26 	0.29 	0.41 	microns, UBVRI from Bessell (1990), JHK from AQ
fv 	1.79 	4.063 	3.636 	3.064 	2.416 	1.589 	1.021 	0.64 	x10-20 erg cm-2 s-1 Hz-1, from Bessell et al. (1998)
fλ 	417.5 	632 	363.1 	217.7 	112.6 	31.47 	11.38 	3.961 	x10-11 erg cm-2 s-1 A-1, from Bessell et al. (1998)
Φλ 	756.1 	1392.6 	995.5 	702.0 	452.0 	193.1 	93.3 	43.6 	photons cm-2 s-1 A-1, calculated from above quantities
"""


zero_points = {
    "z": 602.2,
    "Z": 602.2,
    "i": 714.5,
    "r": 875.4,
    "g": 1134.6,
    "u": 1539.3,
    "U": 756.1,
    "B": 1392.6,
    "V": 995.5,
    "R": 702.0,
    "I": 452.0,
    "J": 193.1,
    "H": 93.3,
    "K": 43.6,
}


def make_system(
    wavels,
    waveband,
    exposure_time,
    system_throughput,
    pupil_diameter,
    min_sim_diameter_frac,
    det_pix_over_pupil,
    num_airy,
    q,
    target_wf_npixels,
    aperture_name,
    oversample_aperture,
    basis_type,
    phase_rms,
    n_terms,
    basis_kwargs,
    has_PIAA,
    phase_mask_basis_name,
    phase_mask_material,
    n_zwfs_rings,
    final_D,
    piaa_material,
    n_sag_terms,
    d,
    init_PIAA_eval_dot_diam,
    read_noise,
    init_zernike_dot_diam=2.0,  # loD
    Nph_per_sec=None,
    magnitude=None,
    debug_plots=False,
):
    """Construct a full `ModelCore` for a ZWFS-style system.

    This helper builds optics, detectors, source and initial parameter
    dictionaries for many experiment variants (PIAA, phase-dot, annular
    phase masks, Fourier/Zernike bases). Returned `ModelCore` is ready for
    evaluation and optimisation.
    """
    central_wavel = wavels[len(wavels) // 2]
    loD = central_wavel / (final_D)  # radians

    magnifier = Magnifier(final_D / pupil_diameter)
    n_init_fit = dlMaterials.material_factory(piaa_material).n(wavels[len(wavels) // 2])

    #### Aperture stuff ####
    min_sim_diameter = min_sim_diameter_frac * pupil_diameter  # metres
    sim_diameter, wf_npixels, det_downsample = sim_dims(
        pupil_diameter, det_pix_over_pupil, min_sim_diameter, target_wf_npixels
    )
    print("Simulation diameter (m):", sim_diameter)
    print("Wavefront npixels:", wf_npixels)
    print("Detector downsample factor:", det_downsample)

    if debug_plots:
        sample_wf = dl.Wavefront(
            npixels=wf_npixels,
            diameter=sim_diameter,
            wavelength=central_wavel,
        )
        extent = [
            sample_wf.coordinates[0].min(),
            sample_wf.coordinates[0].max(),
            sample_wf.coordinates[1].min(),
            sample_wf.coordinates[1].max(),
        ]

    aperture, hcaperture = make_aperture(
        aperture_name, sim_diameter, wf_npixels, oversample_aperture
    )

    if debug_plots:
        plt.imshow(
            aperture.transmission,
            extent=extent,
        )
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Aperture: {aperture_name}")
        plt.colorbar(label="Transmission")
        plt.show()

    # sanity checks
    if magnitude is not None and Nph_per_sec is not None:
        raise ValueError("Specify either magnitude or Nph, not both.")

    if magnitude is not None:
        photons_per_exposure = mag2flux(
            magnitude,
            zero_points[waveband],
            aperture_area(aperture.transmission, wf_npixels, sim_diameter),
            exposure_time,
            wavels.max() - wavels.min(),
        )
    elif Nph_per_sec is not None:
        photons_per_exposure = Nph_per_sec * exposure_time

    photons_per_exposure *= system_throughput
    print(f"Photons per exposure: {photons_per_exposure:.2e}")

    #### Basis stuff ####
    if basis_type == "Fourier":
        phase_aberration_basis = make_freq_basis(
            wf_npixels=wf_npixels,
            sim_diam=sim_diameter,
            pupil_diameter=pupil_diameter,
            pupil=aperture.transmission,
            num_freqs=n_terms,
            **basis_kwargs,
        )[0]
        if basis_kwargs.get("trig_type", None) == "both":
            # remove the cosine 0 mode (index 1)
            phase_aberration_basis = np.concatenate(
                [phase_aberration_basis[0][None, :, :], phase_aberration_basis[2:]],
                axis=0,
            )
        phase_aberration_basis = orthonormalise_basis(
            basis=phase_aberration_basis,
            support=aperture.transmission,
        )

    elif basis_type == "Zernike":
        phase_aberration_basis = make_zernike_basis(
            wf_npixels=wf_npixels,
            sim_diam=sim_diameter,
            pupil_diameter=pupil_diameter,
            n_zernikes=n_terms,
            **basis_kwargs,
        )
    else:
        raise ValueError(f"Unknown basis type {basis_type}")

    phase_aberration_basis = phase_aberration_basis * phase_rms  # per um rms

    if debug_plots:
        for i in range(min(phase_aberration_basis.shape[0], 9)):
            plt.subplot(3, 3, i + 1)
            plt.imshow(
                phase_aberration_basis[i],
                extent=extent,
                cmap="RdBu",
            )
            plt.title(f"Basis mode {i+1}")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.colorbar()
        plt.tight_layout()
        plt.show()

    #### PIAA stuff ####
    if has_PIAA:
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
        z, Z, h, h_inv = WFSoptimize.solve_PIAA_sag_functions(
            n_init_fit, d, final_D, popt
        )

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

        rhos = np.linspace(0, final_D / 2, 500)
        if debug_plots:
            r_vals = dlu.cart2polar(sample_wf.coordinates)[0][:, wf_npixels // 2]

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
            imshow_wavefront(sample_wf)
            sample_wf = piaa_set.apply(sample_wf)
            plt.figure()
            imshow_wavefront(sample_wf)
            plt.figure()
            sample_wf = inv_piaa_set.apply(sample_wf)
            imshow_wavefront(sample_wf)

    ### phase mask stuff ###
    focal_pixel_scale = loD / q  # radians
    focus_npixels = int(2 * num_airy * (loD / focal_pixel_scale))

    init_desired_phase_shift = np.pi / 2  # radians
    phase_to_height = central_wavel / (
        2
        * np.pi
        * (dlMaterials.material_factory(phase_mask_material).n(central_wavel) - 1)
    )
    init_height = init_desired_phase_shift * phase_to_height

    if phase_mask_basis_name == "annular":
        phase_mask_basis = make_annular_basis(
            focus_npixels,
            focal_pixel_scale * focus_npixels,
            n_rings=n_zwfs_rings,
        )
        pix_per_ring = focus_npixels / (2 * n_zwfs_rings)
        target_npix = init_zernike_dot_diam / 2 * (loD / focal_pixel_scale)

        n_rings_used = int(target_npix / pix_per_ring)
        phase_mask_init_coeffs = np.concatenate(
            [
                np.ones(n_rings_used) * init_height * 1e6,
                np.zeros(len(phase_mask_basis) - n_rings_used),
            ]
        )
        mask_at_focus = BasisGlassLayer(
            basis=phase_mask_basis,
            coefficients=phase_mask_init_coeffs,
            refractive_index=dlMaterials.material_factory(phase_mask_material),
        )
    elif phase_mask_basis_name == "circle":
        mask_at_focus = GlassDot(
            diameter=init_zernike_dot_diam,
            thickness=init_desired_phase_shift,
            diameter_scale=loD,
            thickness_scale=phase_to_height,
            material=dlMaterials.material_factory(phase_mask_material),
            coords=dlu.pixel_coords(focus_npixels, focal_pixel_scale * focus_npixels),
        )
    else:
        raise ValueError(f"Unknown phase mask basis {phase_mask_basis_name}")

    layers = [
        aperture,
        dl.BasisLayer(
            basis=phase_aberration_basis,
            coefficients=np.zeros(len(phase_aberration_basis)),
            as_phase=False,
        ),
        magnifier,
    ]
    if has_PIAA:
        layers += [piaa_set]

    layers += [
        ThreePlaneSystem(
            focus_npixels=focus_npixels,
            focus_pixel_scale=focal_pixel_scale,
            mask_at_focus=mask_at_focus,
        ),
    ]
    if has_PIAA:
        layers += [inv_piaa_set]

    optics = dl.LayeredOpticalSystem(wf_npixels, sim_diameter, layers)
    detector = dl.LayeredDetector(layers=[dl.Downsample(det_downsample)])
    source = dl.PointSource(
        wavelengths=wavels,
        flux=photons_per_exposure,
    )
    tscope = dl.Telescope(
        optics=optics,
        detector=detector,
        source=source,
    )

    params = {}
    params["read_noise"] = read_noise
    params["phase_coefficients"] = np.zeros(len(phase_aberration_basis))

    if has_PIAA:
        params["piaa1_coeffs"] = sag1_asphere.coeffs
        params["piaa2_coeffs"] = sag2_asphere.coeffs
        params["piaa1_f_rho_max"] = sag1_asphere.f_rho_max
        params["piaa2_f_rho_max"] = sag2_asphere.f_rho_max
        params["piaa_distance"] = np.array(d)

        params["piaa1_lens1_decentre"] = np.array([0.0, 0.0])
        params["piaa1_lens2_decentre"] = np.array([0.0, 0.0])
        params["piaa2_lens1_decentre"] = np.array([0.0, 0.0])
        params["piaa2_lens2_decentre"] = np.array([0.0, 0.0])

    if isinstance(mask_at_focus, BasisGlassLayer):
        params["phase_mask_coeffs"] = mask_at_focus.coefficients
    elif isinstance(mask_at_focus, GlassDot):
        params["phase_mask_thickness"] = np.array(mask_at_focus.thickness)
        params["phase_mask_diameter"] = np.array(mask_at_focus.diameter)

    core_model = ModelCore(
        params,
        tel=tscope,
    )

    return core_model


class ResolvedUniformDiskSource(dl.PointSource):
    r_max: np.ndarray

    n_r_integral: int
    n_theta_integral: int

    def __init__(
        self: dl.sources.Source,
        r_max: np.ndarray,
        wavelengths=None,
        position=np.zeros(2),
        flux: float = 1.0,
        weights=None,
        spectrum: dl.Spectrum = None,
        n_r_integral: int = 20,
        n_theta_integral: int = 20,
    ):
        """Resolved uniform disk source used for incoherent imaging tests.

        The source integrates over a filled circular disk by quadrature in
        polar coordinates and returns an incoherent PSF (intensity image).
        """
        if isinstance(r_max, (float, int)):
            r_max = np.array(r_max, float)

        self.r_max = r_max

        self.n_r_integral = n_r_integral
        self.n_theta_integral = n_theta_integral

        super().__init__(
            position=position,
            flux=flux,
            spectrum=spectrum,
            wavelengths=wavelengths,
            weights=weights,
        )

    def model(
        self: dl.sources.Source,
        optics=None,
        return_wf: bool = False,
        return_psf: bool = False,
        debug=False,
    ):
        if return_wf:
            raise NotImplementedError(
                "ResolvedUniformDiskSource does not support return_wf=True (incoherent source!)."
            )
        npix = optics.wf_npixels
        full_img = np.zeros((npix, npix), dtype=np.float32)

        # Streamed quadrature over r and theta, JIT-compiled with no large intermediates
        for wavel_idx, wavel in enumerate(self.wavelengths):
            # Create a single-wavelength point source for evaluation
            test_source = dl.PointSource(
                position=self.position,
                flux=self.flux,
                spectrum=self.spectrum,
                wavelengths=np.array([wavel], dtype=np.float32),
            )

            # Get pixel scale once if PSF return requested
            if return_psf:
                test_psf = optics.model(test_source, return_psf=True)

            # Integration nodes (use integral grids only)
            r_vals = np.linspace(0.0, self.r_max, self.n_r_integral, dtype=np.float32)
            theta_vals = np.linspace(
                -np.pi, np.pi, self.n_theta_integral, endpoint=False, dtype=np.float32
            )

            # Trapezoid weights for uniform grids
            if self.n_r_integral > 1:
                dr = r_vals[1] - r_vals[0]
                wr = np.ones_like(r_vals)
                wr = wr.at[0].set(0.5)
                wr = wr.at[-1].set(0.5)
                wr = wr * dr
            else:
                wr = np.array([self.r_max], dtype=np.float32)

            if self.n_theta_integral > 1:
                dth = (2.0 * np.pi) / self.n_theta_integral
                wth = np.ones_like(theta_vals) * dth
            else:
                wth = np.array([2 * np.pi], dtype=np.float32)

            # Skip r=0 (contributes zero due to r factor; also theta-invariant)
            i_start = 1 if self.n_r_integral > 1 else 0

            # JIT-compiled accumulator for this wavelength using fori_loop over theta and r
            def integrate_wavelength(
                optics,
                test_source,
                r_vals,
                theta_vals,
                wr,
                wth,
                wavel_idx,
                i_start,
                npix,
            ):
                img0 = np.zeros((npix, npix), dtype=np.float32)

                def body_theta(j, img):
                    th = theta_vals[j]
                    costh = np.cos(th)
                    sinth = np.sin(th)

                    def body_r(i, img_r):
                        r = r_vals[i]
                        pos = np.array([r * costh, r * sinth], dtype=np.float32)
                        wf = optics.model(
                            test_source.set("position", pos), return_wf=True
                        )
                        intensity = np.abs(wf.phasor[wavel_idx]) ** 2
                        weight = wth[j] * wr[i] * r
                        return img_r + (weight * intensity.astype(np.float32))

                    img = jax.lax.fori_loop(i_start, r_vals.shape[0], body_r, img)
                    return img

                img = jax.lax.fori_loop(0, theta_vals.shape[0], body_theta, img0)
                # Normalise by disk area (analytical)
                area = np.pi * (self.r_max**2)
                img = img / area.astype(np.float32)
                return img

            integrate_wavelength_jit = eqx.filter_jit(integrate_wavelength)
            img = integrate_wavelength_jit(
                optics,
                test_source,
                r_vals,
                theta_vals,
                wr,
                wth,
                wavel_idx,
                i_start,
                npix,
            )

            full_img = full_img + img

        if return_psf:
            psf = dl.PSF(data=full_img, pixel_scale=test_psf.pixel_scale)
            return psf

        if debug:
            return full_img

        return full_img
