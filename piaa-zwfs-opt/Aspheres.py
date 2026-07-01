import jax.numpy as np
import numpy as onp
import zodiax as zdx
import dLux as dl
import jax
import dlMaterials
import dLux.utils as dlu
from abc import abstractmethod
import equinox as eqx
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
try:
    QBFScoeffs = onp.load(_MODULE_DIR / "Qbfs_numeric_coeffs.npy")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Qbfs coefficients file 'Qbfs_numeric_coeffs.npy' not found in {_MODULE_DIR}."
    )


class MildAsphere(zdx.base.Base):  # TODO: maybe becomes dlu?
    rho_max: float
    f_rho_max: float
    M: int
    coeffs: np.ndarray
    scale: eqx.field(static=True)

    """
    Made with Qbfs

    Parameters
    ----------
    aperture_radius : float
        Maximum radius of the asphere (rho_max in the paper)
    sag_at_edge : float
        Sag at the edge of the aperture (f(rho_max) in the paper)
    n_terms : int
        Number of Q basis functions to use (M in the paper)
    init_coeffs : array-like, optional
        Initial coefficients for the Q basis functions (a_m in the paper), by default None
    scale : float, optional
        Scale factor for the sag coefficients and f_rho_max, by default 1.0

    [1] "Shape specification for axially symmetric optical surfaces"
    """

    Qpoly_coeffs = np.array(QBFScoeffs)

    def __init__(
        self,
        aperture_radius,
        sag_at_edge,
        n_terms,
        init_coeffs=None,
        scale=1.0,
    ):
        self.rho_max = aperture_radius
        self.f_rho_max = sag_at_edge
        self.M = int(n_terms)

        if self.M > self.Qpoly_coeffs.shape[0]:
            raise ValueError(
                f"n_terms {self.M} exceeds maximum available {self.Qpoly_coeffs.shape[0]}"
            )

        if init_coeffs is None:
            self.coeffs = np.zeros(n_terms)
        else:
            self.coeffs = init_coeffs

        self.scale = scale

    @property
    def _scaled_f_rho_max(self):
        return self.f_rho_max * self.scale

    @property
    def c_bfs(self):
        return (
            2 * self._scaled_f_rho_max / (self.rho_max**2 + self._scaled_f_rho_max**2)
        )

    def sag(self, rho):
        first = self.c_bfs * rho**2 / (1 + np.sqrt(1 - self.c_bfs**2 * rho**2))
        second = self.D_bfs(rho / self.rho_max)
        return np.where(
            rho > self.rho_max,
            self._scaled_f_rho_max,
            first + second,
        )

    def D_bfs(self, u):
        """
        Eqn 11
        """
        factor = u**2 * (1 - u**2) / np.sqrt(1 - self.c_bfs**2 * self.rho_max**2 * u**2)
        summation = (
            np.sum(
                np.array(
                    [self.coeffs[m] * self.Qbfs(m, u**2) for m in range(int(self.M))]
                ),
                axis=0,
            )
            * self.scale
        )

        return factor * summation

    def Qbfs(self, m, x):
        """Evaluate Q_m at x using precomputed coefficients."""
        a_coeffs = self.Qpoly_coeffs[m]
        return np.polyval(a_coeffs, x)

    @staticmethod
    def from_target_sag(
        target_rhos,
        target_sags,
        n_terms,
        return_misfit=False,
        scale=1e-6,
    ):
        """
        Fit a mild asphere to a target sag profile using least squares.

        Parameters
        ----------
        target_rhos : array-like
            Radial positions where the sag is specified.
        target_sags : array-like
            Target sag values at the specified radial positions.

        Returns
        -------
        MildAsphere
            Fitted MildAsphere instance.
        """
        # Initial guess for aperture radius and sag at edge
        aperture_radius = np.max(target_rhos)
        sag_at_edge = target_sags[np.argmax(target_rhos)] / scale

        sphere = MildAsphere(
            aperture_radius=aperture_radius,
            sag_at_edge=sag_at_edge,
            n_terms=n_terms,
            scale=scale,
        )

        # Compute nominal sag with zero coefficients
        sphere = sphere.set("coeffs", np.zeros(sphere.M))
        nominal_sag = sphere.sag(target_rhos)

        # Build matrix for least squares fitting
        A = np.zeros((target_rhos.size, sphere.M))
        for m in range(sphere.M):
            coeffs_vec = np.zeros(sphere.M)
            coeffs_vec = coeffs_vec.at[m].set(1.0)
            sphere = sphere.set("coeffs", coeffs_vec)
            A = A.at[:, m].set(sphere.sag(target_rhos) - nominal_sag)

        desired_coeffs = np.linalg.lstsq(A, target_sags - nominal_sag, rcond=None)[0]
        sphere = sphere.set("coeffs", desired_coeffs)

        if return_misfit:
            fitted_sag = sphere.sag(target_rhos)
            return sphere, fitted_sag - target_sags

        return sphere


def _fft(phasor, pad=2):
    padded = dlu.resize(phasor, phasor.shape[0] * pad)
    return 1 / padded.shape[0] * np.fft.fft2(padded)


def _ifft(phasor, pad=1):
    padded = dlu.resize(phasor, phasor.shape[0] * pad)
    return phasor.shape[0] * np.fft.ifft2(padded)


def _fftshift(phasor):
    return np.fft.fftshift(phasor)


def transfer_fn(coords, npixels, wavelength, pscale, distance):
    rho_sq = (coords**2).sum(0)
    return _fftshift(np.exp(-1.0j * np.pi * wavelength * distance * rho_sq))


def transfer(wf, distance, pad=2):
    npix = pad * wf.npixels
    diam = pad * wf.diameter
    freqs = np.fft.fftshift(np.fft.fftfreq(npix, diam / npix))
    coords = np.array(np.meshgrid(freqs, freqs))
    return transfer_fn(
        coords, wf.npixels, wf.wavelength, pad * wf.pixel_scale, distance
    )


def plane_to_plane(wf, distance, pad=2):
    fft_wf = _fft(wf.phasor, pad=pad)
    tf = transfer(wf, distance, pad=pad)
    phasor = dlu.resize(_ifft(fft_wf * tf), wf.npixels)
    return wf.set(["phasor"], [phasor])


def thickness_to_opd(thickness, n):
    return thickness * (n - 1.0)


class AbstractRefractiveOptic(dl.BaseLayer):
    material: dlMaterials.Material

    @abstractmethod
    def apply(self, wavefront: dl.Wavefront) -> dl.Wavefront:
        pass


class ParametricRefractiveOptic(AbstractRefractiveOptic):
    @abstractmethod
    def generate_thickness(self) -> np.ndarray:
        pass

    def apply(self, wavefront: dl.Wavefront) -> dl.Wavefront:
        thickness = self.generate_thickness()
        n = self.material.n(wavefront.wavelength)
        opd = thickness_to_opd(thickness, n)
        return wavefront.add_opd(opd)


class AsphericLens(ParametricRefractiveOptic):
    asphere_profile: MildAsphere
    material: dlMaterials.Material
    coords: np.ndarray
    is_inverse: bool = False
    decentre: np.ndarray
    decentre_scale: eqx.field(static=True)

    def __init__(
        self,
        asphere_profile,
        material,
        coords,
        is_inverse=False,
        decentre=None,
        decentre_scale=1.0,
    ):
        self.asphere_profile = asphere_profile
        self.material = material
        self.coords = coords
        self.is_inverse = is_inverse
        self.decentre_scale = decentre_scale

        if decentre is None:
            self.decentre = np.zeros(2)
        else:
            self.decentre = decentre

    def _coords_to_rhos(self):
        shifted_coords = dlu.translate_coords(
            self.coords, self.decentre * self.decentre_scale
        )
        rhos = dlu.cart2polar(shifted_coords)[0]
        return rhos

    def generate_thickness(self) -> np.ndarray:
        sag = self.asphere_profile.sag(self._coords_to_rhos())
        if self.is_inverse:
            sag = -sag
        return sag

    def __call__(self, wavefront: dl.Wavefront) -> dl.Wavefront:
        return self.apply(wavefront)


class PIAAset(dl.optical_layers.OpticalLayer):
    piaa_lens_1: AsphericLens
    piaa_lens_2: AsphericLens
    distance: np.ndarray
    is_inverse: bool

    def __init__(self, piaa_lens_1, piaa_lens_2, distance, is_inverse=False):
        """
        Initialize a PIAA set with two aspheric lenses and a propagation distance.
        Parameters
        ----------
        piaa_lens_1 : AsphericLens
            The first PIAA lens.
        piaa_lens_2 : AsphericLens
            The second PIAA lens.
        distance : float
            The distance between the two lenses.
        is_inverse : bool, optional
            If True, applies the inverse PIAA operation, by default False.
        """
        if isinstance(distance, (int, float)):
            distance = np.array(distance)

        self.piaa_lens_1 = piaa_lens_1
        self.piaa_lens_2 = piaa_lens_2
        self.distance = distance
        self.is_inverse = is_inverse

    def apply(self, wavefront: dl.Wavefront) -> dl.Wavefront:
        """
        Propagates the input wavefront through the PIAA lens pair using
        the `plane_to_plane` function defined above.

        Parameters
        ----------
        wavefront : dl.Wavefront
            Input wavefront.
        return_intermediate_wfs : bool, optional
            If True, returns a list of wavefronts at all planes. The list
            ordering is:
                [input,
                 after first lens applied,
                 after first propagation,
                 after second lens applied,
                 after second propagation (final output)]
            For the inverse case the "first" and "second" lenses are swapped
            accordingly, but the ordering semantics remain the same.

        Returns
        -------
        dl.Wavefront
            Final output wavefront.
        """
        if self.is_inverse:
            # Propagate between lenses (inverse direction)
            wavefront = plane_to_plane(wavefront, self.distance, pad=2)
            # First lens (inverse order)
            wavefront = self.piaa_lens_2.apply(wavefront)
            # Final propagation back
            wavefront = plane_to_plane(wavefront, -self.distance, pad=2)
            # Second lens (inverse order)
            wavefront = self.piaa_lens_1.apply(wavefront)
        else:
            # First lens
            wavefront = self.piaa_lens_1.apply(wavefront)
            # Propagate between lenses
            wavefront = plane_to_plane(wavefront, self.distance, pad=2)
            # Second lens
            wavefront = self.piaa_lens_2.apply(wavefront)
            # Final propagation out
            wavefront = plane_to_plane(wavefront, -self.distance, pad=2)

        return wavefront

    def plot_both_profiles(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        rhos = np.linspace(0, self.piaa_lens_1.asphere_profile.rho_max, 500)
        sag1 = self.piaa_lens_1.asphere_profile.sag(rhos)
        sag2 = self.piaa_lens_2.asphere_profile.sag(rhos)

        ax.plot(rhos, sag1 * 1e6, label="Lens 1 Sag (um)")
        ax.plot(rhos, sag2 * 1e6, label="Lens 2 Sag (um)")
        ax.set_xlabel("Radius (m)")
        ax.set_ylabel("Sag (um)")
        ax.legend()
        ax.set_title("PIAA Lens Profiles")
        return ax

    def __call__(self, wavefront: dl.Wavefront) -> dl.Wavefront:
        return self.apply(wavefront)
