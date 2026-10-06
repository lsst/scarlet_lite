# This file is part of scarlet_lite.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Fit a sum of parametric profiles (Moffat/Gaussian) to a PSF image.

This module fits one or more elliptical parametric components to a
single-band target PSF under a per-pixel weight. It reuses the parametric
machinery in `~lsst.scarlet.lite.models.parametric` but runs its own
gradient loop -- there is no `Blend` or `Observation` and no PSF
convolution: the gradient handed to each component is simply the weighted
residual ``weight * (model - target)``.

The motivating use is extending a truncated PSF (e.g. a PIFF model) by
fitting the wings of an empirical, star-stacked PSF with the asymmetric
core masked out (``core_radius``), and splicing the parametric wing model
onto the accurate core. The fitter is target-agnostic, though: it fits
whatever image and weight it is handed, and with ``core_radius == 0`` it is
a general parametric PSF fitter.

Everything is done in a frame centered on the PSF center (coordinate
``(0, 0)`` at the central pixel; see `Box.centered`), so a fit performed on
one image size can be rendered analytically on any other size via
`PsfFitResult.evaluate`.
"""

from __future__ import annotations

__all__ = [
    "PsfTarget",
    "WingProfile",
    "MoffatProfile",
    "GaussianProfile",
    "PsfFitResult",
    "PsfFitter",
    "default_psf_adaprox_parameterization",
    "default_psf_fista_parameterization",
]

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, cast

import numpy as np

from ..bbox import Box
from ..image import Image
from ..parameters import AdaproxParameter, FistaParameter, relative_step
from .parametric import (
    CartesianFrame,
    EllipseFrame,
    EllipticalParametricComponent,
    bounded_prox,
    gaussian2d,
    grad_gaussian2,
    grad_moffat,
    moffat,
)

# Single-component models use a placeholder band, since the fit is per-band.
_BAND: tuple[str, ...] = ("",)


class PsfTarget:
    """The target PSF image and its per-pixel weight for a fit.

    Bundles the data, weight, and (centered) geometry that the fitter
    consumes. The frame is always centered on the central pixel, so the fit
    and any rendered model share one coordinate system. The derived
    quantities are computed once at construction; treat instances as
    immutable.

    Parameters
    ----------
    data:
        The single-band 2D image to fit (e.g. a stacked, normalized star).
    weight:
        Per-pixel weight, same shape as ``data``. Zero marks pixels excluded
        from the fit (bad/interpolated pixels, masked neighbors); elsewhere
        this is typically an inverse variance. Defaults to all ones.
    """

    def __init__(self, data: np.ndarray, weight: np.ndarray | None = None):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"data must be 2D, got shape {data.shape}")
        if weight is None:
            weight = np.ones_like(data)
        else:
            weight = np.asarray(weight, dtype=float)
            if weight.shape != data.shape:
                raise ValueError(f"weight shape {weight.shape} != data shape {data.shape}")

        self.data = data
        self.weight = weight
        self.bbox = Box.centered(data.shape)
        self.center = self.bbox.int_center
        self.peak = float(np.max(data))

    @classmethod
    def from_image(cls, image: Image, weight: np.ndarray | None = None) -> PsfTarget:
        """Build a target from a single-band `~lsst.scarlet.lite.Image`.

        Parameters
        ----------
        image:
            A 2D scarlet image. Its data are re-centered; any origin is
            ignored.
        weight:
            Optional per-pixel weight.

        Returns
        -------
        target:
            A new `PsfTarget`.
        """
        return cls(np.asarray(image.data, dtype=float), weight)


class WingProfile(EllipticalParametricComponent):
    """An elliptical parametric component used to fit part of a PSF.

    This is a thin specialization of `EllipticalParametricComponent` with a
    friendlier constructor: it assembles the ellipse parameters, bounds, and
    proximal operator from human-readable arguments. Concrete subclasses
    (`MoffatProfile`, `GaussianProfile`) pass their radial function and
    gradient and any extra radial parameters.

    The radial scale is carried by the ellipse semi-axes; ``semi_major`` is
    the semi-major scale (the Moffat ``alpha`` or Gaussian ``sigma``) and the
    semi-minor scale follows from ``axis_ratio``.

    Parameters
    ----------
    bbox:
        The (centered) frame the component is built on. The center is taken
        from ``bbox.int_center``.
    semi_major:
        Initial semi-major scale, in pixels.
    func, grad:
        The radial model and its gradient (set by subclasses).
    axis_ratio:
        Initial ratio of the semi-minor to the semi-major scale. Seeded below
        one to break the circular symmetry that would zero the rotation-angle
        gradient.
    angle:
        Initial counter-clockwise rotation angle, in radians.
    amplitude:
        Initial central amplitude. Appropriate when the target is
        peak-normalized; override otherwise.
    center_window:
        The center may move at most this many pixels from its initial value.
    min_size, max_size:
        Bounds on the semi-axes. ``max_size`` defaults to the larger box
        dimension.
    step:
        Optimizer step size for the morphology parameters.
    flux_factor:
        Relative step factor for the amplitude.
    radial_init, radial_lo, radial_hi:
        Initial values and bounds for the radial parameters that follow
        ``[y0, x0, major, minor, theta]`` (e.g. ``beta`` for a Moffat). Set
        by subclasses.
    """

    def __init__(
        self,
        bbox: Box,
        *,
        semi_major: float,
        func: Callable,
        grad: Callable,
        axis_ratio: float = 0.9,
        angle: float = 0.0,
        amplitude: float = 1.0,
        center_window: float = 3.0,
        min_size: float = 0.5,
        max_size: float | None = None,
        step: float = 0.2,
        flux_factor: float = 0.1,
        radial_init: tuple[float, ...] = (),
        radial_lo: tuple[float, ...] = (),
        radial_hi: tuple[float, ...] = (),
    ):
        cy, cx = bbox.int_center
        major = float(semi_major)
        minor = float(semi_major) * float(axis_ratio)
        if max_size is None:
            max_size = float(max(bbox.shape[-2:]))

        params = np.array([cy, cx, major, minor, angle, *radial_init], dtype=float)
        proxmin = np.array(
            [cy - center_window, cx - center_window, min_size, min_size, -np.pi, *radial_lo],
            dtype=float,
        )
        proxmax = np.array(
            [cy + center_window, cx + center_window, max_size, max_size, np.pi, *radial_hi],
            dtype=float,
        )

        super().__init__(
            bands=_BAND,
            bbox=bbox,
            spectrum=np.array([amplitude], dtype=float),
            morph_params=params,
            morph_func=func,
            morph_grad=grad,
            morph_prox=partial(bounded_prox, proxmin=proxmin, proxmax=proxmax),
            morph_step=np.full(params.shape, step),
        )
        # Per-component optimizer steps, consumed by ``PsfFitter``.
        self.fit_step = float(step)
        self.fit_flux_factor = float(flux_factor)


class MoffatProfile(WingProfile):
    """A `WingProfile` for an elliptical Moffat ``(1 + r**2)**(-beta)``.

    Parameters
    ----------
    bbox:
        The frame to build on.
    alpha:
        Initial semi-major core width, in pixels (the semi-minor follows from
        ``axis_ratio``).
    beta:
        Initial Moffat index. Larger ``beta`` gives a more peaked profile with
        weaker wings; ``beta == 1`` is an ``r**-2`` aureole.
    beta_bounds:
        ``(min, max)`` bounds on the Moffat index. The lower bound is a loose
        sanity floor, not a normalizability requirement (the profile is not
        normalized).
    **kwargs:
        Forwarded to `WingProfile`.
    """

    def __init__(
        self,
        bbox: Box,
        *,
        alpha: float,
        beta: float = 3.0,
        beta_bounds: tuple[float, float] = (0.5, 10.0),
        **kwargs: Any,
    ):
        super().__init__(
            bbox,
            semi_major=alpha,
            func=moffat,
            grad=grad_moffat,
            radial_init=(beta,),
            radial_lo=(beta_bounds[0],),
            radial_hi=(beta_bounds[1],),
            **kwargs,
        )


class GaussianProfile(WingProfile):
    """A `WingProfile` for an elliptical Gaussian ``exp(-r**2)``.

    The Gaussian has no radial parameters beyond the ellipse geometry. It
    falls off too fast to model the PSF aureole and is intended only for the
    inner shoulder.

    Parameters
    ----------
    bbox:
        The frame to build on.
    sigma:
        Initial semi-major width, in pixels (the semi-minor follows from
        ``axis_ratio``).
    **kwargs:
        Forwarded to `WingProfile`.
    """

    def __init__(self, bbox: Box, *, sigma: float, **kwargs: Any):
        super().__init__(
            bbox,
            semi_major=sigma,
            func=gaussian2d,
            grad=grad_gaussian2,
            **kwargs,
        )


@dataclass
class PsfFitResult:
    """The result of a `PsfFitter` fit.

    Parameters
    ----------
    components:
        The fitted components, in the order they were added to the fitter.
    loss:
        The weighted-residual L2 norm at each iteration.
    converged:
        Whether the fit stopped on the ``e_rel`` criterion (rather than
        running out of iterations) with finite parameters.
    """

    components: list[WingProfile]
    loss: list[float]
    converged: bool

    def evaluate(self, shape: tuple[int, int]) -> Image:
        """Render the fitted model on a centered frame of the given shape.

        Because the model is analytic, the fit can be performed on one image
        size and rendered on another (e.g. fit on the large star-stack stamp,
        render on the smaller output PSF stamp).

        Parameters
        ----------
        shape:
            The ``(height, width)`` of the output frame.

        Returns
        -------
        model:
            The summed model as a centered 2D `~lsst.scarlet.lite.Image`.
        """
        bbox = Box.centered(shape)
        data = np.zeros(shape, dtype=float)
        for component in self.components:
            ellipse = EllipseFrame(*component.ellipse_params, bbox)  # type: ignore
            data += component.get_model(frame=ellipse).data[0]
        return Image(data, yx0=cast("tuple[int, int]", bbox.origin))


def default_psf_adaprox_parameterization(component: WingProfile) -> None:
    """Wrap a profile's arrays as proximal-ADAM (adaprox) parameters.

    This is the default parameterization `PsfFitter` applies when its
    optimizer is ``"adaprox"``. It reads the per-profile optimizer steps
    that `WingProfile` stores at construction (``fit_step`` for the
    morphology and ``fit_flux_factor`` for the amplitude), so the single
    callable parameterizes every profile consistently. Pass it -- or a
    custom callable with the same signature -- to `PsfFitter.parameterize`.

    Parameters
    ----------
    component:
        The profile whose ``spectrum`` and morphology arrays are converted
        to `AdaproxParameter` instances in place.
    """
    component._spectrum = AdaproxParameter(
        component.spectrum,
        step=partial(relative_step, factor=component.fit_flux_factor, minimum=1e-16),
    )
    component._params = AdaproxParameter(component._params.x.copy(), step=component.fit_step)


def default_psf_fista_parameterization(component: WingProfile) -> None:
    """Wrap a profile's arrays as FISTA parameters.

    This is the default parameterization `PsfFitter` applies when its
    optimizer is ``"fista"``; it reads the same per-profile steps as
    `default_psf_adaprox_parameterization`.

    Parameters
    ----------
    component:
        The profile whose ``spectrum`` and morphology arrays are converted
        to `FistaParameter` instances in place.
    """
    component._spectrum = FistaParameter(component.spectrum, step=component.fit_flux_factor)
    component._params = FistaParameter(component._params.x.copy(), step=component.fit_step)


class PsfFitter:
    """Fit a sum of parametric profiles to a single-band PSF.

    The fitter owns the `PsfTarget` (and hence the frame), so profiles added
    to it are guaranteed consistent, and `fit` needs no arguments. The
    target, optimizer, and frame are read-only after construction; make a new
    fitter to fit a different target.

    Parameters
    ----------
    target:
        The PSF image and weight to fit.
    optimizer:
        Either ``"adaprox"`` (proximal ADAM) or ``"fista"``.
    core_radius:
        If positive, pixels within this radius of the center are given zero
        weight. The core is the asymmetric, non-parametric part of the PSF
        that will be replaced (e.g. by the PIFF model), so it must not bias
        the wing fit.
    max_iter:
        Maximum number of optimizer iterations.
    e_rel:
        Relative change in the loss below which the fit has converged.
    min_iter:
        Minimum number of iterations before the stopping criterion is checked.
    """

    def __init__(
        self,
        target: PsfTarget,
        *,
        optimizer: str = "adaprox",
        core_radius: float = 0.0,
        max_iter: int = 500,
        e_rel: float = 1e-6,
        min_iter: int = 15,
    ):
        if optimizer not in ("adaprox", "fista"):
            raise ValueError(f"Unknown optimizer {optimizer!r}, expected 'adaprox' or 'fista'.")
        self._target = target
        self._optimizer = optimizer
        self._core_radius = core_radius
        self._max_iter = max_iter
        self._e_rel = e_rel
        self._min_iter = min_iter
        self._profiles: list[WingProfile] = []
        self._parameterized = False

    @property
    def target(self) -> PsfTarget:
        """The PSF target being fit (read-only)."""
        return self._target

    @property
    def bbox(self) -> Box:
        """The centered frame of the fit (read-only)."""
        return self._target.bbox

    @property
    def optimizer(self) -> str:
        """The optimizer name (read-only)."""
        return self._optimizer

    @property
    def profiles(self) -> tuple[WingProfile, ...]:
        """The profiles added so far (read-only)."""
        return tuple(self._profiles)

    def add_profile(self, profile_cls: type[WingProfile], **kwargs: Any) -> WingProfile:
        """Add a profile of the given class to the fit.

        The fitter injects the frame (``bbox``); all other parameters come
        from ``kwargs`` or the profile class defaults. For example::

            fitter.add_profile(MoffatProfile, alpha=4, beta=3.0)
            fitter.add_profile(GaussianProfile, sigma=2)

        Parameters
        ----------
        profile_cls:
            A `WingProfile` subclass.
        **kwargs:
            Keyword arguments for ``profile_cls`` (e.g. ``alpha``/``beta`` for
            a Moffat, ``sigma`` for a Gaussian).

        Returns
        -------
        profile:
            The component that was added.
        """
        if not (isinstance(profile_cls, type) and issubclass(profile_cls, WingProfile)):
            raise TypeError(f"{profile_cls!r} is not a WingProfile subclass")
        profile = profile_cls(self._target.bbox, **kwargs)
        self._profiles.append(profile)
        # A freshly added profile holds raw arrays, so the fit must (re-)wrap
        # every profile before it can run.
        self._parameterized = False
        return profile

    def parameterize(self, parameterization: Callable[[WingProfile], None] | None = None) -> None:
        """Wrap every added profile's arrays as optimizer parameters.

        Mirrors `~lsst.scarlet.lite.Blend.parameterize`: it applies one
        callable to each profile in turn, converting the raw parameter
        arrays into `~lsst.scarlet.lite.parameters.Parameter` instances that
        carry the optimizer's step rule. Call it before `fit` to install a
        custom optimization scheme; otherwise `fit` applies
        `default_parameterization` automatically.

        Parameters
        ----------
        parameterization:
            A callable taking a single `WingProfile` and converting its
            arrays to parameters in place (see
            `default_psf_adaprox_parameterization`). Defaults to
            `default_parameterization` for this fitter's optimizer.
        """
        if parameterization is None:
            if self._optimizer == "adaprox":
                parameterization = default_psf_adaprox_parameterization
            else:
                parameterization = default_psf_fista_parameterization
        for profile in self._profiles:
            profile.parameterize(parameterization)
        self._parameterized = True

    def fit(self) -> PsfFitResult:
        """Fit all added profiles to the target.

        Runs a proximal-gradient loop with no observation or convolution: the
        gradient handed to each component is the weighted residual
        ``weight * (model - target)``. The core (if ``core_radius > 0``) is
        given zero weight on top of the target weight.

        Unless `parameterize` was already called, the profiles are wrapped
        with `default_parameterization` first; call `parameterize` with a
        custom callable beforehand to override the optimizer scheme.

        Returns
        -------
        result:
            The fitted components, loss history, and convergence flag.
        """
        if not self._profiles:
            raise ValueError("Add at least one profile before fitting.")

        bbox = self._target.bbox
        weight = np.array(self._target.weight, dtype=float, copy=True)
        if self._core_radius > 0:
            radius = CartesianFrame(bbox).unscaled_radius_grid
            weight[radius < self._core_radius] = 0.0

        # Wrap the profiles with the default optimizer parameters unless the
        # caller already installed a (possibly custom) parameterization.
        if not self._parameterized:
            self.parameterize()

        target3 = self._target.data[None]
        weight3 = weight[None]
        loss: list[float] = []
        converged = False
        for it in range(self._max_iter):
            model = np.zeros_like(target3)
            for profile in self._profiles:
                model += profile.get_model().data
            residual = (model - target3) * weight3
            loss.append(float(np.sqrt(np.sum(residual**2))))
            for profile in self._profiles:
                profile.update(it, residual)
            if it > self._min_iter and np.abs(loss[-1] - loss[-2]) < self._e_rel * np.abs(loss[-1]):
                converged = True
                break

        finite = all(
            np.all(np.isfinite(profile._params.x)) and np.all(np.isfinite(profile.spectrum))
            for profile in self._profiles
        )
        return PsfFitResult(
            components=list(self._profiles),
            loss=loss,
            converged=converged and finite,
        )
