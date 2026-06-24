"""
distributions.py
----------------
Stellar population distribution utilities for StarWave.

Provides:

* :func:`nearestPD`        – project a matrix onto the nearest positive-definite matrix.
* :func:`isPD`             – test whether a matrix is positive-definite.
* :class:`exponential_decay` – exponential SFH age distribution.
* :class:`SW_SFH`          – 2-D (age, [Fe/H]) SFH wrapper for SciPy distributions.
* :class:`Emp_MDF_Sci_Age` – SFH sampler combining an empirical MDF with a
                              SciPy age distribution.
* :class:`GridSFH`         – grid-based SFH sampler.
* :func:`set_GR_spl`       – single power-law (Salpeter) IMF sampler.
* :func:`set_GR_bpl`       – broken power-law (Kroupa) IMF sampler.
* :func:`set_GR_ln10full`  – lognormal + power-law (Chabrier) IMF sampler.
* :func:`set_GR_dgdm`      – double-Gaussian distance-modulus sampler.
* :func:`set_GR_unif`      – uniform binary mass-ratio sampler.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import sys

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import numpy.linalg as la
from scipy import stats

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
sys.path.append(dir_path)

from generalrandom import GeneralRandom


# ===========================================================================
# Positive-definiteness utilities
# ===========================================================================

def nearestPD(A):
    """
    Find the nearest positive-definite matrix to a given input matrix.

    A Python/NumPy port of John D'Errico's ``nearestSPD`` MATLAB code [1],
    which in turn credits Higham (1988) [2].

    The algorithm symmetrises *A*, computes its polar decomposition, and
    iteratively nudges negative eigenvalues until Cholesky decomposition
    succeeds.

    Parameters
    ----------
    A : ndarray of shape (N, N)
        Input (possibly indefinite) square matrix.

    Returns
    -------
    A3 : ndarray of shape (N, N)
        Nearest symmetric positive-definite matrix to *A*.

    References
    ----------
    .. [1] John D'Errico, *nearestSPD*,
           https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    .. [2] N. J. Higham, "Computing a nearest symmetric positive semidefinite
           matrix" (1988), https://doi.org/10.1016/0024-3795(88)90223-6

    Notes
    -----
    The ``spacing`` parameter differs from D'Errico's MATLAB implementation.
    MATLAB's ``chol`` accepts matrices with exactly zero eigenvalues whereas
    NumPy's does not. Consequently this implementation uses
    ``np.spacing(la.norm(A))`` rather than ``eps(mineig)``. In practice both
    approaches converge to the same result.
    """
    # Symmetrise A to remove numerical asymmetry
    B = (A + A.T) / 2

    # Compute the symmetric polar factor H via SVD
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))

    # Average B and H; symmetrise again to eliminate floating-point drift
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    # Return early if A3 is already positive-definite
    if isPD(A3):
        return A3

    # Iteratively shift the matrix by a small multiple of the identity until
    # it passes the Cholesky test.
    spacing = np.spacing(la.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def isPD(B):
    """
    Test whether a matrix is positive-definite via Cholesky decomposition.

    Parameters
    ----------
    B : ndarray of shape (N, N)
        Square matrix to test.

    Returns
    -------
    bool
        ``True`` if *B* is positive-definite, ``False`` otherwise.
    """
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


# ===========================================================================
# Star-formation history distributions
# ===========================================================================

class exponential_decay:
    """
    Exponential star-formation history that decays back in time from a onset age.

    The probability density is:

    .. math::

        p(x) = \\frac{1}{\\text{scale}}
                \\exp\\!\\left(-\\frac{\\text{loc} - x}{\\text{scale}}\\right),
                \\quad x < \\text{loc}

    Parameters
    ----------
    loc : float, optional
        Location parameter — the maximum (onset) age in Gyr. Default is ``0``.
    scale : float, optional
        Scale parameter — the inverse decay rate (e-folding time) in Gyr.
        Default is ``1``.

    Examples
    --------
    >>> ed = exponential_decay(loc=10.0, scale=2.0)
    >>> ages = ed.rvs(500)   # 500 ages drawn from the distribution
    """

    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def rvs(self, N):
        """
        Draw *N* random ages from the exponential decay distribution.

        Parameters
        ----------
        N : int
            Number of samples to draw.

        Returns
        -------
        ndarray of shape (N,)
            Sampled ages in Gyr, all less than ``self.loc``.
        """
        # Negate a standard exponential (loc=−self.loc) to obtain ages < loc
        return -stats.expon.rvs(scale=self.scale, loc=-self.loc, size=N)


class SW_SFH:
    """
    SciPy-distribution wrapper that adds a bounds-clipped ``.sample()`` method.

    Draws ``(age, [Fe/H])`` pairs from an arbitrary 2-D SciPy distribution
    and masks any samples that fall outside the allowed age and metallicity
    ranges by setting them to ``NaN``.

    Parameters
    ----------
    scipy_dist : scipy.stats distribution
        A 2-D frozen distribution with an ``.rvs(N)`` method returning an
        array of shape ``(N, 2)`` with columns ``[age, feh]``.
    age_range : tuple of float
        ``(min_age, max_age)`` in Gyr.
    feh_range : tuple of float
        ``(min_feh, max_feh)``.
    """

    def __init__(self, scipy_dist, age_range, feh_range):
        self.scipy_dist = scipy_dist
        self.l_age = age_range[0]
        self.u_age = age_range[1]
        self.l_feh = feh_range[0]
        self.u_feh = feh_range[1]

    def sample(self, N):
        """
        Draw *N* ``(age, [Fe/H])`` pairs, masking out-of-range samples.

        Parameters
        ----------
        N : int
            Number of samples to draw.

        Returns
        -------
        ndarray of shape (N, 2)
            Each row is ``[age, feh]``. Rows where either value falls
            outside the allowed range are set to ``[NaN, NaN]``.
        """
        sfh = self.scipy_dist.rvs(N)
        age, feh = sfh.T

        # Build a boolean mask: True where both age and feh are in range
        within = (
            (age > self.l_age)
            & (age < self.u_age)
            & (feh > self.l_feh)
            & (feh < self.u_feh)
        )

        # Set out-of-range samples to NaN so callers can filter them easily
        age[~within] = np.nan
        feh[~within] = np.nan

        return np.vstack((age, feh)).T


class Emp_MDF_Sci_Age:
    """
    SFH sampler combining an empirical metallicity distribution with a SciPy age distribution.

    Metallicities are drawn from a :class:`~generalrandom.GeneralRandom`
    object built from an empirical [Fe/H] distribution; ages are drawn
    from any SciPy-compatible distribution. Samples outside the allowed
    age and metallicity ranges are masked to ``NaN``.

    Parameters
    ----------
    age_dist : scipy.stats distribution or :class:`exponential_decay`
        Age distribution in Gyr. Must expose an ``.rvs(N)`` method.
    feh_gr : GeneralRandom
        Empirical metallicity distribution. Must expose a ``.sample(N)``
        method.
    age_range : tuple of float
        ``(min_age, max_age)`` in Gyr.
    feh_range : tuple of float
        ``(min_feh, max_feh)``.

    Examples
    --------
    >>> feh_gr = GeneralRandom(feh_values, feh_weights, len(feh_values))
    >>> age_dist = scipy.stats.norm(loc=10.0, scale=1.0)
    >>> sfh = Emp_MDF_Sci_Age(age_dist, feh_gr, age_range=(1, 13.4), feh_range=(-4, 1))
    >>> samples = sfh.sample(1000)   # shape (1000, 2) array of [age, feh]
    """

    def __init__(self, age_dist, feh_gr, age_range, feh_range):
        self.age_dist = age_dist
        self.feh_gr = feh_gr
        self.l_age = age_range[0]
        self.u_age = age_range[1]
        self.l_feh = feh_range[0]
        self.u_feh = feh_range[1]

    def sample(self, N):
        """
        Draw *N* ``(age, [Fe/H])`` pairs from the respective distributions.

        Parameters
        ----------
        N : int
            Number of samples to draw.

        Returns
        -------
        ndarray of shape (N, 2)
            Each row is ``[age, feh]``. Out-of-range samples are set to
            ``[NaN, NaN]``.
        """
        age = self.age_dist.rvs(N)
        feh = self.feh_gr.sample(N)

        # Mask samples that fall outside the allowed (age, feh) window
        within = (
            (age > self.l_age)
            & (age < self.u_age)
            & (feh > self.l_feh)
            & (feh < self.u_feh)
        )
        age[~within] = np.nan
        feh[~within] = np.nan

        return np.vstack((age, feh)).T


class GridSFH:
    """
    Grid-based SFH sampler that draws from a discrete (age, [Fe/H]) probability grid.

    Ages and metallicities are sampled in two steps:

    1. A grid cell is selected with probability proportional to the supplied
       weights.
    2. A uniform random offset within the cell is added to produce a
       continuous sample.

    Parameters
    ----------
    sfh_grid : dict
        Dictionary with the following required keys:

        * ``'mets'``          – array of shape ``(M+1,)`` with [Fe/H] bin edges.
        * ``'ages'``          – array of shape ``(A+1,)`` with age (Gyr) bin edges.
        * ``'probabilities'`` – array of shape ``(M*A,)`` or ``(M, A)`` with
          unnormalised weights for each ``(metallicity, age)`` cell.

    Examples
    --------
    >>> sfh_grid = {
    ...     'mets': np.linspace(-2, 0, 11),
    ...     'ages': np.linspace(1, 13, 13),
    ...     'probabilities': np.ones((10, 12)),
    ... }
    >>> grid_sfh = GridSFH(sfh_grid)
    >>> samples = grid_sfh.sample(500)   # shape (500, 2) array of [age, feh]
    """

    def __init__(self, sfh_grid):
        self.sfh_grid = sfh_grid

        mets = sfh_grid["mets"]
        ages = sfh_grid["ages"]
        probs = sfh_grid["probabilities"]

        # Build a meshgrid of bin left-edges (drop the last edge of each axis)
        MM, AA = np.meshgrid(mets[:-1], ages[:-1])

        # Flatten grid coordinates and normalise weights to a probability vector
        self.mm = MM.ravel()   # metallicity left-edge for each cell
        self.aa = AA.ravel()   # age left-edge for each cell
        self.pp = probs.ravel() / np.sum(probs)   # normalised cell probabilities
        self.idxs = np.arange(len(self.pp))

        # Cell widths used for the uniform intra-cell offset
        self.dm = np.diff(mets)[0]
        self.da = np.diff(ages)[0]

        # Random number generator (used in sample())
        self.rng = np.random.default_rng()

    def sample(self, N):
        """
        Draw *N* ``(age, [Fe/H])`` pairs from the probability grid.

        Parameters
        ----------
        N : int
            Number of samples to draw.

        Returns
        -------
        ndarray of shape (N, 2)
            Each row is ``[age, feh]``, drawn by selecting a grid cell
            proportional to its weight and adding a uniform intra-cell offset.
        """
        # Select N grid cells according to the normalised probability weights
        sel_idx = self.rng.choice(self.idxs, p=self.pp, size=N)

        # Add a uniform offset within the selected cell for continuous sampling
        sampled_m = self.mm[sel_idx] + self.rng.uniform(0, self.dm, size=N)
        sampled_a = self.aa[sel_idx] + self.rng.uniform(0, self.da, size=N)

        return np.vstack((sampled_a, sampled_m)).T


# ===========================================================================
# IMF GeneralRandom constructors
# ===========================================================================

def set_GR_spl(slope, mass_range):
    """
    Construct a :class:`~generalrandom.GeneralRandom` sampler for a single power-law IMF.

    Implements the Salpeter (1955) IMF:

    .. math::

        \\xi(\\ln m) \\propto m^{\\alpha + 1}

    where ``slope`` = α is typically ``−2.3``.

    Parameters
    ----------
    slope : float
        Power-law exponent α. The log-mass PDF is proportional to
        ``exp(log_m * (slope + 1))``.
    mass_range : tuple of float
        ``(min_mass, max_mass)`` in M☉ over which to define the IMF.

    Returns
    -------
    GR_spl : GeneralRandom
        Sampler for the single power-law IMF in log-mass space.
    """
    l_logm = np.log(mass_range[0])  # lower log-mass limit
    u_logm = np.log(mass_range[1])  # upper log-mass limit

    x = np.linspace(l_logm, u_logm, 1000)
    y = np.exp(x * (slope + 1))

    GR_spl = GeneralRandom(x, y, 1000)
    return GR_spl


def set_GR_bpl(alow, ahigh, bm, mass_range):
    """
    Construct a :class:`~generalrandom.GeneralRandom` sampler for a broken power-law IMF.

    Implements the Kroupa (2001) IMF with two slopes joined at a break mass:

    .. math::

        \\xi(\\ln m) \\propto
        \\begin{cases}
            m^{\\alpha_{\\text{low}} + 1} & m < m_{\\text{break}} \\\\
            m^{\\alpha_{\\text{high}} + 1} & m \\geq m_{\\text{break}}
        \\end{cases}

    The two segments are normalised to be continuous at ``bm``.

    Parameters
    ----------
    alow : float
        Low-mass power-law slope α_low (typically ``−1.3``).
    ahigh : float
        High-mass power-law slope α_high (typically ``−2.3``).
    bm : float
        Break (transition) mass in M☉ (typically ``0.5``).
    mass_range : tuple of float
        ``(min_mass, max_mass)`` in M☉ over which to define the IMF.

    Returns
    -------
    GR_bpl : GeneralRandom
        Sampler for the broken power-law IMF in log-mass space.
    """
    l_logm = np.log(mass_range[0])  # lower log-mass limit
    u_logm = np.log(mass_range[1])  # upper log-mass limit

    x = np.linspace(l_logm, u_logm, 1000)

    # Continuity constant: log-space offset at the break mass
    lkm = np.log(bm) * (alow - ahigh)

    # Piecewise log-PDF: below break uses alow, at/above uses ahigh
    y = np.where(x < np.log(bm), x * (alow + 1), lkm + x * (ahigh + 1))

    GR_bpl = GeneralRandom(x, np.exp(y), 1000)
    return GR_bpl


def set_GR_ln10full(mc, sm, mt, sl, mass_range):
    """
    Construct a :class:`~generalrandom.GeneralRandom` sampler for a lognormal + power-law IMF.

    Implements the Chabrier (2003) system IMF: a lognormal core below the
    transition mass ``mt`` and a power-law tail above it, normalised to be
    continuous at ``mt``.

    .. math::

        \\xi(\\ln m) \\propto
        \\begin{cases}
            \\exp\\!\\left[-\\tfrac{1}{2}
                \\left(\\frac{\\log_{10} m - \\log_{10} m_c}{\\sigma}\\right)^2
            \\right] & m < m_t \\\\
            m^{\\alpha + 1} / k & m \\geq m_t
        \\end{cases}

    where *k* is the continuity constant computed at ``mt``.

    Parameters
    ----------
    mc : float
        Characteristic (peak) mass of the lognormal in M☉ (typically
        ``0.25``).
    sm : float
        Width (sigma) of the lognormal in dex (typically ``0.55``).
    mt : float
        Transition mass in M☉ above which the power-law takes over
        (typically ``1.0``).
    sl : float
        High-mass power-law slope α (typically ``−2.3``).
    mass_range : tuple of float
        ``(min_mass, max_mass)`` in M☉ over which to define the IMF.

    Returns
    -------
    GR_ln10full : GeneralRandom
        Sampler for the lognormal + power-law IMF in log-mass space.
    """
    l_logm = np.log(mass_range[0])  # lower log-mass limit
    u_logm = np.log(mass_range[1])  # upper log-mass limit

    x = np.linspace(l_logm, u_logm, 1000)

    # Boolean mask: True where log-mass is at or above the transition mass
    BMtr = x >= np.log(mt)

    # Continuity constant: ratio of power-law to lognormal value at mt
    lkm = np.exp(np.log(mt) * (sl + 1)) / np.exp(
        -0.5 * ((np.log(mt) / np.log(10) - np.log10(mc)) / sm) ** 2
    )

    y = np.empty_like(x)

    # Lognormal segment (below transition mass)
    y[~BMtr] = np.exp(
        -0.5 * ((x[~BMtr] / np.log(10) - np.log10(mc)) / sm) ** 2
    )

    # Power-law segment (at and above transition mass), scaled for continuity
    y[BMtr] = np.exp(x[BMtr] * (sl + 1)) / lkm

    GR_ln10full = GeneralRandom(x, y, 1000)
    return GR_ln10full


# ===========================================================================
# Ancillary GeneralRandom constructors
# ===========================================================================

def set_GR_dgdm(mu1, deltamu, sigma1, sigma2, amprat):
    """
    Construct a :class:`~generalrandom.GeneralRandom` sampler for a double-Gaussian distance-modulus distribution.

    Useful for modelling line-of-sight depth effects where two stellar
    sub-populations sit at different distances (e.g. two components of a
    dwarf galaxy or a foreground/background pair).

    Parameters
    ----------
    mu1 : float
        Distance modulus of the *nearer* Gaussian component.
    deltamu : float
        Separation between the two Gaussians in distance modulus. If
        positive, ``mu1`` is the nearer component and ``mu1 + deltamu``
        is the farther one.
    sigma1 : float
        Standard deviation of the nearer Gaussian in distance modulus.
    sigma2 : float
        Standard deviation of the farther Gaussian in distance modulus.
    amprat : float
        Amplitude ratio ``A1 / A2`` of the two Gaussians.

    Returns
    -------
    GR_dgdm : GeneralRandom
        Sampler for the double-Gaussian distance-modulus distribution.
    """
    # Derive normalised amplitudes from the amplitude ratio
    a1 = amprat / (1 + amprat)
    a2 = 1 / (1 + amprat)

    # Build the two frozen Gaussian components
    ngauss = stats.norm(loc=mu1, scale=sigma1)               # nearer component
    fgauss = stats.norm(loc=mu1 + deltamu, scale=sigma2)     # farther component

    # Set the evaluation range to 5σ beyond either Gaussian
    gxmin = np.min([mu1 - 5 * sigma1, mu1 + deltamu - 5 * sigma2])
    gxmax = np.max([mu1 + 5 * sigma1, mu1 + deltamu + 5 * sigma2])

    # Evaluate the mixed PDF on a uniform grid
    x = np.linspace(gxmin, gxmax, 1000)
    y = ngauss.pdf(x) * a1 + fgauss.pdf(x) * a2

    GR_dgdm = GeneralRandom(x, y, 1000)
    return GR_dgdm


def set_GR_unif(bf):
    """
    Construct a :class:`~generalrandom.GeneralRandom` sampler for the binary mass-ratio distribution.

    Returns a piecewise-constant distribution that places fraction ``bf``
    of probability mass uniformly on ``[0, 1]`` (binary systems with mass
    ratio ``q ∈ [0, 1]``) and fraction ``1 − bf`` on ``(−∞, 0)`` (encoded
    as the interval ``[−1, 0)`` here) to represent single stars.

    In :func:`~getmags.get_absolute_mags`, a drawn value ``binq < 0``
    signals a single star, while ``binq ≥ 0`` signals a binary with mass
    ratio ``binq``.

    Parameters
    ----------
    bf : float
        Binary fraction; the probability that a randomly drawn system is
        a binary. Must be in ``[0, 1]``.

    Returns
    -------
    GR_unif : GeneralRandom
        Sampler for the binary / single-star indicator distribution.
    """
    # x: knot positions spanning the single-star (< 0) and binary (≥ 0) regions
    # y: piecewise-constant density with weight (1-bf) for singles, bf for binaries
    x = np.array([-1, -1e-6, 0.0, 1])
    y = np.array([1 - bf, 1 - bf, bf, bf])

    GR_unif = GeneralRandom(x, y, 1000)
    return GR_unif