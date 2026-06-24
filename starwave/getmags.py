"""
getmags.py
----------
Magnitude utilities for StarWave's synthetic CMD generation.

Provides three functions that together convert isochrone-interpolated
absolute magnitudes into noise-injected observable magnitudes:

* :func:`get_absolute_mags`  – isochrone lookup for single or binary stars.
* :func:`get_observable_mags` – apply distance modulus and extinction.
* :func:`get_noisy_mags`     – inject photometric noise via an artificial-star
                               KD-tree lookup.
"""

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np


# ===========================================================================
# Absolute magnitude computation
# ===========================================================================

def get_absolute_mags(logsysmass, age, met, binq, iso_int, bands):
    """
    Compute absolute magnitudes for a single or binary stellar system.

    Interpolates the isochrone grid at the given age and metallicity to
    retrieve per-band absolute magnitudes. For binary systems the flux
    contributions of both components are summed in linear flux space before
    converting back to magnitudes.

    Parameters
    ----------
    logsysmass : float
        Natural logarithm of the total system mass (M☉). The function
        exponentiates this internally so that the IMF sampler can work in
        log-mass space.
    age : float
        Stellar age (Gyr), passed directly to the isochrone interpolator.
    met : float
        Stellar metallicity [Fe/H], passed directly to the isochrone
        interpolator.
    binq : float
        Binary mass ratio ``q = M2 / M1``. A value ``binq >= 0`` indicates
        a binary system; ``binq < 0`` indicates a single star.
    iso_int : intNN
        Isochrone interpolator instance (see ``intNN.py``) with signature
        ``iso_int(mass, age, met) → {band: magnitude}``.
    bands : list of str
        Photometric band names to retrieve (must match the interpolator's
        band list).

    Returns
    -------
    absmags : ndarray of shape (len(bands),)
        Absolute magnitudes in each requested band. Individual entries are
        ``NaN`` where the isochrone interpolation returned no valid value
        (e.g. the mass falls outside the grid).

    Notes
    -----
    For a binary with total mass *M* and mass ratio *q*, the component
    masses are:

    .. math::

        M_1 = \\frac{M}{1 + q}, \\quad M_2 = q \\cdot M_1

    Flux summation is performed as:

    .. math::

        F_{\\text{tot}} = F_1 + F_2, \\quad
        m_{\\text{tot}} = -2.5 \\log_{10}(F_{\\text{tot}})

    where ``NaN`` components are excluded from the sum.
    """
    # Convert log-mass back to linear mass (M☉)
    sysmass = np.exp(logsysmass)
    nb = len(bands)

    if binq >= 0:
        # ---- Binary system ----
        # Decompose total mass into the two component masses:
        #   M1 = sysmass / (1 + q),  M2 = q * M1
        mass_2 = sysmass / (1.0 + binq) * np.array([1.0, binq], dtype=float)

        # Retrieve per-component magnitudes from the isochrone interpolator
        sysmags = np.nan * np.ones([2, nb])
        for j, mcomp in enumerate(mass_2):
            row = iso_int(mcomp, age, met)
            for i in range(nb):
                sysmags[j, i] = float(row[bands[i]])

        # Convert to linear flux (arbitrary zero-point cancels in the sum)
        sysmags = 10 ** (-0.4 * sysmags)

        # Sum fluxes per band (ignoring NaN components) and convert back to magnitudes
        absmags = np.nan * np.ones(nb)
        for j in range(nb):
            oneband = sysmags[:, j]
            msk = ~np.isnan(oneband)
            if ~np.any(msk):
                # Both components are invalid for this band
                absmags[j] = np.nan
            else:
                absmags[j] = -2.5 * np.log10(np.sum(oneband[msk]))

    else:
        # ---- Single star ----
        absmags = np.empty(nb)
        row = iso_int(sysmass, age, met)
        for i in range(nb):
            absmags[i] = float(row[bands[i]])

    return absmags


# ===========================================================================
# Apparent magnitude computation
# ===========================================================================

def get_observable_mags(absmags, DM=0.0, ext=[0.0, 0.0]):
    """
    Convert absolute magnitudes to apparent (observable) magnitudes.

    Applies the distance modulus and band-specific extinction:

    .. math::

        m_{\\text{obs}} = M_{\\text{abs}} + \\text{DM} + A_{\\lambda}

    Parameters
    ----------
    absmags : ndarray of shape (N,)
        Absolute magnitudes in each photometric band.
    DM : float, optional
        Distance modulus ``μ = 5 log10(d/10 pc)``. Default is ``0.0``.
    ext : array-like of shape (N,), optional
        Band-specific extinction values ``A_λ`` (magnitudes). Default is
        ``[0.0, 0.0]``.

    Returns
    -------
    ndarray of shape (N,)
        Apparent magnitudes after applying distance modulus and extinction.
    """
    return absmags + ext + DM


# ===========================================================================
# Photometric noise injection
# ===========================================================================

def get_noisy_mags(
    obsmags,
    ASKDtree,
    AS_mag1_in,
    AS_mag2_in,
    AS_mag1_out,
    AS_mag2_out,
    AS_det,
):
    """
    Inject photometric noise using an artificial-star KD-tree lookup.

    Finds the nearest artificial star (AS) to ``obsmags`` in input-magnitude
    space, then applies the corresponding input-to-output magnitude offset as
    a noise realisation. Stars that were not recovered in the AS experiment
    are returned as ``NaN``.

    Parameters
    ----------
    obsmags : ndarray of shape (2,)
        Noiseless apparent magnitudes in two bands.
    ASKDtree : sklearn.neighbors.KDTree
        KD-tree built on the artificial-star *input* magnitudes, used for
        fast nearest-neighbour lookup.
    AS_mag1_in : ndarray of shape (M,)
        Input magnitudes of artificial stars in band 1.
    AS_mag2_in : ndarray of shape (M,)
        Input magnitudes of artificial stars in band 2.
    AS_mag1_out : ndarray of shape (M,)
        Recovered output magnitudes of artificial stars in band 1.
    AS_mag2_out : ndarray of shape (M,)
        Recovered output magnitudes of artificial stars in band 2.
    AS_det : ndarray of bool, shape (M,)
        Detection flag for each artificial star (``True`` if recovered).

    Returns
    -------
    ndarray of shape (2,)
        Noise-injected apparent magnitudes ``[m1_noisy, m2_noisy]``, or
        ``[NaN, NaN]`` if the nearest artificial star was not detected.

    Notes
    -----
    The noise offset is computed as:

    .. math::

        m_{k,\\text{noisy}} = m_{k,\\text{obs}}
            + (m_{k,\\text{out}} - m_{k,\\text{in}})_{\\text{NN}}

    where ``NN`` denotes the nearest artificial-star neighbour.
    """
    # Find the index of the nearest artificial star in input-magnitude space
    ind = ASKDtree.query(obsmags.reshape(1, -1), return_distance=False)[0]

    if AS_det[ind] == True:
        # Apply the AS input→output offset as the noise realisation
        m1 = AS_mag1_out[ind] - AS_mag1_in[ind] + obsmags[0]
        m2 = AS_mag2_out[ind] - AS_mag2_in[ind] + obsmags[1]
        return np.array([m1, m2]).flatten()
    else:
        # Star not recovered in the AS experiment — treat as undetected
        return np.nan * np.ones(2)