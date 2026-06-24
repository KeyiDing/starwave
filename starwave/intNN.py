"""
intNN.py
--------
Isochrone interpolator for resolved stellar population fitting.

Author: Mario Gennaro

Defines the :class:`intNN` class, which interpolates isochrone magnitudes
in mass, age, and metallicity using pre-built per-(age, metallicity) linear
interpolants. At call time, the nearest grid point in (age, [Fe/H]) is
located and the interpolant is evaluated at the requested stellar mass.
"""

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------
import findNN_arr


# ===========================================================================
# Isochrone interpolator
# ===========================================================================

class intNN:
    """
    Nearest-neighbour isochrone interpolator in age and metallicity.

    On initialisation, a 1-D linear mass interpolant is pre-built for every
    ``(age, [Fe/H])`` grid node and every requested photometric band.  At
    call time the nearest grid node is located with :mod:`findNN_arr` and the
    corresponding interpolant is evaluated at the requested stellar mass.

    Parameters
    ----------
    isoPD : pandas.DataFrame
        Multi-indexed isochrone DataFrame with index levels
        ``('[Fe/H]', 'age', 'mass')`` and one column per photometric band.
    photbands : list of str
        Names of the photometric bands to interpolate (must match column
        names in ``isoPD``).

    Attributes
    ----------
    isoages : ndarray of shape (A,)
        Sorted unique age grid points (Gyr) extracted from ``isoPD``.
    isomets : ndarray of shape (Z,)
        Sorted unique metallicity grid points ([Fe/H]) extracted from
        ``isoPD``.
    iso_intp : dict
        Nested structure ``iso_intp[band][age_idx][met_idx]`` holding a
        ``scipy.interpolate.interp1d`` object for each
        ``(band, age, metallicity)`` combination.
    iso_mrng : list of list of tuple
        ``iso_mrng[age_idx][met_idx]`` gives the ``(min_mass, max_mass)``
        range covered by the isochrone at that grid node.
    photbands : list of str
        Copy of the requested photometric band names.
    l_age : float
        Lower bound of the age grid (Gyr).
    u_age : float
        Upper bound of the age grid (Gyr).

    Examples
    --------
    >>> interp = intNN(isodf, photbands=['F606W', 'F814W'])
    >>> mags = interp(mass=0.8, age=10.0, met=-1.5)
    >>> mags['F606W']   # scalar apparent magnitude
    """

    def __init__(self, isoPD, photbands):
        # ------------------------------------------------------------------ #
        # Extract unique age and metallicity grid points                      #
        # ------------------------------------------------------------------ #
        self.isoages = np.unique(
            np.asarray([isoPD.index.get_level_values("age")]).T
        )
        self.l_age = np.min(self.isoages)  # lower age bound for range checking
        self.u_age = np.max(self.isoages)  # upper age bound for range checking

        self.isomets = np.unique(
            np.asarray([isoPD.index.get_level_values("[Fe/H]")]).T
        )

        self.photbands = photbands

        # ------------------------------------------------------------------ #
        # Pre-allocate interpolant and mass-range storage                     #
        # ------------------------------------------------------------------ #
        # iso_intp[band][age_idx][met_idx] → interp1d object
        self.iso_intp = {}
        for band in photbands:
            self.iso_intp[band] = [
                [0 for _ in range(len(self.isomets))]
                for _ in range(len(self.isoages))
            ]

        # iso_mrng[age_idx][met_idx] → (min_mass, max_mass) tuple
        self.iso_mrng = [
            [0 for _ in range(len(self.isomets))]
            for _ in range(len(self.isoages))
        ]

        # Flag to suppress repeated out-of-range warnings (reserved for future use)
        self.has_warned = False

        # ------------------------------------------------------------------ #
        # Build one linear interpolant per (age, metallicity, band) node     #
        # ------------------------------------------------------------------ #
        print(
            "Interpolating %i ages and %i metallicities..."
            % (len(self.isoages), len(self.isomets))
        )

        for aa, age in tqdm(enumerate(self.isoages)):
            for zz, met in enumerate(self.isomets):

                # Extract the mass grid and magnitude columns for this node
                isomass = np.asarray(
                    isoPD.loc[met].loc[age].index.get_level_values("mass")
                )
                self.iso_mrng[aa][zz] = (np.amin(isomass), np.amax(isomass))

                isomags = isoPD.loc[met].loc[age][self.photbands]

                # Build a linear interp1d for each band; extrapolation returns NaN
                for band in self.photbands:
                    self.iso_intp[band][aa][zz] = interp1d(
                        isomass,
                        isomags[band],
                        kind="linear",
                        assume_sorted=True,
                        bounds_error=False,
                        fill_value=np.nan,
                    )

    def __call__(self, mss, age, met):
        """
        Interpolate isochrone magnitudes at a given mass, age, and metallicity.

        The nearest grid node in ``(age, [Fe/H])`` is selected and the
        corresponding mass interpolant is evaluated at ``mss``. Stars outside
        the valid age range return ``NaN`` for all bands.

        Parameters
        ----------
        mss : float
            Stellar mass (M☉) at which to evaluate the isochrone.
        age : float
            Stellar age (Gyr). If outside ``[l_age, u_age]``, all magnitudes
            are returned as ``NaN``.
        met : float
            Stellar metallicity [Fe/H]. The nearest grid point is used.

        Returns
        -------
        isointp : dict
            ``{band: magnitude}`` mapping. Values are scalar floats, or
            ``NaN`` when the requested point lies outside the interpolation
            domain.
        """
        # Return NaN for all bands if age falls outside the isochrone grid
        if age < self.l_age or age > self.u_age:
            isointp = {band: np.nan for band in self.photbands}

        else:
            # Locate the nearest grid indices in age and metallicity
            nage_idx = findNN_arr.find_nearest_idx(self.isoages, age)
            nmet_idx = findNN_arr.find_nearest_idx(self.isomets, met)

            # Evaluate each band's interpolant at the requested mass
            isointp = {
                band: self.iso_intp[band][nage_idx][nmet_idx](mss)
                for band in self.photbands
            }

        return isointp