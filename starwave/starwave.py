"""
starwave.py
-----------
StarWave: fitting the stellar birth function of resolved stellar populations
with approximate Bayesian computation.

This module provides the main StarWave class, which fits stellar population
parameters (IMF, SFH, distance modulus, and extinction) to an observed
color-magnitude diagram (CMD) using Sequential Neural Posterior Estimation (SNPE).
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import sys
import functools
import logging

# ---------------------------------------------------------------------------
# Third-party: numerical / scientific
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree, NearestNeighbors

# ---------------------------------------------------------------------------
# Third-party: PyTorch and SBI
# ---------------------------------------------------------------------------
import torch
import sbi
from sbi import utils as utils
from sbi.utils import user_input_checks
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

# ---------------------------------------------------------------------------
# Third-party: extinction
# ---------------------------------------------------------------------------
import extinction

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
sys.path.append(dir_path)

from generalrandom import GeneralRandom
from distributions import *
from plot import *
from parameters import *
from getmags import *
import intNN

# ---------------------------------------------------------------------------
# Joblib / parallelism configuration
# ---------------------------------------------------------------------------
from joblib.externals.loky import set_loky_pickler

set_loky_pickler("dill")


# ===========================================================================
# Main class
# ===========================================================================

class StarWave:
    """
    StarWave: fitting the stellar birth function of resolved stellar populations
    with approximate Bayesian computation.

    This is the main class that performs CMD fitting. It is instantiated with
    an isochrone dataframe, an artificial-star database, and specifications
    for the IMF and SFH parameterizations to fit or sample from.

    The fitting workflow is:
        1. Instantiate ``StarWave`` with isochrone data, artificial-star data,
           and model choices.
        2. Call :meth:`fit_cmd` on an observed CMD array to run Sequential
           Neural Posterior Estimation (SNPE) and retrieve a posterior object.

    Parameters
    ----------
    isodf : pandas.DataFrame
        Multi-indexed DataFrame containing isochrone data for the required
        photometric bands. Must be indexed on ``(age, [Fe/H], mass)``.
    asdf : pandas.DataFrame
        Artificial-star database containing input and output magnitudes for
        artificially injected stars in all required photometric bands.
    bands : list of str
        Names of the photometric bands used. These names must be consistent
        between ``isodf`` and ``asdf``.
    band_lambdas : list of float
        Effective wavelengths (Å) corresponding to each band in ``bands``,
        used to compute band-specific extinction via ``extinction.ccm89``.
    imf_type : {'spl', 'bpl', 'ln'}
        IMF parameterization to fit:
        ``'spl'`` – single power-law,
        ``'bpl'`` – broken power-law,
        ``'ln'``  – lognormal + high-mass power-law.
    sfh_type : {'gaussian', 'grid', 'empirical_mdf'}, optional
        Star-formation history type:
        ``'gaussian'``      – single-burst 2-D Gaussian in (age, [Fe/H]),
        ``'grid'``          – discrete grid-based SFH (requires ``sfh_grid``),
        ``'empirical_mdf'`` – Gaussian/exponential age with empirical MDF
                              (requires ``feh_dis`` and ``age_type``).
        Default is ``'gaussian'``.
    dm_type : {'gaussian', 'dg'}, optional
        Distance-modulus distribution:
        ``'dg'``       – fixed double-Gaussian line-of-sight distance,
        ``'gaussian'`` – Gaussian with mean ``dm`` and spread ``sig_dm``.
        Default is ``'gaussian'``.
    av_type : {'lognormal', 'gaussian'}, optional
        Extinction distribution:
        ``'lognormal'`` – lognormal parameterised by ``av_logn_mu`` and
                          ``av_logn_sigma``,
        ``'gaussian'``  – Gaussian with mean ``av`` and spread ``sig_av``.
        Default is ``'lognormal'``.
    sfh_grid : dict or None, optional
        Required when ``sfh_type='grid'``. Must contain:

        * ``'mets'``          – array of *M* [Fe/H] grid points,
        * ``'ages'``          – array of *A* age (Gyr) grid points,
        * ``'probabilities'`` – *M × A* weight matrix.
    Rv : float, optional
        Total-to-selective extinction ratio. Default is ``3.1``.
    trgb : float, optional
        Tip of the Red Giant Branch magnitude; stars brighter than this
        value are excluded. Default is ``-100`` (no cut applied).
    mass_range : tuple of float or None, optional
        ``(min_mass, max_mass)`` in solar masses. Falls back to the
        isochrone's lower limit – 8 M☉ if ``None`` or invalid.
    color_range : list of tuple or None, optional
        Per-color selection window as a list of ``(min, max)`` tuples,
        one per color index. Defaults to ``(-100, 100)`` for each color.
    age_range : tuple of float or None, optional
        ``(min_age, max_age)`` in Gyr. Falls back to the full isochrone
        age range if ``None`` or invalid.
    feh_range : tuple of float or None, optional
        ``(min_feh, max_feh)``. Falls back to the full isochrone
        metallicity range if ``None`` or invalid.
    feh_dis : array-like of shape (N, 2) or None, optional
        Required when ``sfh_type='empirical_mdf'``. Column 0 gives [Fe/H]
        values; column 1 gives the probability density at each value.
    age_type : {'gaussian', 'exponential'} or None, optional
        Age distribution type used with ``sfh_type='empirical_mdf'``.
    color_corr: scipy.interpolate.BSpline or None, optional
        Empirical color-correction spline to apply to the synthetic CMD. If ``None``, no correction is applied.
    params_kwargs : dict or None, optional
        Additional keyword arguments forwarded to ``make_params`` for
        customising prior parameter definitions.

    Attributes
    ----------
    params : SWParameters
        StarWave parameter object describing priors and fixed values.
    iso_int : intNN.intNN
        Neural-network-based isochrone interpolator.
    kdtree : sklearn.neighbors.KDTree
        KD-tree built on the artificial-star *input* magnitudes, used for
        fast noise injection via nearest-neighbour lookup.
    posteriors : list
        Accumulated SNPE posterior objects, one per inference round.
        Populated by :meth:`fit_cmd`.

    Examples
    --------
    >>> sw = StarWave(isodf, asdf, bands=['F606W', 'F814W'],
    ...               band_lambdas=[5921, 8057], imf_type='bpl')
    >>> posterior = sw.fit_cmd(observed_cmd, n_rounds=5, n_sims=500)
    >>> samples = posterior.sample((1000,), x=sw.obs)
    """

    def __init__(
        self,
        isodf,
        asdf,
        bands,
        band_lambdas,
        imf_type,
        sfh_type="gaussian",
        dm_type="gaussian",
        av_type="lognormal",
        sfh_grid=None,
        Rv=3.1,
        trgb=-100,
        mass_range=None,
        color_range=None,
        age_range=None,
        feh_range=None,
        feh_dis=None,
        age_type=None,
        color_corr=None,
        params_kwargs=None,
    ):
        # ------------------------------------------------------------------ #
        # Input validation                                                     #
        # ------------------------------------------------------------------ #
        if sfh_type == "grid" and sfh_grid is None:
            raise ValueError(
                "Please pass an sfh_grid if you want to use grid-based SFH sampling!"
            )

        if sfh_type == "empirical_mdf" and feh_dis is None:
            raise ValueError(
                "Please pass feh_dis if you want to use empirical SFH sampling!"
            )

        if sfh_type == "empirical_mdf" and age_type not in ["gaussian", "exponential"]:
            raise ValueError(
                "Currently only 'gaussian' and 'exponential' age distributions are "
                "supported for empirical_mdf sampling!"
            )

        # ------------------------------------------------------------------ #
        # Store model-type choices                                            #
        # ------------------------------------------------------------------ #
        self.imf_type = imf_type
        self.sfh_type = sfh_type
        self.dm_type = dm_type
        self.av_type = av_type
        self.params_kwargs = params_kwargs

        # ------------------------------------------------------------------ #
        # Build parameter object and initialise priors                        #
        # ------------------------------------------------------------------ #
        self.params = make_params(imf_type, sfh_type, dm_type, av_type, age_type, self.params_kwargs)
        self.make_prior(self.params)  # populates self.fixed_params and self.param_mapper

        # ------------------------------------------------------------------ #
        # Photometric bands                                                   #
        # ------------------------------------------------------------------ #
        self.bands = bands
        self.bands_in = [band + "_in" for band in bands]
        self.bands_out = [band + "_out" for band in bands]

        # ------------------------------------------------------------------ #
        # Isochrone interpolator                                               #
        # ------------------------------------------------------------------ #
        self.iso_int = intNN.intNN(isodf, self.bands)

        # ------------------------------------------------------------------ #
        # Artificial-star database and noise model                            #
        # ------------------------------------------------------------------ #
        self.asdf = asdf
        self.return_inputmags = False  # if True, return noiseless mags in cmd_sim

        # Pre-compute the noise residuals (output – input) for each AS entry.
        self.asdf_noise = (
            self.asdf[self.bands_out].to_numpy()
            - self.asdf[self.bands_in].to_numpy()
        )

        # KD-tree for fast nearest-neighbour noise injection
        self.kdtree = KDTree(asdf[self.bands_in])

        # ------------------------------------------------------------------ #
        # Extinction law and TRGB cut                                         #
        # ------------------------------------------------------------------ #
        self.Rv = Rv
        self.band_lambdas = band_lambdas
        self.trgb = trgb

        # ------------------------------------------------------------------ #
        # Grid-based / empirical SFH ancillary data                           #
        # ------------------------------------------------------------------ #
        self.sfh_grid = sfh_grid
        self.feh_dis = feh_dis
        self.age_type = age_type

        # ------------------------------------------------------------------ #
        # Color selection window (one interval per color index)               #
        # ------------------------------------------------------------------ #
        n_colors = len(bands) - 1
        if color_range is not None and len(color_range) == n_colors:
            self.color_range = color_range
        else:
            self.color_range = [(-100, 100) for _ in range(n_colors)]

        # ------------------------------------------------------------------ #
        # Empirical color correction spline                                      #
        # ------------------------------------------------------------------ #
        if color_corr is not None:
            self.color_corr = color_corr
            print('applying empirical color correction to synthetic CMDs')
        else:
            self.color_corr = None
            print('no color correction applied to synthetic CMDs')

        # ------------------------------------------------------------------ #
        # Minimum log-mass threshold (ln 0.1 M☉ ≈ –2.3)                      #
        # ------------------------------------------------------------------ #
        self.lim_logmass = np.log(0.1)

        # ------------------------------------------------------------------ #
        # Set parameter ranges from isochrone grid                            #
        # ------------------------------------------------------------------ #
        self.set_param_range(isodf, mass_range, age_range, feh_range)

        # Debug flag (set to True to enable verbose internal logging)
        self.debug = False

        # ------------------------------------------------------------------ #
        # Summary                                                              #
        # ------------------------------------------------------------------ #
        print(
            "Initialised StarWave with %s bands, %s IMF, and default priors"
            % (str(bands), imf_type)
        )
        print("Using Rv = %.1f" % self.Rv)
        self.params.summary()

    # ======================================================================= #
    # Parameter-range helpers                                                  #
    # ======================================================================= #

    def set_param_range(self, isodf, mass_range, age_range, feh_range):
        """
        Set the mass, age, and metallicity ranges from the isochrone grid.

        If any provided range is ``None``, has the wrong length, or falls
        outside the isochrone grid's extent, the full grid range is used
        as a fallback and a warning is printed.

        Parameters
        ----------
        isodf : pandas.DataFrame
            Multi-indexed isochrone DataFrame (indexed on age, [Fe/H], mass).
        mass_range : tuple of float or None
            Desired ``(min_mass, max_mass)`` in M☉. Upper limit is capped at
            8 M☉ regardless of the isochrone grid.
        age_range : tuple of float or None
            Desired ``(min_age, max_age)`` in Gyr.
        feh_range : tuple of float or None
            Desired ``(min_feh, max_feh)``.
        """
        # ---- Retrieve unique grid values ----
        iso_ages = isodf.index.get_level_values("age").unique()
        iso_age_min, iso_age_max = iso_ages.min(), iso_ages.max()

        iso_feh = isodf.index.get_level_values("[Fe/H]").unique()
        iso_feh_min, iso_feh_max = iso_feh.min(), iso_feh.max()

        iso_masses = isodf.index.get_level_values("mass").unique()
        iso_mass_min = iso_masses.min()
        default_upper_mass = 8.0  # upper mass cut for binary systems (M☉)

        # ---- Mass range ----
        mass_valid = (
            mass_range is not None
            and len(mass_range) == 2
            and mass_range[0] >= iso_mass_min
            and mass_range[1] <= default_upper_mass
            and mass_range[0] < mass_range[1]
        )
        if mass_valid:
            self.mass_range = mass_range
            print(
                "Using provided mass range: %.2f – %.2f M☉"
                % (mass_range[0], mass_range[1])
            )
        else:
            self.mass_range = (iso_mass_min, default_upper_mass)
            print(
                "Mass range not provided or invalid; using %.2f – %.2f M☉"
                % (iso_mass_min, default_upper_mass)
            )

        # ---- Age range ----
        age_valid = (
            age_range is not None
            and len(age_range) == 2
            and age_range[0] >= iso_age_min
            and age_range[1] <= iso_age_max
            and age_range[0] < age_range[1]
        )
        if age_valid:
            self.age_range = age_range
            print(
                "Using provided age range: %.2f – %.2f Gyr"
                % (age_range[0], age_range[1])
            )
        else:
            self.age_range = (iso_age_min, iso_age_max)
            print(
                "Age range not provided or invalid; using full isochrone range: %.2f – %.2f Gyr"
                % (iso_age_min, iso_age_max)
            )

        # ---- Metallicity range ----
        feh_valid = (
            feh_range is not None
            and len(feh_range) == 2
            and feh_range[0] >= iso_feh_min
            and feh_range[1] <= iso_feh_max
            and feh_range[0] < feh_range[1]
        )
        if feh_valid:
            self.feh_range = feh_range
            print(
                "Using provided metallicity range: %.2f – %.2f"
                % (feh_range[0], feh_range[1])
            )
        else:
            self.feh_range = (iso_feh_min, iso_feh_max)
            print(
                "Metallicity range not provided or invalid; using full isochrone range: %.2f – %.2f"
                % (iso_feh_min, iso_feh_max)
            )

    # ======================================================================= #
    # Kernel initialisation                                                    #
    # ======================================================================= #

    def init_scaler(self, observed_cmd, gamma=None, n_components=50, best_gamma_kw={}):
        """
        Initialise min-max scaling of the CMD and fit the Nyström kernel mapping.

        The scaler maps all CMD coordinates to [0, 1]. The resulting scaled CMD
        is then used to fit a Nyström approximation of an RBF kernel, which
        provides a fixed-dimensional summary statistic for the inference.

        Parameters
        ----------
        observed_cmd : array-like of shape (N, D)
            Observed CMD magnitudes/colors (unscaled).
        gamma : float or None, optional
            RBF kernel bandwidth parameter ``γ``. If ``None``, ``gamma`` is
            chosen automatically via :meth:`best_gamma`.
        n_components : int, optional
            Number of Nyström components (dimension of the summary statistic).
            Default is ``50``.
        best_gamma_kw : dict, optional
            Keyword arguments forwarded to :meth:`best_gamma` when ``gamma``
            is not provided.

        Returns
        -------
        scaled_observed_cmd : ndarray of shape (N, D)
            The min-max–scaled observed CMD.
        """
        # Fit min-max scaler to the observed CMD
        self.cmd_scaler = MinMaxScaler()
        self.cmd_scaler.fit(observed_cmd)
        scaled_observed_cmd = self.cmd_scaler.transform(observed_cmd)

        # Determine kernel bandwidth automatically if not provided
        if gamma is None:
            print("Finding optimal kernel width...")
            gamma = self.best_gamma(scaled_observed_cmd, **best_gamma_kw)
            print("Setting gamma = %i" % gamma)

        # Fit Nyström kernel approximation and store the transform
        Phi_approx = Nystroem(kernel="rbf", n_components=n_components, gamma=gamma)
        Phi_approx.fit(scaled_observed_cmd)
        self.mapping = Phi_approx.transform

        print("Scaler initialised and mapping defined!")
        return scaled_observed_cmd

    # ======================================================================= #
    # CMD simulation                                                           #
    # ======================================================================= #

    def get_cmd(self, nstars, gr_dict, pdict):
        """
        Sample a synthetic CMD for a given set of stellar population parameters.

        For each of the ``nstars`` stars drawn from the generative distributions,
        the method:

        1. Samples mass, binary-mass-ratio, age/[Fe/H], distance modulus, and Av.
        2. Looks up absolute magnitudes from the isochrone interpolator.
        3. Applies distance modulus and CCM89 extinction.
        4. Injects photometric noise via the nearest-neighbour artificial-star
           lookup.
        5. Applies color-range and NaN filters.

        Parameters
        ----------
        nstars : int
            Number of stars to attempt to generate (Poisson-drawn).
        gr_dict : dict
            Dictionary of ``GeneralRandom`` samplers keyed by
            ``'logM'``, ``'BinQ'``, ``'SFH'``, ``'DM'``, ``'av'``.
        pdict : dict
            Current StarWave parameter dictionary (used indirectly via
            distributions already set in ``gr_dict``).

        Returns
        -------
        input_mags : ndarray of shape (M, D)
            Noiseless apparent magnitudes of accepted stars.
        output_mags : ndarray of shape (K, D)
            Noise-injected apparent magnitudes of accepted stars (K ≤ M).
        sdict : dict or None
            Dictionary of raw samples and quality masks with keys
            ``'masses'``, ``'binqs'``, ``'sfhs'``, ``'dms'``, ``'avs'``,
            ``'exts'``, ``'BM_in_good'``, ``'BM_out_good'``. Returns
            ``None`` when ``input_mags`` is empty.
        """
        # Pre-allocate magnitude and extinction arrays; fill with NaN
        input_mags = np.empty((nstars, len(self.bands)))
        input_mags[:] = np.nan
        exts = np.empty((nstars, len(self.bands)))
        exts[:] = np.nan

        # Draw all stellar properties up-front (vectorised)
        masses = gr_dict["logM"].sample(nstars)
        binqs = gr_dict["BinQ"].sample(nstars)
        sfhs = gr_dict["SFH"].sample(nstars)
        dms = gr_dict["DM"].sample(nstars)
        avs = gr_dict["av"].sample(nstars)

        # Loop over individual stars to compute absolute magnitudes
        for ii in range(nstars):
            mass = masses[ii]
            binq = binqs[ii]
            age, feh = sfhs[ii]
            dm = dms[ii]
            av = avs[ii]

            # Skip stars below the log-mass threshold or with invalid age/feh
            if mass < self.lim_logmass or np.isnan(age) or np.isnan(feh):
                continue

            # Retrieve absolute magnitudes from isochrone interpolator
            input_mag = get_absolute_mags(mass, age, feh, binq, self.iso_int, self.bands)

            # Apply distance modulus
            input_mags[ii, :] = input_mag + dm

            # Compute band-specific extinction using CCM89 law
            exts[ii, :] = np.array(
                [
                    extinction.ccm89(np.array([band_lambda]), av, self.Rv)[0]
                    for band_lambda in self.band_lambdas
                ]
            )

        # ---- Filter: remove stars with NaN magnitudes or above the TRGB ----
        BM_in_good = ~((np.isnan(input_mags) + (input_mags < self.trgb)).any(axis=1))
        input_mags = input_mags[BM_in_good]
        exts = exts[BM_in_good]

        # Return early if no stars survive the initial filter
        if len(input_mags) == 0:
            return input_mags, input_mags, None

        # Apply extinction to apparent magnitudes
        input_mags += exts

        # ---- Noise injection via artificial-star nearest neighbours ----
        idxs = self.kdtree.query(input_mags)[1][:, 0]
        output_mags = input_mags + self.asdf_noise[idxs]

        # ---- Filter: reject stars outside the color selection windows ----
        output_colors = np.zeros((len(output_mags), len(self.bands) - 1))
        color_mask = np.zeros(len(output_mags), dtype=bool)
        for ii in range(len(self.bands) - 1):
            output_colors[:, ii] = output_mags[:, ii + 1] - output_mags[:, 0]
            color_mask += (
                (output_colors[:, ii] < self.color_range[ii][0])
                + (output_colors[:, ii] > self.color_range[ii][1])
            )

        output_good = ~(np.isnan(output_mags).any(axis=1)) & ~color_mask
        output_mags = output_mags[output_good]

        # ---- Reconstruct global boolean mask for bookkeeping ----
        BM_out_good = np.zeros(len(BM_in_good), dtype=bool)
        output_good_true_indexes = np.nonzero(output_good)[0]
        BM_in_good_true_indexes = np.nonzero(BM_in_good)[0]
        BM_out_good[BM_in_good_true_indexes[output_good_true_indexes]] = True

        # Collect diagnostic information for downstream use
        sdict = {
            "masses": masses,
            "binqs": binqs,
            "sfhs": sfhs,
            "dms": dms,
            "avs": avs,
            "exts": exts,
            "BM_in_good": BM_in_good,
            "BM_out_good": BM_out_good,
        }

        return input_mags, output_mags, sdict

    def make_cmd(self, mags, sim=False):
        """
        Convert an array of per-band magnitudes into a CMD representation.

        The reference (bluest) band is kept as an apparent magnitude; all
        other bands are replaced by their color relative to the reference
        band (``band[i] – band[0]``).

        Parameters
        ----------
        mags : ndarray of shape (N, D)
            Per-band apparent magnitudes, where ``D`` is the number of bands.
        sim : bool, default False
            If ``True``, the input is assumed to be a simulated CMD and the
            empirical color correction spline (if provided) is applied to the
            colors.

        Returns
        -------
        cmd : ndarray of shape (N, D)
            Modified in-place: column 0 is the reference magnitude; columns
            1 … D-1 are colors ``mag[i] – mag[0]``.
        """
        cmd = mags
        for ii in range(mags.shape[1] - 1):
            cmd[:, ii + 1] -= cmd[:, 0]
            if self.color_corr is not None and sim == True:
                    cmd[:, ii + 1] += self.color_corr(mags[:, 0])
        return cmd

    # ======================================================================= #
    # Kernel bandwidth heuristic                                               #
    # ======================================================================= #

    def best_gamma(self, cmd, q=0.68, fac=1, NN=650):
        """
        Estimate the optimal RBF kernel bandwidth using a nearest-neighbour heuristic.

        For each point in ``cmd``, the distance to its *NN*-th nearest
        neighbour is computed. The kernel bandwidth is then set so that the
        Gaussian places ~68 % of its mass within the *q*-th quantile of
        those distances (scaled by ``fac``).

        Parameters
        ----------
        cmd : ndarray of shape (N, D)
            Unit-scaled CMD used to calibrate the kernel.
        q : float, optional
            Quantile of nearest-neighbour distances used as the characteristic
            length scale. Default is ``0.68``.
        fac : float, optional
            Multiplicative fudge factor applied to the distance quantile.
            Default is ``1``.
        NN : int, optional
            Number of neighbours considered when computing distances.
            Default is ``650``.

        Returns
        -------
        gamma : float
            Optimal ``γ = 1 / (2 σ²)`` for the RBF kernel.
        """
        # Build a KD-tree and compute the distance to the NN-th neighbour
        nbr = NearestNeighbors(
            n_neighbors=NN, algorithm="kd_tree", metric="minkowski", p=2
        )
        nbr.fit(cmd)
        dst, idx = nbr.kneighbors(cmd, return_distance=True)
        dst = dst[:, -1]  # distance to the NN-th (furthest retained) neighbour

        best_dist = np.quantile(dst, q)
        gamma = 1 / (2 * (fac * best_dist) ** 2)

        return gamma

    # ======================================================================= #
    # Distribution constructors                                                #
    # ======================================================================= #

    def set_sfh_dist(self, pdict, sfh_type):
        """
        Construct and return a sampleable SFH distribution object.

        Parameters
        ----------
        pdict : dict
            Parameter dictionary containing the SFH parameters relevant to
            the chosen ``sfh_type``.
        sfh_type : {'gaussian', 'grid', 'empirical_mdf'}
            Type of SFH to instantiate.

        Returns
        -------
        sfh_dist : SW_SFH or GridSFH or Emp_MDF_Sci_Age
            An object with a ``.sample(n)`` method returning ``(age, feh)``
            pairs clipped to ``self.age_range`` and ``self.feh_range``.

        Raises
        ------
        RuntimeError
            If ``sfh_type='grid'`` but ``self.sfh_grid`` is ``None``.
        """
        if sfh_type == "gaussian":
            # Build 2-D covariance matrix with age-metallicity correlation
            cov = pdict["age_feh_corr"] * pdict["sig_age"] * pdict["sig_feh"]
            covmat = np.array(
                [
                    [pdict["sig_age"] ** 2, cov],
                    [cov, pdict["sig_feh"] ** 2],
                ]
            )

            # Ensure positive-definiteness; project to nearest PD matrix if needed
            if not isPD(covmat):
                covmat = nearestPD(covmat)
                print("Found nearest SFH covariance matrix...")

            means = np.array([pdict["age"], pdict["feh"]])
            return SW_SFH(
                stats.multivariate_normal(mean=means, cov=covmat, allow_singular=True),
                self.age_range,
                self.feh_range,
            )

        elif sfh_type == "grid":
            if self.sfh_grid is None:
                raise RuntimeError("Must pass an sfh_grid to use grid-based sampling!")
            return GridSFH(self.sfh_grid)

        elif sfh_type == "empirical_mdf":
            # Build a GeneralRandom sampler from the empirical [Fe/H] distribution
            feh_gr = GeneralRandom(
                self.feh_dis[:, 0], self.feh_dis[:, 1], len(self.feh_dis[:, 0])
            )

            # Build the age distribution
            if self.age_type == "gaussian":
                age_dist = stats.norm(loc=pdict["age"], scale=pdict["sig_age"])
            elif self.age_type == "exponential":
                scale = 1.0 / pdict["tau"]
                loc = pdict["t0"]
                age_dist = exponential_decay(loc=loc, scale=scale)

            return Emp_MDF_Sci_Age(age_dist, feh_gr, self.age_range, self.feh_range)

    def set_dm_dist(self, pdict, dm_type):
        """
        Construct and return a sampleable distance-modulus distribution.

        Parameters
        ----------
        pdict : dict
            Parameter dictionary. Required keys depend on ``dm_type``:

            * ``'dg'``      – ``mu1``, ``deltamu``, ``sigma1``, ``sigma2``,
              ``amprat``.
            * otherwise     – ``dm``, ``sig_dm``.
        dm_type : str
            If ``'dg'``, returns a double-Gaussian LOS distance distribution
            (constructed via :func:`set_GR_dgdm`). Otherwise returns a
            wrapped ``scipy.stats.norm``.

        Returns
        -------
        dm_dist : SWDist or GeneralRandom
            A sampleable distance-modulus distribution.
        """
        if dm_type == "dg":
            # Double-Gaussian line-of-sight distance distribution
            return set_GR_dgdm(
                pdict["mu1"],
                pdict["deltamu"],
                pdict["sigma1"],
                pdict["sigma2"],
                pdict["amprat"],
            )
        else:
            # Simple Gaussian distance-modulus distribution
            return SWDist(stats.norm(loc=pdict["dm"], scale=pdict["sig_dm"]))

    def set_av_dist(self, pdict, av_type):
        """
        Construct and return a sampleable dust-extinction (Av) distribution.

        Parameters
        ----------
        pdict : dict
            Parameter dictionary. Required keys depend on ``av_type``:

            * ``'lognormal'`` – ``av_logn_mu``, ``av_logn_sigma``.
            * otherwise       – ``av``, ``sig_av``.
        av_type : str
            If ``'lognormal'``, returns a lognormal Av distribution.
            Otherwise returns a wrapped ``scipy.stats.norm``.

        Returns
        -------
        av_dist : SWDist
            A sampleable Av distribution (values ≥ 0 by construction for
            the lognormal case).

        Notes
        -----
        The lognormal is parameterised by its *mean* (``av_logn_mu``) and
        *standard deviation* (``av_logn_sigma``) rather than the underlying
        normal's ``μ`` and ``σ``, and is converted internally.
        """
        if av_type == "lognormal":
            # Convert mean/std parameterisation to scipy's (s, scale) form
            a = 1 + (pdict["av_logn_sigma"] / pdict["av_logn_mu"]) ** 2
            s_logn = np.sqrt(np.log(a))
            scale_logn = pdict["av_logn_mu"] / np.sqrt(a)
            return SWDist(stats.lognorm(s=s_logn, scale=scale_logn))
        else:
            # Simple Gaussian extinction distribution
            return SWDist(stats.norm(loc=pdict["av"], scale=pdict["sig_av"]))

    # ======================================================================= #
    # Prior construction                                                       #
    # ======================================================================= #

    def make_prior(self, parameters):
        """
        Build the prior distribution over all *free* model parameters.

        Fixed parameters are stored in ``self.fixed_params``; free parameters
        are collected into ``self.param_mapper`` (name → index into the
        sampled-parameter vector) and returned as a list of
        ``torch.distributions`` objects.

        Parameters
        ----------
        parameters : SWParameters
            StarWave parameter object (as returned by :func:`make_params`).

        Returns
        -------
        priors : list of torch.distributions.Distribution
            One distribution per free parameter, in the order they appear
            in ``parameters.dict``.

        Raises
        ------
        ValueError
            If a free parameter specifies an unrecognised distribution name,
            or if a ``'norm'`` distribution lacks required ``mean``/``sigma``
            keys in ``dist_kwargs``.
        """
        priors = []
        self.fixed_params = {}
        self.param_mapper = {}
        idx = 0  # index into the free-parameter vector

        for ii, (name, param) in enumerate(parameters.dict.items()):

            if param.fixed:
                # Record fixed value; exclude from sampled vector
                self.fixed_params[name] = param.value
                continue

            lower, upper = param.bounds

            if param.distribution == "uniform":
                distribution = torch.distributions.Uniform(
                    lower * torch.ones(1), upper * torch.ones(1)
                )

            elif param.distribution == "norm":
                try:
                    mean = param.dist_kwargs["mean"]
                    sigma = param.dist_kwargs["sigma"]
                except KeyError:
                    raise ValueError(
                        "Please pass valid distribution arguments ('mean' and 'sigma') "
                        "for parameter '%s'!" % name
                    )
                distribution = torch.distributions.Normal(
                    torch.tensor(mean), torch.tensor(sigma)
                )

            else:
                raise ValueError(
                    "Invalid distribution name '%s' for parameter '%s'."
                    % (param.distribution, name)
                )

            priors.append(distribution)
            self.param_mapper[name] = idx
            idx += 1

        return priors

    # ======================================================================= #
    # CMD sampling wrappers                                                    #
    # ======================================================================= #

    def sample_cmd(self, params, model):
        """
        Sample a full synthetic CMD for a given parameter set.

        Accepts parameters as a PyTorch tensor, a NumPy array/list, or a
        plain dictionary. Reconstructs the full ``pdict`` (merging fixed and
        free parameters), sets up the generative distributions, and calls
        :meth:`get_cmd`.

        Parameters
        ----------
        params : torch.FloatTensor, list, ndarray, or dict
            Model parameters. If a tensor or array, values are mapped to
            names via ``self.param_mapper``. If a dict, values are used
            directly.
        model : {'spl', 'bpl', 'ln'}
            IMF parameterization to use for sampling.

        Returns
        -------
        cmd_in : ndarray
            Noiseless CMD (magnitudes converted to mag + colors).
        cmd_out : ndarray
            Noise-injected CMD.
        sdict : dict or None
            Diagnostic sample dictionary from :meth:`get_cmd`.
        """
        # ---- Normalise input type ----
        is_pdict = False
        if isinstance(params, torch.FloatTensor):
            params = params.detach().cpu().numpy()
        elif isinstance(params, (list, np.ndarray)):
            pass
        elif isinstance(params, dict):
            is_pdict = True

        # Re-initialise priors to refresh fixed-parameter bookkeeping
        self.make_prior(self.params)

        # Build the complete parameter dictionary
        pdict = {}
        for name in self.params.keys():
            if name in self.fixed_params:
                pdict[name] = self.fixed_params[name]
            else:
                if is_pdict:
                    pdict[name] = params[name]
                else:
                    pdict[name] = params[self.param_mapper[name]]

        # ---- Construct IMF sampler ----
        if model == "spl":
            gr_dict = {"logM": set_GR_spl(pdict["slope"], self.mass_range)}
        elif model == "bpl":
            gr_dict = {
                "logM": set_GR_bpl(pdict["alow"], pdict["ahigh"], pdict["bm"], self.mass_range)
            }
        elif model == "ln":
            gr_dict = {
                "logM": set_GR_ln10full(
                    pdict["mean"], pdict["sigma"], pdict["bm"], pdict["slope"], self.mass_range
                )
            }
        else:
            print("Unrecognised model '%s'!" % model)

        # ---- Construct remaining generative distributions ----
        gr_dict["BinQ"] = set_GR_unif(pdict["bf"])
        gr_dict["SFH"] = self.set_sfh_dist(pdict, self.sfh_type)
        gr_dict["DM"] = self.set_dm_dist(pdict, self.dm_type)
        gr_dict["av"] = self.set_av_dist(pdict, self.av_type)

        # Draw number of stars from Poisson distribution
        intensity = 10 ** pdict["log_int"]
        nstars = int(stats.poisson.rvs(intensity))

        # Sample the CMD
        mags_in, mags_out, sdict = self.get_cmd(nstars, gr_dict, pdict)
        cmd_in = self.make_cmd(mags_in, sim=True)
        cmd_out = self.make_cmd(mags_out, sim=True)

        return cmd_in, cmd_out, sdict

    def sample_norm_cmd(self, params, model):
        """
        Sample a unit-normalised synthetic CMD.

        A thin wrapper around :meth:`sample_cmd` that applies the min-max
        scaler fitted in :meth:`init_scaler`. Falls back to ``self.dummy_cmd``
        if the sampled CMD is empty.

        Parameters
        ----------
        params : torch.FloatTensor, list, ndarray, or dict
            Model parameters (forwarded to :meth:`sample_cmd`).
        model : {'spl', 'bpl', 'ln'}
            IMF parameterization.

        Returns
        -------
        in_cmd_scaled : ndarray
            Unit-scaled noiseless CMD.
        out_cmd_scaled : ndarray
            Unit-scaled noise-injected CMD.
        """
        in_cmd, out_cmd, sdict = self.sample_cmd(params, model)

        if len(in_cmd) == 0 or len(out_cmd) == 0:
            print("Empty CMD!")
            return self.dummy_cmd, self.dummy_cmd

        return self.cmd_scaler.transform(in_cmd), self.cmd_scaler.transform(out_cmd)

    # ======================================================================= #
    # Kernel representation                                                    #
    # ======================================================================= #

    def kernel_representation(self, P, mapping):
        """
        Project a CMD onto the kernel feature space.

        Each row of ``P`` is mapped through ``mapping`` and the resulting
        feature vectors are summed to produce a single fixed-length summary
        statistic.

        Parameters
        ----------
        P : ndarray of shape (N, D)
            Unit-scaled CMD to project.
        mapping : callable
            Nyström transform (``Phi_approx.transform``) fitted in
            :meth:`init_scaler`.

        Returns
        -------
        Phi_P : ndarray of shape (n_components,)
            Sum of per-star kernel features, used as the summary statistic.
        """
        Phi_P = mapping(P).sum(axis=0)
        return Phi_P

    def cmd_sim(self, params, imf_type):
        """
        Simulate a kernel-represented CMD given a parameter vector.

        This is the function registered with SBI as the *simulator*. It
        chains :meth:`sample_norm_cmd` and :meth:`kernel_representation`
        and optionally returns the noiseless representation when
        ``self.return_inputmags`` is ``True``.

        Parameters
        ----------
        params : torch.FloatTensor or ndarray
            Model parameters.
        imf_type : {'spl', 'bpl', 'ln'}
            IMF parameterization.

        Returns
        -------
        representation : ndarray of shape (n_components,)
            Kernel-space summary statistic of the simulated CMD.
        """
        in_cmd, out_cmd = self.sample_norm_cmd(params, model=imf_type)

        if self.return_inputmags:
            return self.kernel_representation(in_cmd, self.mapping)
        else:
            return self.kernel_representation(out_cmd, self.mapping)

    # ======================================================================= #
    # Main fitting method                                                      #
    # ======================================================================= #

    def fit_cmd(
        self,
        observed_cmd,
        n_rounds=5,
        n_sims=100,
        savename="starwave",
        min_acceptance_rate=0.0001,
        gamma=None,
        n_components=50,
        cores=1,
        alpha=0.5,
        statistic="output",
        best_gamma_kw={},
        train_kw={},
    ):
        """
        Fit an observed CMD using Sequential Neural Posterior Estimation (SNPE).

        This is the primary entry point for inference. The method:

        1. Scales the observed CMD and fits the Nyström kernel summary statistic.
        2. Computes the kernel representation of the observed data (``self.obs``).
        3. Iterates SNPE rounds, each time simulating ``n_sims`` synthetic CMDs
           from the current proposal, training a neural density estimator, and
           updating the proposal to the refined posterior.

        Parameters
        ----------
        observed_cmd : array-like of shape (N, D)
            The observed CMD (D = number of bands).
        n_rounds : int, optional
            Number of SNPE rounds. More rounds allow the proposal to concentrate
            around the posterior at the cost of additional simulations.
            Default is ``5``.
        n_sims : int, optional
            Number of simulations per round. Default is ``100``.
        savename : str, optional
            Base name for saving outputs (not yet implemented). Default is
            ``'starwave'``.
        min_acceptance_rate : float, optional
            Minimum acceptance rate threshold (reserved for future use).
            Default is ``0.0001``.
        gamma : float or None, optional
            RBF kernel bandwidth. If ``None``, determined automatically.
            Default is ``None``.
        n_components : int, optional
            Number of Nyström components in the summary statistic. Default
            is ``50``.
        cores : int, optional
            Number of CPU cores for parallel simulation. Multi-core support
            is not yet implemented. Default is ``1``.
        alpha : float, optional
            Weighting parameter for CMD kernel distance (reserved for future
            use). Default is ``0.5``.
        statistic : str, optional
            Summary-statistic type (reserved for future use). Default is
            ``'output'``.
        best_gamma_kw : dict, optional
            Keyword arguments forwarded to :meth:`best_gamma`.
        train_kw : dict, optional
            Keyword arguments forwarded to ``SNPE.train()``.

        Returns
        -------
        posterior : sbi.inference.posteriors.DirectPosterior
            The SNPE posterior from the final round, conditioned on the
            observed summary statistic ``self.obs``. Draw samples with
            ``posterior.sample((n,), x=self.obs)``.
        """
        if cores == 1:
            pass  # Multi-core support is planned for a future release

        # ---- Initialise CMD scaler and kernel mapping ----
        scaled_observed_cmd = self.init_scaler(
            observed_cmd,
            gamma=gamma,
            n_components=n_components,
            best_gamma_kw=best_gamma_kw,
        )

        # Compute and store the kernel representation of the observed CMD
        obs = torch.tensor(self.kernel_representation(scaled_observed_cmd, self.mapping))
        self.obs = obs

        # Placeholder CMD used when a simulation returns an empty result
        self.dummy_cmd = np.zeros(observed_cmd.shape)

        # Curry the simulator with the chosen IMF type so it matches SBI's API
        def simcmd(imf_type):
            return lambda params: self.cmd_sim(params, imf_type=imf_type)

        Nobs = len(scaled_observed_cmd)  # noqa: F841 – available for log_int tuning

        # Print a summary of all priors before fitting
        print_prior_summary(self.params)

        # ---- Build SBI prior and prepare simulator ----
        prior = user_input_checks.MultipleIndependent(self.make_prior(self.params))
        simulator = simcmd(self.imf_type)
        self.simulator, self.prior = prepare_for_sbi(simulator, prior)

        # ---- Sequential inference rounds ----
        inference = SNPE(prior=self.prior)
        self.posteriors = []
        proposal = self.prior

        for round_idx in range(n_rounds):
            print("Starting round %i of neural inference..." % (round_idx + 1))

            # Simulate from current proposal and train density estimator
            theta, x = simulate_for_sbi(
                self.simulator, proposal, num_simulations=n_sims, num_workers=cores
            )
            density_estimator = inference.append_simulations(
                theta, x, proposal=proposal
            ).train(**train_kw)

            # Build posterior and set as next round's proposal
            posterior = inference.build_posterior(density_estimator)
            self.posteriors.append(posterior)
            proposal = posterior.set_default_x(obs)

        return self.posteriors[-1]


# ===========================================================================
# Module-level entry point (for quick sanity checks)
# ===========================================================================

if __name__ == "__main__":
    sw = StarWave()
    sw.params.pretty_print()