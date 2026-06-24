"""
parameters.py
-------------
Parameter and prior infrastructure for StarWave.

This module defines:

* :class:`SWDist`        – thin wrapper around a SciPy distribution that
                           exposes ``.sample()`` and ``.log_prob()``.
* :class:`SWParameter`   – ``OrderedDict``-based container for a single
                           model parameter (value, bounds, prior, fixed flag).
* :class:`SWParameters`  – ``OrderedDict``-based collection of
                           :class:`SWParameter` objects with convenience
                           methods for prior construction and printing.
* :func:`make_params`    – factory that assembles the full
                           :class:`SWParameters` object for a given
                           combination of IMF, SFH, DM, and Av types.
* :func:`print_prior_summary` – human-readable prior summary to stdout
                                and/or a log file.
* :class:`MultipleIndependent` – joint PyTorch distribution built from an
                                  ordered sequence of independent marginals.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import copy
import logging
from collections import OrderedDict
from typing import Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import torch
from torch import Tensor, float32
from torch.distributions import Distribution
import sbi
from scipy import stats


# ===========================================================================
# SciPy distribution wrapper
# ===========================================================================

class SWDist:
    """
    Thin wrapper around a SciPy continuous distribution.

    Exposes the ``.sample()`` and ``.log_prob()`` interface expected by
    StarWave's generative sampling code, delegating to the underlying
    SciPy object's ``rvs`` and ``logpdf`` methods.

    Parameters
    ----------
    distribution : scipy.stats continuous_frozen
        Any frozen SciPy distribution (e.g. ``scipy.stats.norm(0, 1)``).

    Examples
    --------
    >>> from scipy import stats
    >>> d = SWDist(stats.norm(loc=18.5, scale=0.1))
    >>> samples = d.sample(100)      # returns ndarray of shape (100,)
    >>> lp = d.log_prob(18.5)        # scalar log-probability
    """

    def __init__(self, distribution):
        self.dist = distribution  # frozen SciPy distribution object

    def sample(self, N):
        """
        Draw *N* independent samples.

        Parameters
        ----------
        N : int
            Number of samples to draw.

        Returns
        -------
        ndarray of shape (N,)
            Random variates from the wrapped distribution.
        """
        return self.dist.rvs(N)

    def log_prob(self, x):
        """
        Evaluate the log probability density at *x*.

        Parameters
        ----------
        x : float or array-like
            Point(s) at which to evaluate the log-PDF.

        Returns
        -------
        float or ndarray
            Log probability density value(s).
        """
        return self.dist.logpdf(x)


# ===========================================================================
# Single-parameter container
# ===========================================================================

class SWParameter(OrderedDict):
    """
    ``OrderedDict``-based container for a single StarWave model parameter.

    Stores the parameter's current value, prior bounds, prior distribution
    type, and whether the parameter is held fixed during inference.

    Parameters
    ----------
    name : str
        Parameter name (e.g. ``'slope'``, ``'dm'``).
    value : float
        Initial (or fixed) value of the parameter.
    bounds : array-like of length 2
        ``[lower, upper]`` bounds. Used directly when
        ``distribution='uniform'``; treated as soft constraints otherwise.
    distribution : str, optional
        Prior distribution type. Currently supported:
        ``'uniform'`` (default) and ``'norm'``.
    dist_kwargs : dict or None, optional
        Extra keyword arguments for non-uniform priors. Required keys for
        ``'norm'``: ``'mean'`` and ``'sigma'``.
    fixed : bool, optional
        If ``True``, the parameter is excluded from inference and held at
        ``value``. Default is ``False``.

    Methods
    -------
    set(**attributes) :
        Update one or more attributes in-place. Any attribute not explicitly
        passed is left unchanged.

    Examples
    --------
    >>> p = SWParameter('slope', -2.3, [-4, -1])
    >>> p.set(value=-2.0, fixed=False)
    """

    def __init__(
        self,
        name,
        value,
        bounds,
        distribution="uniform",
        dist_kwargs=None,
        fixed=False,
    ):
        self.name = name
        self.value = value
        self.bounds = bounds
        self.distribution = distribution
        self.dist_kwargs = dist_kwargs
        self.fixed = fixed

        # Keep an internal dict representation for OrderedDict compatibility
        self.param_dict = dict(
            name=self.name,
            value=self.value,
            bounds=self.bounds,
            distribution=self.distribution,
            dist_kwargs=self.dist_kwargs,
            fixed=self.fixed,
        )
        super().__init__(self.param_dict)

    def set(
        self,
        value=None,
        bounds=None,
        distribution=None,
        dist_kwargs=None,
        fixed=None,
    ):
        """
        Update one or more parameter attributes in-place.

        Only the attributes explicitly passed (non-``None``) are modified;
        all others retain their current values. The internal ``param_dict``
        and the ``OrderedDict`` representation are refreshed automatically.

        Parameters
        ----------
        value : float or None, optional
            New parameter value.
        bounds : array-like of length 2 or None, optional
            New ``[lower, upper]`` bounds.
        distribution : str or None, optional
            New prior distribution name.
        dist_kwargs : dict or None, optional
            New distribution keyword arguments.
        fixed : bool or None, optional
            New fixed flag.
        """
        if value is not None:
            self.value = value
        if bounds is not None:
            self.bounds = bounds
        if distribution is not None:
            self.distribution = distribution
        if dist_kwargs is not None:
            self.dist_kwargs = dist_kwargs
        if fixed is not None:
            self.fixed = fixed

        # Refresh the internal dict and the OrderedDict base class
        self.param_dict = dict(
            name=self.name,
            value=self.value,
            bounds=self.bounds,
            distribution=self.distribution,
            dist_kwargs=self.dist_kwargs,
            fixed=self.fixed,
        )
        super().__init__(self.param_dict)


# ===========================================================================
# Parameter collection
# ===========================================================================

class SWParameters(OrderedDict):
    """
    ``OrderedDict``-based collection of :class:`SWParameter` objects.

    Acts as the central parameter registry for a StarWave model run.
    Provides convenience accessors, a human-readable summary, and
    converters to external inference frameworks.

    Parameters
    ----------
    params_dict : dict
        Ordered mapping of parameter name → :class:`SWParameter`.
    kwargs : dict or None, optional
        Optional printing/saving options:

        * ``'filename'`` *(str or None)* – if set, the prior summary is
          written to this log file.
        * ``'verbose'`` *(bool)* – if ``True``, the summary is also
          printed to stdout.

    Methods
    -------
    get_values() :
        Return a plain ``{name: value}`` dictionary for all parameters.
    summary() :
        Print (and optionally save) the prior summary via
        :func:`print_prior_summary`.
    to_torch() :
        Convert free parameters to a list of PyTorch prior distributions.
    to_pyabc() :
        Convert free parameters to a ``pyabc`` prior ``Distribution``.
    """

    def __init__(self, params_dict, kwargs=None):
        super().__init__(params_dict)
        self.dict = params_dict
        self.kwargs = kwargs

    def __getitem__(self, param):
        return self.dict[param]

    def get_values(self):
        """
        Return a plain value dictionary for all parameters.

        Returns
        -------
        dict
            ``{name: value}`` for every parameter in the collection.
        """
        values_dict = {}
        for name, param in self.dict.items():
            values_dict[name] = param.value
        return values_dict

    def summary(self):
        """
        Print (and optionally log) a human-readable prior summary.

        Delegates to :func:`print_prior_summary` using any ``'filename'``
        and ``'verbose'`` keys found in ``self.kwargs``.
        """
        if self.kwargs is not None:
            print_prior_summary(
                self.dict,
                filename=self.kwargs.get("filename"),
                verbose=self.kwargs.get("verbose"),
            )

    def to_torch(self):
        """
        Convert free parameters to a list of PyTorch prior distributions.

        Returns
        -------
        list of torch.distributions.Distribution
            One distribution per *free* (non-fixed) parameter.
        """
        return make_prior(self)

    def to_pyabc(self):
        """
        Convert free parameters to a ``pyabc`` prior ``Distribution``.

        Returns
        -------
        pyabc.Distribution
            Joint prior distribution compatible with the pyABC library.
        """
        return make_prior_pyabc(self)


# ===========================================================================
# Parameter factory
# ===========================================================================

def make_params(imf_type, sfh_type, dm_type, av_type, age_type=None, kwargs=None):
    """
    Assemble a :class:`SWParameters` object for a given model configuration.

    Constructs all relevant :class:`SWParameter` entries for the chosen
    IMF, SFH, distance-modulus, and extinction model types. Default values
    and bounds reflect typical resolved stellar population fitting ranges.
    All physical parameters (DM, Av) are initialised as *fixed* — the user
    should call ``.set(fixed=False)`` on any parameter they wish to sample.

    Parameters
    ----------
    imf_type : {'spl', 'bpl', 'ln'}
        IMF parameterization:

        * ``'spl'`` – single power-law (free parameter: ``slope``).
        * ``'bpl'`` – broken power-law (free: ``alow``, ``ahigh``, ``bm``).
        * ``'ln'``  – lognormal + high-mass power-law (free: ``mean``,
          ``sigma``, ``bm``, ``slope``).
    sfh_type : {'gaussian', 'grid', 'empirical_mdf'}
        Star-formation history type:

        * ``'gaussian'``      – 2-D Gaussian in (age, [Fe/H]); adds
          ``age``, ``sig_age``, ``feh``, ``sig_feh``, ``age_feh_corr``.
        * ``'grid'``          – no additional SFH parameters.
        * ``'empirical_mdf'`` – empirical [Fe/H] + Gaussian or exponential
          age (see ``age_type``).
    dm_type : {'gaussian', 'dg'}
        Distance-modulus distribution:

        * ``'dg'``      – double-Gaussian; adds ``mu1``, ``deltamu``,
          ``sigma1``, ``sigma2``, ``amprat`` (all fixed by default).
        * ``'gaussian'``– single Gaussian; adds ``dm``, ``sig_dm`` (fixed).
    av_type : {'lognormal', 'gaussian'}
        Extinction distribution:

        * ``'lognormal'`` – adds ``av_logn_mu``, ``av_logn_sigma`` (fixed).
        * ``'gaussian'``  – adds ``av``, ``sig_av`` (fixed).
    age_type : {'gaussian', 'exponential'} or None, optional
        Age distribution for ``sfh_type='empirical_mdf'``:

        * ``'gaussian'``    – adds ``age``, ``sig_age``.
        * ``'exponential'`` – adds ``t0``, ``tau`` (both fixed by default).
    kwargs : dict or None, optional
        Passed directly to :class:`SWParameters` for printing/saving
        control (see :class:`SWParameters` for supported keys).

    Returns
    -------
    SWParameters
        Fully populated parameter collection ready for prior construction
        or summary printing.
    """
    parameters = {}

    # ------------------------------------------------------------------ #
    # Universal parameters                                                 #
    # ------------------------------------------------------------------ #
    parameters["log_int"] = SWParameter("log_int", 2, [2, 6])
    parameters["bf"] = SWParameter("bf", 0.2, [0, 1])

    # ------------------------------------------------------------------ #
    # Distance-modulus parameters                                          #
    # ------------------------------------------------------------------ #
    if dm_type == "dg":
        # Double-Gaussian line-of-sight distance distribution
        parameters["mu1"] = SWParameter("mu1", 18.96, [18.8, 19.2], fixed=True)
        parameters["deltamu"] = SWParameter("deltamu", 0.2, [0.01, 0.4], fixed=True)
        parameters["sigma1"] = SWParameter("sigma1", 0.3, [0.01, 1.0], fixed=True)
        parameters["sigma2"] = SWParameter("sigma2", 0.1, [0.01, 1.0], fixed=True)
        parameters["amprat"] = SWParameter("amprat", 0.5, [0.01, 100], fixed=True)
    else:
        # Single Gaussian distance modulus
        parameters["dm"] = SWParameter("dm", 0, [0, 1], fixed=True)
        parameters["sig_dm"] = SWParameter("sig_dm", 0.1, [0, 0.5], fixed=True)

    # ------------------------------------------------------------------ #
    # Extinction parameters                                                #
    # ------------------------------------------------------------------ #
    if av_type == "lognormal":
        parameters["av_logn_mu"] = SWParameter(
            "av_logn_mu", 0.1, [0.01, 0.5], fixed=True
        )
        parameters["av_logn_sigma"] = SWParameter(
            "av_logn_sigma", 0.05, [0.001, 0.5], fixed=True
        )
    else:
        # Simple Gaussian extinction
        parameters["av"] = SWParameter("av", 0, [0, 1], fixed=True)
        parameters["sig_av"] = SWParameter("sig_av", 0.015, [0, 1], fixed=True)

    # ------------------------------------------------------------------ #
    # IMF parameters                                                       #
    # ------------------------------------------------------------------ #
    if imf_type == "spl":
        # Single power-law: one slope
        parameters["slope"] = SWParameter("slope", -2.3, [-4, -1])

    elif imf_type == "bpl":
        # Broken power-law: low-mass slope, high-mass slope, break mass
        parameters["alow"] = SWParameter("alow", -1.3, [-2, 0])
        parameters["ahigh"] = SWParameter("ahigh", -2.3, [-4, -1])
        parameters["bm"] = SWParameter("bm", 0.5, [0.2, 0.8])

    elif imf_type == "ln":
        # Lognormal core + high-mass power-law tail
        parameters["mean"] = SWParameter("mean", 0.25, [0.1, 0.8])
        parameters["sigma"] = SWParameter("sigma", 0.6, [0.1, 1])
        parameters["bm"] = SWParameter("bm", 1, [0.8, 1.2])
        parameters["slope"] = SWParameter("slope", -2.3, [-3, -1])

    # ------------------------------------------------------------------ #
    # SFH parameters                                                       #
    # ------------------------------------------------------------------ #
    if sfh_type == "gaussian":
        # 2-D Gaussian in (age, [Fe/H]) with cross-correlation term
        parameters["age"] = SWParameter("age", 5, [0.1, 13.4])
        parameters["sig_age"] = SWParameter("sig_age", 1, [0.1, 5])
        parameters["feh"] = SWParameter("feh", -1, [-4, 1])
        parameters["sig_feh"] = SWParameter("sig_feh", 0.1, [0.05, 1])
        parameters["age_feh_corr"] = SWParameter("age_feh_corr", -0.5, [-1, 0])

    elif sfh_type == "empirical_mdf":
        # Age distribution only; [Fe/H] is drawn from the empirical MDF
        if age_type == "gaussian":
            parameters["age"] = SWParameter("age", 5, [0.1, 13.4])
            parameters["sig_age"] = SWParameter("sig_age", 1, [0.1, 5])
        elif age_type == "exponential":
            # Exponential decay: t0 is the onset time, tau is the e-folding time
            parameters["t0"] = SWParameter("t0", 10, [0.1, 13.4], fixed=True)
            parameters["tau"] = SWParameter("tau", 1, [0.1, 100], fixed=True)

    return SWParameters(parameters, kwargs=kwargs)


# ===========================================================================
# Prior summary printer
# ===========================================================================

def print_prior_summary(parameters, filename=None, verbose=True):
    """
    Print and/or log a human-readable summary of all model parameters.

    For each parameter the summary includes: distribution type, bounds,
    current value, fixed flag, and any extra distribution keyword arguments.

    Parameters
    ----------
    parameters : dict
        Mapping of parameter name → :class:`SWParameter`, typically
        ``SWParameters.dict``.
    filename : str or None, optional
        If provided, the summary is written to this file via Python's
        ``logging`` module (mode ``'w'``, overwrites existing content).
        Default is ``None`` (no file output).
    verbose : bool, optional
        If ``True``, the summary is also printed to stdout. Default is
        ``True``.
    """
    # ---- Stdout output ----
    if verbose:
        print("verbose")
        for name, param in parameters.items():
            print("-" * 10)
            print(name)
            print("-" * 10)
            print("Distribution: ", end=" ")
            print(param.distribution)
            print("Bounds: ", end=" ")
            print(param.bounds)
            print("Value: ", end=" ")
            print(param.value)
            print("Fixed: ", end=" ")
            print(param.fixed)
            print("dist_kwargs: ", end=" ")
            print(param.dist_kwargs)

    # ---- File output ----
    if filename is not None:
        logging.basicConfig(
            filename=filename, filemode="w", format="%(message)s"
        )
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        for name, param in parameters.items():
            logger.info("-" * 10)
            logger.info(name)
            logger.info("-" * 10)
            logger.info("Distribution: ")
            logger.info(param.distribution)
            logger.info("Bounds: ")
            logger.info(param.bounds)
            logger.info("Value: ")
            logger.info(param.value)
            logger.info("Fixed: ")
            logger.info(param.fixed)
            logger.info("dist_kwargs: ")
            logger.info(param.dist_kwargs)


# ===========================================================================
# Joint PyTorch distribution
# ===========================================================================

class MultipleIndependent(Distribution):
    """
    Joint PyTorch distribution built from an ordered sequence of independent marginals.

    Each element of the input sequence is treated as statistically independent
    of all others. Individual elements may themselves be multivariate with
    *internally* dependent dimensions (e.g. a multivariate Gaussian), but
    no cross-distribution correlations are modelled.

    Examples
    --------
    >>> joint = MultipleIndependent([
    ...     torch.distributions.Gamma(torch.zeros(1), torch.ones(1)),
    ...     torch.distributions.Beta(torch.zeros(1), torch.ones(1)),
    ...     torch.distributions.MultivariateNormal(
    ...         torch.ones(2), torch.tensor([[1., .1], [.1, 1.]])
    ...     ),
    ... ])

    >>> joint = MultipleIndependent([
    ...     torch.distributions.Uniform(torch.zeros(1), torch.ones(1)),
    ...     torch.distributions.Uniform(torch.ones(1), 2.0 * torch.ones(1)),
    ... ])

    Parameters
    ----------
    dists : Sequence[Distribution]
        Ordered sequence of at least two PyTorch distributions. Each
        distribution must have ``batch_shape`` of size ≤ 1 and at least
        one non-scalar dimension (``event_shape`` or ``batch_shape`` > 0).
    validate_args : bool or None, optional
        Forwarded to ``torch.distributions.Distribution.__init__``.
    """

    def __init__(self, dists: Sequence[Distribution], validate_args=None):
        self._check_distributions(dists)

        self.dists = dists

        # Use numel() rather than event_shape because individual distributions
        # may use batch_shape=[1] *or* event_shape=[1] to represent scalars.
        self.dims_per_dist = torch.as_tensor(
            [d.sample().numel() for d in self.dists]
        )
        self.ndims = torch.sum(torch.as_tensor(self.dims_per_dist)).item()

        super().__init__(
            batch_shape=torch.Size([]),          # batch size guaranteed ≤ 1
            event_shape=torch.Size([self.ndims]),  # total event dimension
            validate_args=validate_args,
        )

    # ------------------------------------------------------------------ #
    # Validation helpers                                                   #
    # ------------------------------------------------------------------ #

    def _check_distributions(self, dists):
        """
        Validate that *dists* is a non-trivial sequence of compatible distributions.

        Parameters
        ----------
        dists : any
            Object to validate as an acceptable distribution sequence.

        Raises
        ------
        AssertionError
            If ``dists`` is not a ``Sequence``, has fewer than 2 elements,
            or contains an invalid distribution (see :meth:`_check_distribution`).
        """
        assert isinstance(dists, Sequence), (
            f"The combination of independent priors must be of type Sequence, "
            f"is {type(dists)}."
        )
        assert len(dists) > 1, "Provide at least 2 distributions to combine."

        # Validate each distribution individually
        [self._check_distribution(d) for d in dists]

    def _check_distribution(self, dist: Distribution):
        """
        Validate type and shape of a single input distribution.

        Parameters
        ----------
        dist : Distribution
            A candidate marginal distribution.

        Raises
        ------
        AssertionError
            If ``dist`` is a nested :class:`MultipleIndependent`, is not a
            PyTorch ``Distribution``, has an oversized batch shape, or is
            defined over a pure scalar.
        """
        assert not isinstance(dist, MultipleIndependent), (
            "Nesting of combined distributions is not possible."
        )
        assert isinstance(dist, Distribution), (
            "Distribution must be a PyTorch distribution."
        )
        # Batch shape must be 0 or 1 to avoid ambiguous batch/event semantics
        assert dist.batch_shape in (
            torch.Size([1]),
            torch.Size([0]),
            torch.Size([]),
        ), "The batch shape of every distribution must be smaller or equal to 1."

        assert len(dist.batch_shape) > 0 or len(dist.event_shape) > 0, (
            "One of the distributions you passed is defined over a scalar only. "
            "Make sure to pass distributions with event_shape or batch_shape > 0:\n"
            "  - Instead of Uniform(0.0, 1.0)   pass Uniform(torch.zeros(1), torch.ones(1))\n"
            "  - Instead of Beta(1.0, 2.0)       pass Beta(tensor([1.0]), tensor([2.0]))"
        )

    # ------------------------------------------------------------------ #
    # Core distribution API                                                #
    # ------------------------------------------------------------------ #

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        """
        Draw samples from the joint distribution.

        Parameters
        ----------
        sample_shape : torch.Size, optional
            Shape of the sample batch. Default is ``torch.Size()`` (a single
            sample returned as a 1-D tensor of length ``self.ndims``).

        Returns
        -------
        Tensor
            Concatenated samples with shape ``(ndims,)`` for a single draw
            or ``(n, ndims)`` for a batch of *n* draws.
        """
        # Sample each marginal and concatenate along the last dimension
        sample = torch.cat([d.sample(sample_shape) for d in self.dists], dim=-1)

        # Reshape to ensure consistent output for scalar vs. batch sample shapes
        if sample_shape == torch.Size():
            sample = sample.reshape(self.ndims)
        else:
            sample = sample.reshape(-1, self.ndims)

        return sample

    def log_prob(self, value) -> Tensor:
        """
        Evaluate the joint log-probability of *value*.

        The log-probability is computed as the sum of each marginal's
        log-probability evaluated on its corresponding slice of *value*.

        Parameters
        ----------
        value : Tensor of shape (ndims,) or (N, ndims)
            Point(s) at which to evaluate the joint log-PDF.

        Returns
        -------
        Tensor of shape (N,)
            Joint log-probability for each sample in the batch.
        """
        value = self._prepare_value(value)

        num_samples = value.shape[0]
        log_probs = []
        dims_covered = 0

        for idx, d in enumerate(self.dists):
            ndims = self.dims_per_dist[idx].item()

            # Extract the slice of dimensions belonging to this marginal
            v = value[:, dims_covered : dims_covered + ndims]

            # Reshape to (N, 1) so all log_probs can be concatenated uniformly
            log_probs.append(d.log_prob(v).reshape(num_samples, 1))
            dims_covered += ndims

        # Sum across marginals to obtain the joint log-probability
        return torch.cat(log_probs, dim=1).sum(-1)

    def _prepare_value(self, value) -> Tensor:
        """
        Ensure *value* has the expected 2-D shape ``(N, ndims)``.

        Parameters
        ----------
        value : Tensor
            Input tensor with 1 or 2 dimensions.

        Returns
        -------
        Tensor of shape (N, ndims)
            Value tensor with a leading batch dimension added if necessary.

        Raises
        ------
        AssertionError
            If *value* has more than 2 dimensions or the second dimension
            does not match ``self.ndims``.
        """
        if value.ndim < 2:
            value = value.unsqueeze(0)

        assert value.ndim == 2, (
            f"value in log_prob must have ndim <= 2, it is {value.ndim}."
        )

        batch_shape, num_value_dims = value.shape

        assert num_value_dims == self.ndims, (
            f"Number of dimensions must match dimensions of this joint: {self.ndims}."
        )

        return value

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def mean(self) -> Tensor:
        """
        Mean of the joint distribution.

        Returns
        -------
        Tensor of shape (ndims,)
            Concatenated means of all marginal distributions.
        """
        return torch.cat([d.mean for d in self.dists])

    @property
    def variance(self) -> Tensor:
        """
        Variance of the joint distribution.

        Returns
        -------
        Tensor of shape (ndims,)
            Concatenated variances of all marginal distributions.
        """
        return torch.cat([d.variance for d in self.dists])


# ===========================================================================
# Module-level entry point (quick sanity check)
# ===========================================================================

if __name__ == "__main__":
    p = make_params("bpl", sfh_type="gaussian", dm_type="gaussian", av_type="lognormal")
    print(make_prior(p))
    print_prior_summary(p.dict)