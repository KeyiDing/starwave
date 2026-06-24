"""
plot.py
-------
Data-loading and CMD visualisation utilities for StarWave.

Provides two classes:

* :class:`MagData`   - container for observed photometric catalogues with
                       CMD plotting and format-conversion helpers.
* :class:`SimTable`  - loader for artificial-star simulation dictionaries,
                       with support for single and multi-file concatenation.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import bz2
import pickle

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas
import matplotlib.pyplot as plt


# ===========================================================================
# Observed photometric catalogue
# ===========================================================================

class MagData:
    """
    Container for an observed photometric catalogue.

    Can be populated either by passing magnitudes directly as keyword
    arguments at construction time, or by calling :meth:`load_catalog` to
    read a compressed pickle file on disk.

    Parameters
    ----------
    **kwargs : optional
        If provided, must include:

        * ``magnitudes`` - array-like of shape ``(N, B)`` containing
          magnitudes for *N* stars in *B* bands.
        * ``names``      - list of *B* band-name strings used as column
          headers in the internal DataFrame.

    Attributes
    ----------
    mags : ndarray of shape (N, B)
        Raw magnitude array.
    names : list of str
        Photometric band names corresponding to each column of ``mags``.
    data : pandas.DataFrame
        DataFrame view of ``mags`` with ``names`` as column headers.

    Examples
    --------
    >>> md = MagData(magnitudes=mag_array, names=['F606W', 'F814W'])
    >>> cmd = md.to_cmd('F606W', 'F814W')

    >>> md = MagData()
    >>> md.load_catalog('catalog.bz2')
    """

    def __init__(self, **kwargs):
        if len(kwargs) > 0:
            print(kwargs)
            self.mags = kwargs["magnitudes"]
            self.names = kwargs["names"]
            self.data = pandas.DataFrame(self.mags, columns=self.names)

    def to_cmd(self, band1, band2):
        """
        Convert stored magnitudes to a two-column CMD array.

        Returns the CMD as a ``(color, magnitude)`` array where:

        * magnitude = ``band2``
        * color     = ``band1 − band2``

        Parameters
        ----------
        band1 : str
            Name of the bluer band (used to compute the color).
        band2 : str
            Name of the redder band (used as the magnitude axis).

        Returns
        -------
        ndarray of shape (N, 2)
            Each row is ``[color, magnitude]`` where
            ``color = band1 − band2``.
        """
        mag = self.data[band2]
        color = self.data[band1] - self.data[band2]
        return np.asarray([color, mag]).T

    def plot_cmd(self, band1=None, band2=None):
        """
        Plot a color-magnitude diagram for the stored catalogue.

        If ``band1`` or ``band2`` are not provided, the first two band names
        in ``self.names`` are used as defaults.

        Parameters
        ----------
        band1 : str or None, optional
            Bluer band name (color = ``band1 − band2``). Defaults to
            ``self.names[0]``.
        band2 : str or None, optional
            Redder band name (magnitude axis). Defaults to
            ``self.names[1]``.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the CMD scatter plot, with the
            magnitude axis inverted (brighter stars at the top).
        """
        # Fall back to the first two stored bands if none are specified
        if band1 is None:
            band1 = self.names[0]
        if band2 is None:
            band2 = self.names[1]

        mag = self.data[band2]
        color = self.data[band1] - self.data[band2]

        f = plt.figure(figsize=(8, 5))
        plt.scatter(color, mag, s=10, alpha=0.75, color="k")
        plt.gca().invert_yaxis()
        plt.ylabel(band1)
        plt.xlabel(band1 + "$-$" + band2)
        return f

    def load_catalog(self, filename):
        """
        Load a photometric catalogue from a bz2-compressed pickle file.

        The pickle file must be a dictionary where each key is a band name
        and its value is an array of per-star magnitudes. A special boolean
        key ``'dat_det'`` is expected; only stars where ``dat_det`` is
        ``True`` are retained.

        Parameters
        ----------
        filename : str
            Path to the ``.bz2`` compressed pickle file.

        Side-effects
        ------------
        Sets ``self.mags``, ``self.names``, and ``self.data`` in-place.
        Prints ``'catalog loaded!'`` on success.
        """
        # Load the compressed catalogue dictionary
        with bz2.BZ2File(filename, "rb") as f:
            catalog = pickle.load(f)

        # Extract and remove the detection mask before iterating over bands
        mask = catalog["dat_det"]
        del catalog["dat_det"]

        # Collect per-band magnitude arrays and their names
        mags = []
        names = []
        for key in catalog.keys():
            mags.append(catalog[key])
            names.append(key)

        # Stack into (N, B) array and apply the detection mask
        mags = np.asarray(mags).T[mask]

        self.mags = mags
        self.names = names
        self.data = pandas.DataFrame(self.mags, columns=self.names)
        print("catalog loaded!")

    def rename(self, newnames):
        """
        Rename the photometric bands in-place.

        Updates both ``self.names`` and the column headers of ``self.data``.

        Parameters
        ----------
        newnames : list of str
            New band names. Must have the same length as the current
            ``self.names``.
        """
        self.names = newnames
        self.data.columns = newnames


# ===========================================================================
# Artificial-star simulation table
# ===========================================================================

class SimTable:
    """
    Loader and container for artificial-star simulation dictionaries.

    Reads one or more bz2-compressed pickle files containing the results of
    an artificial-star experiment and exposes the combined data as a
    :class:`pandas.DataFrame` (``self.simdf``).

    Attributes
    ----------
    simdf : pandas.DataFrame
        Combined simulation DataFrame with columns including
        ``'output_mag1'``, ``'output_mag2'``, ``'input_mag1'``,
        ``'input_mag2'``, and any other keys present in the source
        dictionaries.

    Examples
    --------
    >>> st = SimTable()
    >>> st.load_simdict('artificial_stars.bz2')
    >>> st.simdf.head()

    >>> st.load_simdict(['part1.bz2', 'part2.bz2'])   # concatenate files
    """

    def __init__(self, **kwargs):
        pass

    def load_simdict(self, sim_dict):
        """
        Load and flatten an artificial-star simulation dictionary.

        Accepts either a single filename or a list of filenames. When a
        list is provided the dictionaries are concatenated along the star
        axis before being converted to a DataFrame.

        In all cases the compound ``'Output Mags'`` and ``'Input Mags'``
        columns (shape ``(N, 2)``) are split into individual scalar columns
        (``output_mag1``, ``output_mag2``, ``input_mag1``, ``input_mag2``)
        and the originals are removed.

        Parameters
        ----------
        sim_dict : str or list of str
            Path to a single bz2-compressed pickle file, or an ordered list
            of paths to be concatenated (first file first).

        Side-effects
        ------------
        Sets ``self.simdf`` in-place. Prints progress messages to stdout.
        Prints ``'simulation dataframe loaded!'`` on success.
        """
        if isinstance(sim_dict, str):
            # ---- Single-file path ----
            with bz2.BZ2File(sim_dict, "rb") as f:
                print("loading dictionary...")
                simdict = pickle.load(f)

        elif isinstance(sim_dict, list):
            # ---- Multi-file path: load and concatenate ----
            with bz2.BZ2File(sim_dict[0], "rb") as f:
                print("loading first dictionary...")
                simdict = pickle.load(f)

            # Split the 2-D magnitude arrays into scalar columns for concatenation
            simdict["outmag1"] = simdict["Output Mags"][:, 0]
            simdict["outmag2"] = simdict["Output Mags"][:, 1]

            # Append each subsequent file to the running dictionary
            for kk in range(len(sim_dict) - 1):
                with bz2.BZ2File(sim_dict[kk + 1]) as f:
                    t_simdict = pickle.load(f)
                    t_simdict["outmag1"] = t_simdict["Output Mags"][:, 0]
                    t_simdict["outmag2"] = t_simdict["Output Mags"][:, 1]

                for key in simdict.keys():
                    simdict[key] = np.append(simdict[key], t_simdict[key])
                print("adding additional dictionary...")

            # Reassemble the concatenated magnitude columns back into a 2-D array
            simdict["Output Mags"] = np.vstack(
                (simdict["outmag1"], simdict["outmag2"])
            ).T

            # Remove the temporary scalar columns used for concatenation
            del simdict["outmag1"]
            del simdict["outmag2"]

        # ---- Flatten 2-D magnitude arrays into named scalar columns ----
        simdict["output_mag1"] = simdict["Output Mags"][:, 0]
        simdict["output_mag2"] = simdict["Output Mags"][:, 1]
        simdict["input_mag1"] = simdict["Input Mags"][:, 0]
        simdict["input_mag2"] = simdict["Input Mags"][:, 1]

        # Remove the original compound columns now that they have been split
        del simdict["Output Mags"]
        del simdict["Input Mags"]

        self.simdf = pandas.DataFrame.from_dict(simdict)
        print("simulation dataframe loaded!")