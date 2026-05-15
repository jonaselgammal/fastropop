"""Plotting helpers for semi-analytic population outputs."""

from __future__ import annotations

import numpy as np


def plot_binned_spectrum(spec, freqs=None, hc2_values=None):
    r"""
    Plot a PTA-binned characteristic-strain spectrum.

    Parameters
    ----------
    spec : array-like, shape (n_bins, 2)
        Two-column spectrum array returned by `fastropop.binning`.
        Column 0 contains \(\log_{10} f\), while column 1 contains the
        binned \(f h_c^2\)-style quantity before the final square root and
        logarithm are applied for display.
    freqs : array-like, optional
        Observer-frame reference frequencies in Hz for an overplotted smooth
        comparison curve.
    hc2_values : array-like, optional
        Reference characteristic-strain values or equivalent comparison curve
        to plot against the binned realization.
    """
    import matplotlib.pyplot as plt

    spec = np.asarray(spec)
    if freqs is not None:
        freqs = np.asarray(freqs)
    if hc2_values is not None:
        hc2_values = np.asarray(hc2_values)

    plt.figure(figsize=(8, 6))
    plt.plot(spec[:, 0], np.log10(np.sqrt(spec[:, 1])), "o-", label="Binned Spectrum", color="C3")

    if freqs is not None and hc2_values is not None:
        plt.plot(np.log10(freqs), np.log10(hc2_values), "-", color="C0", label="log hc ref")

    plt.xlabel(r"$\log_{10}(f)\ [\mathrm{Hz}]$")
    plt.ylabel(r"$\log_{10}(h_c)$")
    plt.title("Characteristic Strain Spectrum")
    plt.legend()
    plt.grid(True, which="both", alpha=0.6)
    plt.ylim(-20, -13)
    plt.show()


def plot_sample_distributions(
    distM,
    distz,
    distlog10f,
    Mgrid,
    pdfM,
    zgrid,
    pdfz,
    fgrid_log10,
    pdflogf,
):
    r"""
    Plot sampled distributions against the target PDFs used for inverse-CDF sampling.

    Parameters
    ----------
    distM : array-like
        Sampled chirp masses returned by
        `fastropop.SemiAnalyticPopulation.sample_dist`.
    distz : array-like
        Sampled redshifts returned by
        `fastropop.SemiAnalyticPopulation.sample_dist`.
    distlog10f : array-like
        Sampled \(\log_{10} f\) values returned by
        `fastropop.SemiAnalyticPopulation.sample_dist`.
    Mgrid : array-like
        Mass grid used to define the target mass PDF.
    pdfM : array-like
        Target mass PDF evaluated on `Mgrid`.
    zgrid : array-like
        Redshift grid used to define the target redshift PDF.
    pdfz : array-like
        Target redshift PDF evaluated on `zgrid`.
    fgrid_log10 : array-like
        Frequency grid in \(\log_{10} f\).
    pdflogf : array-like
        Target PDF evaluated on `fgrid_log10`.
    """
    import matplotlib.pyplot as plt

    distM = np.asarray(distM)
    distz = np.asarray(distz)
    distlog10f = np.asarray(distlog10f)
    Mgrid = np.asarray(Mgrid)
    pdfM = np.asarray(pdfM)
    zgrid = np.asarray(zgrid)
    pdfz = np.asarray(pdfz)
    fgrid_log10 = np.asarray(fgrid_log10)
    pdflogf = np.asarray(pdflogf)

    plt.figure(figsize=(7, 5))
    plt.loglog(Mgrid, pdfM, label="PDF")
    plt.hist(distM, bins=300, density=True, histtype="step", color="C3", label="Random samples")
    plt.xlabel("M [kg]")
    plt.ylabel("PDF")
    plt.ylim(10**-43.5, 10**-35)
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(zgrid, pdfz, label="PDF")
    plt.hist(distz, bins=30, density=True, histtype="step", color="C3", label="Random samples")
    plt.xlabel("z")
    plt.ylabel("PDF")
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.semilogy(fgrid_log10, pdflogf, label="PDF")
    plt.hist(distlog10f, bins=30, density=True, histtype="step", color="C3", label="Random samples")
    plt.xlabel("log10(f) [Hz]")
    plt.ylabel("PDF")
    plt.legend()
    plt.show()


def plot_realizations(log10f, yvals, median, q_low, q_high, freqs=None, hc2_values=None):
    r"""
    Plot many Monte Carlo realizations and their central interval.

    Parameters
    ----------
    log10f : array-like
        Frequency-bin centers in \(\log_{10} f\).
    yvals : array-like, shape (n_realizations, n_bins)
        Realization values, usually \(\log_{10}(h_c)\) after binning.
    median : array-like
        Pointwise median of the realization ensemble.
    q_low : array-like
        Lower envelope of the central interval.
    q_high : array-like
        Upper envelope of the central interval.
    freqs : array-like, optional
        Optional smooth comparison frequency grid in Hz.
    hc2_values : array-like, optional
        Optional smooth comparison strain curve.
    """
    import matplotlib.pyplot as plt

    log10f = np.asarray(log10f)
    yvals = np.asarray(yvals)
    median = np.asarray(median)
    q_low = np.asarray(q_low)
    q_high = np.asarray(q_high)
    if freqs is not None:
        freqs = np.asarray(freqs)
    if hc2_values is not None:
        hc2_values = np.asarray(hc2_values)

    plt.figure(figsize=(8, 6))
    plt.plot(log10f, median, "C3", lw=2, label="Average")
    plt.plot(log10f, q_low, "C7", lw=2)
    plt.plot(log10f, q_high, "C7", lw=2)

    for realization in yvals:
        plt.plot(log10f, realization, color="C3", alpha=0.1)

    plt.fill_between(log10f, q_low, q_high, color="C7", alpha=0.3, label="$68\\% C.I.$")

    if hc2_values is not None and freqs is not None:
        plt.plot(np.log10(freqs), np.log10(hc2_values), "--", color="C0", label="log hc ref")

    plt.xlabel(r"$\log_{10}(f) [\mathrm{Hz}]$")
    plt.ylabel(r"$\log_{10}(h_c)$")
    plt.title(f"Ensemble of {len(yvals)} realizations")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.show()


def plot_skymap(skymaps, freq_index=0, polarization="total", log_scale=False, title=None):
    r"""
    Plot a single HEALPix skymap slice for one PTA frequency bin.

    Parameters
    ----------
    skymaps : array-like
        Either a stacked array of shape ``(2, npix, n_frequencies)`` containing
        plus and cross polarizations, or a single array of shape
        ``(npix, n_frequencies)``.
    freq_index : int, optional
        Frequency-bin index to display.
    polarization : {"total", "plus", "cross"}, optional
        Which polarization content to visualize. For stacked maps,
        ``"total"`` shows \(\sqrt{|h_+|^2 + |h_\times|^2}\).
    log_scale : bool, optional
        If ``True``, display \(\log_{10}\) of the amplitude.
    title : str, optional
        Plot title. If omitted, a descriptive default is created.
    """
    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("`healpy` is required for skymap plotting.") from exc

    skymaps = np.asarray(skymaps)

    if skymaps.ndim == 3:
        if skymaps.shape[0] != 2:
            raise ValueError("Stacked skymaps must have shape (2, npix, n_frequencies).")
        if polarization == "total":
            map_slice = np.sqrt(
                np.abs(skymaps[0, :, freq_index]) ** 2 + np.abs(skymaps[1, :, freq_index]) ** 2
            )
        elif polarization == "plus":
            map_slice = np.abs(skymaps[0, :, freq_index])
        elif polarization == "cross":
            map_slice = np.abs(skymaps[1, :, freq_index])
        else:
            raise ValueError("polarization must be one of {'total', 'plus', 'cross'}.")
    elif skymaps.ndim == 2:
        map_slice = np.abs(skymaps[:, freq_index])
    else:
        raise ValueError(
            "skymaps must have shape (npix, n_frequencies) or (2, npix, n_frequencies)."
        )

    if log_scale:
        map_slice = np.log10(np.maximum(map_slice, np.finfo(float).tiny))

    if title is None:
        scale_label = "log10 amplitude" if log_scale else "amplitude"
        title = f"{polarization.capitalize()} skymap, bin {freq_index} ({scale_label})"

    hp.mollview(map_slice, title=title, hold=False)


__all__ = ["plot_binned_spectrum", "plot_realizations", "plot_sample_distributions", "plot_skymap"]
