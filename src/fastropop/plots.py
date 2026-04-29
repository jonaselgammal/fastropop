"""Plotting helpers for semi-analytic population outputs."""

from __future__ import annotations

import numpy as np


def plot_binned_spectrum(spec, freqs=None, hc2_values=None):
    """Plot a binned characteristic-strain spectrum."""
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
    """Plot sampled distributions against their target PDFs."""
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
    """Plot many GW-spectrum realizations and their central interval."""
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
    """Plot one HEALPix skymap slice for a selected frequency bin."""
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
