#!/usr/bin/env python3

import argparse
import configparser
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blip.src.geometry import geometry
from blip.src.instrNoise import instrNoise

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams["figure.figsize"] = (10, 6)
mpl.rcParams.update({"font.size": 14})


H0_OVER_h = 100.0 * 1000.0 / 3.085677581491367e22
DEFAULT_NOISE_NP = 9.0e-42
DEFAULT_NOISE_NA = 3.6e-49
PAIR_DEFINITIONS = [
    ("AA", 0, 0, 1.0),
    ("EE", 1, 1, 1.0),
    ("TT", 2, 2, 1.0),
    ("AE", 0, 1, 2.0),
    ("AT", 0, 2, 2.0),
    ("ET", 1, 2, 2.0),
]


def trapezoid_compat(y, x=None, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


class MultipoleSensitivityContext(geometry, instrNoise):
    def __init__(self, params, inj, freqs, f0, tsegmid):
        self.params = params
        self.inj = inj
        self.injection = False
        self.armlength = 2.5e9
        self.fs = freqs
        self.f0 = f0
        self.tsegmid = tsegmid
        self.time_dim = tsegmid.size
        geometry.__init__(self)

        if self.params["tdi_lev"] == "aet":
            self.instr_noise_spectrum = self.aet_noise_spectrum
        elif self.params["tdi_lev"] == "xyz":
            self.instr_noise_spectrum = self.xyz_noise_spectrum
        elif self.params["tdi_lev"] == "michelson":
            self.instr_noise_spectrum = self.mich_noise_spectrum
        else:
            raise ValueError("Unknown tdi_lev '{}'. Expected 'aet', 'xyz', or 'michelson'.".format(self.params["tdi_lev"]))


def parse_multipoles(spec):
    values = []
    for token in spec.split(","):
        token = token.strip()
        if token == "":
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(token))
    values = sorted(set(values))
    if len(values) == 0:
        raise ValueError("No multipoles were requested.")
    if values[0] < 0:
        raise ValueError("All multipoles must be >= 0.")
    return values


def load_config(path):
    config = configparser.ConfigParser()
    read_files = config.read(path)
    if len(read_files) == 0:
        raise FileNotFoundError("Could not read config '{}'.".format(path))
    return config


def parse_noise_levels(config):
    if config.has_option("inj", "truevals"):
        try:
            truevals = eval(str(config.get("inj", "truevals")))
        except Exception:
            truevals = {}
        noise_dict = truevals.get("noise", {})
        np_level = float(noise_dict.get("Np", DEFAULT_NOISE_NP))
        na_level = float(noise_dict.get("Na", DEFAULT_NOISE_NA))
        return np_level, na_level
    return DEFAULT_NOISE_NP, DEFAULT_NOISE_NA


def parse_prior_bounds(config, key, default):
    if config.has_option("priors", key):
        return list(eval(str(config.get("priors", key))))
    return default


def build_params(config):
    params = {
        "fmin": float(config.get("params", "fmin")),
        "fmax": float(config.get("params", "fmax")),
        "dur": float(config.get("params", "duration")),
        "seglen": float(config.get("params", "seglen", fallback=1e5)),
        "fs": float(config.get("params", "fs", fallback=0.25)),
        "Shfile": config.get("params", "Shfile", fallback="LISA_2017_PSD_M.npy"),
        "load_data": int(config.get("params", "load_data", fallback=0)),
        "datatype": str(config.get("params", "datatype", fallback="strain")),
        "datafile": str(config.get("params", "datafile", fallback="")),
        "fref": float(config.get("params", "fref", fallback=1e-3)),
        "model": "noise",
        "alias": eval(str(config.get("params", "alias", fallback="{}"))),
        "tdi_lev": str(config.get("params", "tdi_lev", fallback="aet")),
        "lisa_config": str(config.get("params", "lisa_config", fallback="orbiting")),
        "nside": int(config.get("params", "nside", fallback=4)),
        "lmax": int(config.get("params", "lmax", fallback=1)),
        "multipole_l": int(config.get("params", "multipole_l", fallback=0)),
        "tstart": float(config.get("params", "tstart", fallback=0)),
        "prior_bounds": {
            "log_Np": parse_prior_bounds(config, "log_Np", [-44, -39]),
            "log_Na": parse_prior_bounds(config, "log_Na", [-51, -46]),
            "alpha": parse_prior_bounds(config, "alpha", [-5, 5]),
            "log_omega0": parse_prior_bounds(config, "log_omega0", [-14, 8]),
            "log_A_L": parse_prior_bounds(config, "log_A_L", [-14, 8]),
            "log_A_c": parse_prior_bounds(
                config,
                "log_A_c",
                parse_prior_bounds(config, "log_A_L", [-14, 8]),
            ),
            "alpha1": parse_prior_bounds(config, "alpha1", [-4, 6]),
            "alpha2": parse_prior_bounds(config, "alpha2", [0, 40]),
            "log_fbreak": parse_prior_bounds(config, "log_fbreak", [-4, -2]),
            "log_fcut": parse_prior_bounds(config, "log_fcut", [-4, -2]),
            "log_fscale": parse_prior_bounds(config, "log_fscale", [-4, -2]),
        },
        "debug_priors": 0,
        "sph_flag": True,
        "sample_method": str(config.get("run_params", "sample_method", fallback="rslice")),
    }
    inj = {
        "doInj": int(config.get("inj", "doInj", fallback=0)),
        "multipole_l": int(config.get("inj", "multipole_l", fallback=params["multipole_l"])),
        "inj_lmax": int(
            config.get(
                "inj",
                "inj_lmax",
                fallback=max(params["lmax"], params["multipole_l"]),
            )
        ),
        "sph_flag": True,
        "pop_flag": False,
    }
    return params, inj


def get_log_ac_prior_bounds(params):
    if "log_A_c" in params["prior_bounds"]:
        return params["prior_bounds"]["log_A_c"]
    return params["prior_bounds"].get("log_A_L", [-14, 8])


def build_analysis_axes(params):
    fstar = 3e8 / (2 * np.pi * 2.5e9)
    nperseg = int(params["fs"] * params["seglen"])
    if nperseg <= 0:
        raise ValueError("seglen * fs must be >= 1.")

    freqs = np.fft.rfftfreq(nperseg, 1.0 / params["fs"])
    band = (freqs >= params["fmin"]) & (freqs <= params["fmax"])
    fdata = freqs[band]
    if fdata.size == 0:
        raise ValueError("No analysis frequencies remain after applying fmin/fmax.")

    nsegs = int(np.floor(params["dur"] / params["seglen"])) - 1
    if nsegs <= 0:
        raise ValueError("duration must exceed seglen enough to create at least one analysis segment.")

    tsegmid = params["tstart"] + params["seglen"] * np.arange(nsegs) + 0.5 * params["seglen"]
    f0 = fdata / (2 * fstar)
    delta_f = 1.0 / params["seglen"]

    return fdata, f0, tsegmid, delta_f


def get_response_method(context, tdi_level):
    if tdi_level == "aet":
        return context.asgwb_aet_response
    if tdi_level == "xyz":
        return context.asgwb_xyz_response
    if tdi_level == "michelson":
        return context.asgwb_mich_response
    raise ValueError("Unknown tdi_lev '{}'. Expected 'aet', 'xyz', or 'michelson'.".format(tdi_level))


def get_noise_diagonal(context, freqs, f0, np_level, na_level):
    noise_cov = context.instr_noise_spectrum(freqs, f0, Np=np_level, Na=na_level)
    return np.real(np.stack([noise_cov[0, 0], noise_cov[1, 1], noise_cov[2, 2]], axis=0))


def compute_multipole_response(context, freqs, f0, tsegmid, multipole_l):
    response_method = get_response_method(context, context.params["tdi_lev"])
    response_basis = response_method(f0, tsegmid, set_almax=multipole_l)

    lm_indices = []
    for alm_idx in range(response_basis.shape[-1]):
        lval, _ = context.idxtoalm(multipole_l, alm_idx)
        if lval == multipole_l:
            lm_indices.append(alm_idx)

    if len(lm_indices) != (2 * multipole_l + 1):
        raise ValueError(
            "Expected {} (l,m) coefficients for L={}, found {}.".format(
                2 * multipole_l + 1,
                multipole_l,
                len(lm_indices),
            )
        )

    # The paper's rotation-invariant single-L response is sqrt(sum_m |R_lm|^2).
    # We preserve the time integral by averaging the squared response over the analysis segments.
    response_power = np.mean(np.sum(np.abs(response_basis[..., lm_indices]) ** 2, axis=-1), axis=-1)
    return np.sqrt(np.maximum(response_power, 0.0))


def compute_channel_sensitivities(freqs, response_l, noise_diag):
    prefactor = 4.0 * np.pi**2 * np.sqrt(4.0 * np.pi) / (3.0 * H0_OVER_h**2)

    pair_curves = {}
    invsq_sum = np.zeros_like(freqs, dtype=float)

    for label, ii, jj, weight in PAIR_DEFINITIONS:
        response_pair = np.maximum(np.abs(response_l[ii, jj]), 0.0)
        noise_pair = np.sqrt(np.maximum(noise_diag[ii], 0.0) * np.maximum(noise_diag[jj], 0.0))
        omega_pair = np.full_like(freqs, np.inf, dtype=float)
        valid = response_pair > 0.0
        omega_pair[valid] = prefactor * freqs[valid] ** 3 * noise_pair[valid] / response_pair[valid]
        pair_curves[label] = omega_pair

        finite = np.isfinite(omega_pair) & (omega_pair > 0.0)
        invsq_sum[finite] += weight / omega_pair[finite] ** 2

    omega_total = np.full_like(freqs, np.inf, dtype=float)
    valid_total = invsq_sum > 0.0
    omega_total[valid_total] = 1.0 / np.sqrt(invsq_sum[valid_total])

    return pair_curves, omega_total


def compute_minimum_pivot_amplitudes(freqs, omega_total, alpha, pivot_frequency, duration, delta_f, snr_target, bin_snr_target):
    spectral_shape = (freqs / pivot_frequency) ** alpha

    ac_min_bin = np.full_like(freqs, np.inf, dtype=float)
    finite = np.isfinite(omega_total) & (omega_total > 0.0) & np.isfinite(spectral_shape) & (spectral_shape != 0.0)
    ac_min_bin[finite] = bin_snr_target * omega_total[finite] / (np.sqrt(duration * delta_f) * np.abs(spectral_shape[finite]))

    integrand = np.zeros_like(freqs, dtype=float)
    integrand[finite] = (spectral_shape[finite] / omega_total[finite]) ** 2
    band_power = trapezoid_compat(integrand, freqs)
    if band_power <= 0.0:
        ac_min_band = np.inf
    else:
        ac_min_band = snr_target / np.sqrt(duration * band_power)

    if np.any(np.isfinite(ac_min_bin)):
        best_idx = np.nanargmin(np.where(np.isfinite(ac_min_bin), ac_min_bin, np.nan))
        best_bin = {
            "frequency_hz": float(freqs[best_idx]),
            "ac_min_bin_h2": float(ac_min_bin[best_idx]),
            "omega_n_h2": float(omega_total[best_idx]),
        }
    else:
        best_bin = {
            "frequency_hz": np.nan,
            "ac_min_bin_h2": np.nan,
            "omega_n_h2": np.nan,
        }

    return ac_min_bin, float(ac_min_band), best_bin


def write_curve_table(out_dir, freqs, curve_rows):
    header_parts = ["freq_hz"]
    data_cols = [freqs]

    for row in curve_rows:
        label = "L{:02d}".format(row["multipole_l"])
        header_parts.append("omega_n_{}_h2".format(label))
        header_parts.append("ac_min_bin_{}_h2".format(label))
        data_cols.append(row["omega_total"])
        data_cols.append(row["ac_min_bin"])

    table = np.column_stack(data_cols)
    np.savetxt(out_dir / "multipole_sensitivity_curves.txt", table, header=" ".join(header_parts))


def write_band_summary(out_dir, rows, alpha, pivot_frequency, duration, delta_f, snr_target, bin_snr_target, np_level, na_level):
    header = (
        "multipole_l alpha pivot_frequency_hz duration_s delta_f_hz "
        "band_snr_target bin_snr_target noise_Np noise_Na "
        "best_bin_frequency_hz best_bin_omega_n_h2 best_bin_ac_min_h2 ac_min_band_h2"
    )
    table = []
    for row in rows:
        table.append(
            [
                row["multipole_l"],
                alpha,
                pivot_frequency,
                duration,
                delta_f,
                snr_target,
                bin_snr_target,
                np_level,
                na_level,
                row["best_bin"]["frequency_hz"],
                row["best_bin"]["omega_n_h2"],
                row["best_bin"]["ac_min_bin_h2"],
                row["ac_min_band"],
            ]
        )
    np.savetxt(out_dir / "multipole_sensitivity_summary.txt", np.asarray(table), header=header)


def make_plots(out_dir, rows, alpha, pivot_frequency, snr_target):
    fig1, ax1 = plt.subplots()
    for row in rows:
        ax1.plot(row["freqs"], row["omega_total"], label="L={}".format(row["multipole_l"]))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel(r"$\Omega_{{\rm GW},n}^{L}(f)\,h^2$")
    ax1.set_title("Single-L sensitivity kernel")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "multipole_omega_sensitivity.png", dpi=200)
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    for row in rows:
        ax2.plot(row["freqs"], row["ac_min_bin"], label="L={}".format(row["multipole_l"]))
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel(r"$A_{{c,{\rm min}}}^{\rm bin}(f)\,h^2$")
    ax2.set_title(r"Per-bin minimum pivot amplitude ($\alpha={:.3g}$)".format(alpha))
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "multipole_ac_min_bin.png", dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    multipoles = [row["multipole_l"] for row in rows]
    ac_band = [row["ac_min_band"] for row in rows]
    ax3.plot(multipoles, ac_band, marker="o")
    ax3.set_yscale("log")
    ax3.set_xlabel("Multipole L")
    ax3.set_ylabel(r"$A_{{c,{\rm min}}}^{\rm band}\,h^2$")
    ax3.set_title(
        r"Band-integrated minimum pivot amplitude ($\alpha={:.3g}$, SNR={:.3g}$)".format(alpha, snr_target)
    )
    fig3.tight_layout()
    fig3.savefig(out_dir / "multipole_ac_min_band.png", dpi=200)
    plt.close(fig3)


class MultipoleForecastSubmodel:
    def __init__(self, parameters):
        self.parameters = parameters


class MultipoleForecastModel:
    def __init__(self, params, freqs, sigma, pivot_frequency, data, multipole_l):
        self.params = params
        self.freqs = freqs
        self.sigma = sigma
        self.pivot_frequency = pivot_frequency
        self.data = data
        self.multipole_l = multipole_l
        self.parameters = {"all": [r"$\alpha$", r"$\log_{10}(A_c)$"]}
        self.Npar = len(self.parameters["all"])
        self.submodel_names = ["forecast"]
        self.submodels = {"forecast": MultipoleForecastSubmodel(self.parameters["all"])}

    def prior(self, unit_theta):
        alpha_bounds = self.params["prior_bounds"]["alpha"]
        log_ac_bounds = get_log_ac_prior_bounds(self.params)
        alpha = alpha_bounds[0] + unit_theta[0] * (alpha_bounds[1] - alpha_bounds[0])
        log_ac = log_ac_bounds[0] + unit_theta[1] * (log_ac_bounds[1] - log_ac_bounds[0])
        return [alpha, log_ac]

    def likelihood(self, theta):
        alpha, log_ac = theta
        model = (10.0 ** log_ac) * (self.freqs / self.pivot_frequency) ** alpha
        resid = self.data - model
        var = self.sigma ** 2
        logl = -0.5 * np.sum(resid ** 2 / var + np.log(2.0 * np.pi * var))
        return float(np.real(logl))


class MultipoleForecastRunner:
    def __init__(self, model):
        self.Model = model


class MultipoleForecastInjection:
    def __init__(self, alpha_true, log_ac_true):
        self.component_names = ["forecast"]
        self.truevals = {
            "forecast": {
                r"$\alpha$": alpha_true,
                r"$\log_{10}(A_c)$": log_ac_true,
            }
        }


def simple_interval(samples):
    return tuple(np.quantile(samples, [0.025, 0.5, 0.975]))


def density_interval(grid, density):
    area = trapezoid_compat(density, grid)
    if not np.isfinite(area) or area <= 0.0:
        return np.nan, np.nan, np.nan

    density = density / area
    cdf = np.zeros_like(grid, dtype=float)
    if grid.size > 1:
        cdf[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))

    if cdf[-1] <= 0.0:
        return np.nan, np.nan, np.nan

    cdf = cdf / cdf[-1]
    return tuple(np.interp([0.025, 0.5, 0.975], cdf, grid))


def highest_density_levels(pdf, masses=(0.68, 0.95)):
    flat = np.ravel(pdf)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0 or np.sum(flat) <= 0.0:
        return []

    sort_desc = np.sort(flat)[::-1]
    cdf = np.cumsum(sort_desc)
    cdf = cdf / cdf[-1]

    levels = []
    for mass in masses:
        idx = np.searchsorted(cdf, mass, side="left")
        idx = min(idx, sort_desc.size - 1)
        levels.append(sort_desc[idx])

    return sorted(set(levels))


def plot_grid_corner(
    run_dir,
    alpha_grid,
    logac_grid,
    posterior_pdf,
    alpha_pdf,
    logac_pdf,
    alpha_true,
    logac_true,
    alpha_interval,
    logac_interval,
):
    fig = plt.figure(figsize=(10, 10))
    grid_spec = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.05)

    ax_alpha = fig.add_subplot(grid_spec[0, 0])
    ax_blank = fig.add_subplot(grid_spec[0, 1])
    ax_joint = fig.add_subplot(grid_spec[1, 0], sharex=ax_alpha)
    ax_logac = fig.add_subplot(grid_spec[1, 1], sharey=ax_joint)

    ax_blank.axis("off")

    ax_alpha.plot(alpha_grid, alpha_pdf, color="black", lw=1.5)
    ax_alpha.axvline(alpha_true, color="tab:green", ls="--", lw=1.2)
    ax_alpha.axvline(alpha_interval[1], color="tab:blue", lw=1.2)
    ax_alpha.axvline(alpha_interval[0], color="tab:blue", ls=":", lw=1.0)
    ax_alpha.axvline(alpha_interval[2], color="tab:blue", ls=":", lw=1.0)
    ax_alpha.set_ylabel("Posterior density")
    ax_alpha.tick_params(axis="x", labelbottom=False)
    ax_alpha.set_title(
        r"$\alpha = {:.3g}^{{+{:.3g}}}_{{-{:.3g}}}$".format(
            alpha_interval[1],
            alpha_interval[2] - alpha_interval[1],
            alpha_interval[1] - alpha_interval[0],
        ),
        loc="left",
    )

    alpha_mesh, logac_mesh = np.meshgrid(alpha_grid, logac_grid, indexing="ij")
    pcm = ax_joint.pcolormesh(alpha_mesh, logac_mesh, posterior_pdf, shading="auto", cmap="magma")
    levels = highest_density_levels(posterior_pdf)
    if len(levels) > 0:
        ax_joint.contour(alpha_mesh, logac_mesh, posterior_pdf, levels=levels, colors="white", linewidths=1.0)
    ax_joint.axvline(alpha_true, color="tab:green", ls="--", lw=1.2)
    ax_joint.axhline(logac_true, color="tab:green", ls="--", lw=1.2)
    ax_joint.set_xlabel(r"$\alpha$")
    ax_joint.set_ylabel(r"$\log_{10}(A_c)$")

    ax_logac.plot(logac_pdf, logac_grid, color="black", lw=1.5)
    ax_logac.axhline(logac_true, color="tab:green", ls="--", lw=1.2)
    ax_logac.axhline(logac_interval[1], color="tab:blue", lw=1.2)
    ax_logac.axhline(logac_interval[0], color="tab:blue", ls=":", lw=1.0)
    ax_logac.axhline(logac_interval[2], color="tab:blue", ls=":", lw=1.0)
    ax_logac.tick_params(axis="y", labelleft=False)
    ax_logac.set_xlabel("Posterior density")
    ax_logac.set_title(
        r"$\log_{{10}}(A_c) = {:.3g}^{{+{:.3g}}}_{{-{:.3g}}}$".format(
            logac_interval[1],
            logac_interval[2] - logac_interval[1],
            logac_interval[1] - logac_interval[0],
        ),
        loc="left",
    )

    cbar = fig.colorbar(pcm, ax=[ax_alpha, ax_joint, ax_logac], fraction=0.03, pad=0.02)
    cbar.set_label("Posterior density")

    fig.savefig(run_dir / "corners.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def sample_grid_posterior(alpha_grid, logac_grid, posterior_pdf, nsamples, seed):
    rng = np.random.RandomState(seed)
    weights = posterior_pdf.ravel().astype(float)
    if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0.0:
        return np.empty((0, 2))

    weights = weights / np.sum(weights)
    draw_idx = rng.choice(weights.size, size=nsamples, p=weights)
    alpha_idx, logac_idx = np.unravel_index(draw_idx, posterior_pdf.shape)

    dalpha = alpha_grid[1] - alpha_grid[0] if alpha_grid.size > 1 else 0.0
    dlogac = logac_grid[1] - logac_grid[0] if logac_grid.size > 1 else 0.0

    alpha_samples = alpha_grid[alpha_idx] + rng.uniform(-0.5 * dalpha, 0.5 * dalpha, size=nsamples)
    logac_samples = logac_grid[logac_idx] + rng.uniform(-0.5 * dlogac, 0.5 * dlogac, size=nsamples)

    alpha_samples = np.clip(alpha_samples, alpha_grid[0], alpha_grid[-1])
    logac_samples = np.clip(logac_samples, logac_grid[0], logac_grid[-1])

    return np.column_stack([alpha_samples, logac_samples])


def run_grid_posterior_forecast(
    run_dir,
    params,
    freqs,
    sigma,
    pivot_frequency,
    data,
    multipole_l,
    log10_ac_inj,
    alpha_inj,
    seed,
    grid_size,
    posterior_samples,
):
    alpha_bounds = params["prior_bounds"]["alpha"]
    logac_bounds = get_log_ac_prior_bounds(params)

    alpha_grid = np.linspace(alpha_bounds[0], alpha_bounds[1], grid_size)
    logac_grid = np.linspace(logac_bounds[0], logac_bounds[1], grid_size)
    var = sigma ** 2
    log_norm = np.sum(np.log(2.0 * np.pi * var))
    loglike = np.empty((alpha_grid.size, logac_grid.size), dtype=float)

    amp_grid = 10.0 ** logac_grid
    scaled_freq = freqs / pivot_frequency
    for alpha_idx, alpha_val in enumerate(alpha_grid):
        model_shape = scaled_freq ** alpha_val
        model_grid = amp_grid[:, None] * model_shape[None, :]
        resid = data[None, :] - model_grid
        loglike[alpha_idx, :] = -0.5 * np.sum(resid ** 2 / var[None, :], axis=1) - 0.5 * log_norm

    max_loglike = np.max(loglike)
    like_shifted = np.exp(loglike - max_loglike)

    dalpha = alpha_grid[1] - alpha_grid[0] if alpha_grid.size > 1 else 1.0
    dlogac = logac_grid[1] - logac_grid[0] if logac_grid.size > 1 else 1.0
    prior_area = (alpha_bounds[1] - alpha_bounds[0]) * (logac_bounds[1] - logac_bounds[0])
    logz = max_loglike + np.log(np.sum(like_shifted) * dalpha * dlogac) - np.log(prior_area)

    posterior_pdf = like_shifted / np.sum(like_shifted)
    posterior_pdf = posterior_pdf / (dalpha * dlogac)

    alpha_pdf = trapezoid_compat(posterior_pdf, logac_grid, axis=1)
    logac_pdf = trapezoid_compat(posterior_pdf, alpha_grid, axis=0)

    alpha_interval = density_interval(alpha_grid, alpha_pdf)
    logac_interval = density_interval(logac_grid, logac_pdf)
    post_samples = sample_grid_posterior(
        alpha_grid,
        logac_grid,
        posterior_pdf * dalpha * dlogac,
        posterior_samples,
        seed + 2000 + multipole_l,
    )

    np.savetxt(run_dir / "post_samples.txt", post_samples)
    np.savetxt(run_dir / "logz.txt", np.atleast_1d(logz))
    np.savetxt(run_dir / "logzerr.txt", np.atleast_1d(np.nan))
    np.savez(
        run_dir / "posterior_grid.npz",
        alpha_grid=alpha_grid,
        log10_Ac_grid=logac_grid,
        posterior_pdf=posterior_pdf,
        alpha_pdf=alpha_pdf,
        log10_Ac_pdf=logac_pdf,
        loglike=loglike,
    )

    plot_grid_corner(
        run_dir,
        alpha_grid,
        logac_grid,
        posterior_pdf,
        alpha_pdf,
        logac_pdf,
        alpha_inj,
        log10_ac_inj,
        alpha_interval,
        logac_interval,
    )

    with open(run_dir / "posterior_summary.txt", "w") as outfile:
        outfile.write("method grid\n")
        outfile.write("multipole_l {}\n".format(multipole_l))
        outfile.write("alpha_true {:.12e}\n".format(alpha_inj))
        outfile.write("log10_Ac_true {:.12e}\n".format(log10_ac_inj))
        outfile.write("alpha_q025 {:.12e}\n".format(alpha_interval[0]))
        outfile.write("alpha_median {:.12e}\n".format(alpha_interval[1]))
        outfile.write("alpha_q975 {:.12e}\n".format(alpha_interval[2]))
        outfile.write("log10_Ac_q025 {:.12e}\n".format(logac_interval[0]))
        outfile.write("log10_Ac_median {:.12e}\n".format(logac_interval[1]))
        outfile.write("log10_Ac_q975 {:.12e}\n".format(logac_interval[2]))
        outfile.write("logz {:.12e}\n".format(float(logz)))
        outfile.write("logzerr nan\n")

    return {
        "method": "grid",
        "multipole_l": multipole_l,
        "run_dir": run_dir,
        "logz": float(logz),
        "logzerr": np.nan,
        "alpha_true": alpha_inj,
        "log10_ac_true": log10_ac_inj,
        "alpha_median": alpha_interval[1],
        "log10_ac_median": logac_interval[1],
        "alpha_q025": alpha_interval[0],
        "alpha_q975": alpha_interval[2],
        "log10_ac_q025": logac_interval[0],
        "log10_ac_q975": logac_interval[2],
    }


def run_dynesty_posterior_forecast(
    run_dir,
    params,
    freqs,
    sigma,
    pivot_frequency,
    data,
    multipole_l,
    log10_ac_inj,
    alpha_inj,
    seed,
    nlive,
):
    from blip.src.dynesty_engine import dynesty_engine
    from blip.tools.plotmaker import plotmaker

    model = MultipoleForecastModel(params, freqs, sigma, pivot_frequency, data, multipole_l)
    runner = MultipoleForecastRunner(model)
    engine, parameters = dynesty_engine.define_engine(
        runner,
        params,
        nlive=nlive,
        nthread=1,
        randst=np.random.RandomState(seed + 1000 + multipole_l),
        pool=None,
        resume=False,
    )
    post_samples, logz, logzerr = dynesty_engine.run_engine(engine)

    np.savetxt(run_dir / "post_samples.txt", post_samples)
    np.savetxt(run_dir / "logz.txt", np.atleast_1d(logz))
    np.savetxt(run_dir / "logzerr.txt", np.atleast_1d(logzerr))

    plot_params = dict(params)
    plot_params["out_dir"] = str(run_dir) + "/"
    plot_params["load_data"] = 0
    inj = MultipoleForecastInjection(alpha_inj, log10_ac_inj)
    plotmaker(post_samples, plot_params, parameters, {}, model, Injection=inj)

    alpha_qlo, alpha_med, alpha_qhi = simple_interval(post_samples[:, 0])
    logac_qlo, logac_med, logac_qhi = simple_interval(post_samples[:, 1])
    with open(run_dir / "posterior_summary.txt", "w") as outfile:
        outfile.write("method dynesty\n")
        outfile.write("multipole_l {}\n".format(multipole_l))
        outfile.write("alpha_true {:.12e}\n".format(alpha_inj))
        outfile.write("log10_Ac_true {:.12e}\n".format(log10_ac_inj))
        outfile.write("alpha_q025 {:.12e}\n".format(alpha_qlo))
        outfile.write("alpha_median {:.12e}\n".format(alpha_med))
        outfile.write("alpha_q975 {:.12e}\n".format(alpha_qhi))
        outfile.write("log10_Ac_q025 {:.12e}\n".format(logac_qlo))
        outfile.write("log10_Ac_median {:.12e}\n".format(logac_med))
        outfile.write("log10_Ac_q975 {:.12e}\n".format(logac_qhi))
        outfile.write("logz {:.12e}\n".format(float(np.ravel(logz)[-1])))
        outfile.write("logzerr {:.12e}\n".format(float(np.ravel(logzerr)[-1])))

    return {
        "method": "dynesty",
        "multipole_l": multipole_l,
        "run_dir": run_dir,
        "logz": float(np.ravel(logz)[-1]),
        "logzerr": float(np.ravel(logzerr)[-1]),
        "alpha_true": alpha_inj,
        "log10_ac_true": log10_ac_inj,
        "alpha_median": alpha_med,
        "log10_ac_median": logac_med,
        "alpha_q025": alpha_qlo,
        "alpha_q975": alpha_qhi,
        "log10_ac_q025": logac_qlo,
        "log10_ac_q975": logac_qhi,
    }


def run_posterior_forecast(
    base_out_dir,
    params,
    multipole_l,
    freqs,
    omega_total,
    pivot_frequency,
    delta_f,
    log10_ac_inj,
    alpha_inj,
    noisy,
    seed,
    nlive,
    grid_size,
    posterior_samples,
):
    run_dir = base_out_dir / "posterior_l{:02d}".format(multipole_l)
    run_dir.mkdir(parents=True, exist_ok=True)

    sigma = omega_total / np.sqrt(params["dur"] * delta_f)
    signal = (10.0 ** log10_ac_inj) * (freqs / pivot_frequency) ** alpha_inj
    if noisy:
        rand = np.random.RandomState(seed + multipole_l)
        data = signal + rand.normal(scale=sigma, size=freqs.size)
    else:
        data = signal.copy()

    np.savetxt(run_dir / "forecast_data.txt", np.column_stack([freqs, data, sigma]), header="freq_hz data sigma")

    try:
        return run_dynesty_posterior_forecast(
            run_dir,
            params,
            freqs,
            sigma,
            pivot_frequency,
            data,
            multipole_l,
            log10_ac_inj,
            alpha_inj,
            seed,
            nlive,
        )
    except Exception as exc:
        print(
            "Falling back to direct grid posterior forecast for L={} because the BLIP sampler stack is unavailable: {}".format(
                multipole_l,
                exc,
            )
        )
        return run_grid_posterior_forecast(
            run_dir,
            params,
            freqs,
            sigma,
            pivot_frequency,
            data,
            multipole_l,
            log10_ac_inj,
            alpha_inj,
            seed,
            grid_size,
            posterior_samples,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compute paper-style single-L SGWB sensitivity curves and minimum pivot amplitudes using BLIP response/noise functions."
    )
    parser.add_argument("config", help="BLIP-style ini file with the desired LISA setup.")
    parser.add_argument(
        "--multipoles",
        default="0-10",
        help="Comma-separated list and/or ranges of multipoles, e.g. '0-10' or '0,2,4-6'.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Spectral index alpha for the power-law signal A_c (f/f_c)^alpha.",
    )
    parser.add_argument(
        "--fc",
        type=float,
        default=None,
        help="Pivot frequency in Hz for A_c. Defaults to fref from the config.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=1.0,
        help="Target band-integrated SNR used to define A_c,min^band.",
    )
    parser.add_argument(
        "--bin-snr",
        type=float,
        default=1.0,
        help="Target single-bin SNR used to define A_c,min^bin(f).",
    )
    parser.add_argument(
        "--Np",
        type=float,
        default=None,
        help="Override the position noise level. Defaults to [inj] truevals noise Np or the BLIP paper value.",
    )
    parser.add_argument(
        "--Na",
        type=float,
        default=None,
        help="Override the acceleration noise level. Defaults to [inj] truevals noise Na or the BLIP paper value.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <config_dir>/Storage/multipole_sensitivity/.",
    )
    parser.add_argument(
        "--posterior",
        action="store_true",
        help="Run a BLIP-style posterior forecast on log10(A_c) and alpha for each requested multipole.",
    )
    parser.add_argument(
        "--posterior-log10Ac",
        type=float,
        default=None,
        help="Injected log10(A_c) for the posterior forecast. Defaults to the band-threshold amplitude for each L.",
    )
    parser.add_argument(
        "--posterior-alpha",
        type=float,
        default=None,
        help="Injected alpha for the posterior forecast. Defaults to --alpha.",
    )
    parser.add_argument(
        "--posterior-noisy",
        action="store_true",
        help="Add one Gaussian noise realization to the posterior forecast data. By default the forecast uses the mean signal only.",
    )
    parser.add_argument(
        "--posterior-seed",
        type=int,
        default=42,
        help="Random seed used for posterior forecast noise realizations and nested sampling.",
    )
    parser.add_argument(
        "--posterior-nlive",
        type=int,
        default=500,
        help="Number of live points for the posterior forecast nested sampler.",
    )
    parser.add_argument(
        "--posterior-grid-size",
        type=int,
        default=201,
        help="Number of grid points per dimension for the direct alpha-log10(A_c) posterior fallback.",
    )
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=20000,
        help="Number of posterior samples to draw from the direct posterior grid for saved sample products.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    params, inj = build_params(config)
    np_level, na_level = parse_noise_levels(config)
    if args.Np is not None:
        np_level = args.Np
    if args.Na is not None:
        na_level = args.Na

    multipoles = parse_multipoles(args.multipoles)
    pivot_frequency = params["fref"] if args.fc is None else args.fc
    out_dir = (
        (config_path.parent / "Storage" / "multipole_sensitivity").resolve()
        if args.out_dir is None
        else Path(args.out_dir).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    freqs, f0, tsegmid, delta_f = build_analysis_axes(params)
    context = MultipoleSensitivityContext(params, inj, freqs, f0, tsegmid)
    noise_diag = get_noise_diagonal(context, freqs, f0, np_level, na_level)

    rows = []
    posterior_rows = []
    for multipole_l in multipoles:
        response_l = compute_multipole_response(context, freqs, f0, tsegmid, multipole_l)
        _, omega_total = compute_channel_sensitivities(freqs, response_l, noise_diag)
        ac_min_bin, ac_min_band, best_bin = compute_minimum_pivot_amplitudes(
            freqs,
            omega_total,
            args.alpha,
            pivot_frequency,
            params["dur"],
            delta_f,
            args.snr,
            args.bin_snr,
        )
        rows.append(
            {
                "multipole_l": multipole_l,
                "freqs": freqs,
                "omega_total": omega_total,
                "ac_min_bin": ac_min_bin,
                "ac_min_band": ac_min_band,
                "best_bin": best_bin,
            }
        )

        if args.posterior:
            alpha_inj = args.posterior_alpha if args.posterior_alpha is not None else args.alpha
            if args.posterior_log10Ac is None:
                log10_ac_inj = np.log10(ac_min_band)
            else:
                log10_ac_inj = args.posterior_log10Ac
            posterior_rows.append(
                run_posterior_forecast(
                    out_dir,
                    params,
                    multipole_l,
                    freqs,
                    omega_total,
                    pivot_frequency,
                    delta_f,
                    log10_ac_inj,
                    alpha_inj,
                    args.posterior_noisy,
                    args.posterior_seed,
                    args.posterior_nlive,
                    args.posterior_grid_size,
                    args.posterior_samples,
                )
            )

    write_curve_table(out_dir, freqs, rows)
    write_band_summary(
        out_dir,
        rows,
        args.alpha,
        pivot_frequency,
        params["dur"],
        delta_f,
        args.snr,
        args.bin_snr,
        np_level,
        na_level,
    )
    make_plots(out_dir, rows, args.alpha, pivot_frequency, args.snr)

    print("Wrote multipole sensitivity products to {}".format(out_dir))
    print("Noise levels: Np = {:.6e}, Na = {:.6e}".format(np_level, na_level))
    print(
        "Using alpha = {:.6g}, f_c = {:.6e} Hz, T = {:.6e} s, delta_f = {:.6e} Hz".format(
            args.alpha,
            pivot_frequency,
            params["dur"],
            delta_f,
        )
    )
    for row in rows:
        print(
            "L = {:2d} | best-bin f = {:.6e} Hz | best-bin A_c,min h^2 = {:.6e} | band A_c,min h^2 = {:.6e}".format(
                row["multipole_l"],
                row["best_bin"]["frequency_hz"],
                row["best_bin"]["ac_min_bin_h2"],
                row["ac_min_band"],
            )
        )

    if len(posterior_rows) > 0:
        posterior_table = []
        for row in posterior_rows:
            posterior_table.append(
                [
                    row["multipole_l"],
                    row["alpha_true"],
                    row["log10_ac_true"],
                    row["alpha_q025"],
                    row["alpha_median"],
                    row["alpha_q975"],
                    row["log10_ac_q025"],
                    row["log10_ac_median"],
                    row["log10_ac_q975"],
                    row["logz"],
                    row["logzerr"],
                ]
            )
        np.savetxt(
            out_dir / "posterior_forecast_summary.txt",
            np.asarray(posterior_table),
            header="multipole_l alpha_true log10_Ac_true alpha_q025 alpha_median alpha_q975 log10_Ac_q025 log10_Ac_median log10_Ac_q975 logz logzerr",
        )
        print("Posterior forecast runs:")
        for row in posterior_rows:
            print(
                "L = {:2d} | method = {} | injected log10(A_c) = {:.6f} | median log10(A_c) = {:.6f} | run = {}".format(
                    row["multipole_l"],
                    row.get("method", "unknown"),
                    row["log10_ac_true"],
                    row["log10_ac_median"],
                    row["run_dir"],
                )
            )


if __name__ == "__main__":
    main()
