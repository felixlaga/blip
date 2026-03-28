import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from blip.src.geometry import geometry
from blip.src.instrNoise import instrNoise


mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams["figure.figsize"] = (10, 6)
mpl.rcParams.update({"font.size": 14})


H0_OVER_h = 100.0 * 1000.0 / 3.085677581491367e22
DEFAULT_NOISE_NP = 9.0e-42
DEFAULT_NOISE_NA = 3.6e-49
DEFAULT_LOG_AC_PRIOR = [-14, 8]
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
            raise ValueError(
                "Unknown tdi_lev '{}'. Expected 'aet', 'xyz', or 'michelson'.".format(self.params["tdi_lev"])
            )


def get_log_ac_prior_bounds(params):
    prior_bounds = params.get("prior_bounds", {})
    if "log_A_c" in prior_bounds:
        return prior_bounds["log_A_c"]
    if "log_omega0" in prior_bounds:
        return prior_bounds["log_omega0"]
    return DEFAULT_LOG_AC_PRIOR


def get_noise_levels(inj):
    noise_truevals = inj.get("truevals", {}).get("noise", {})
    np_level = float(noise_truevals.get("log_Np", np.log10(DEFAULT_NOISE_NP)))
    na_level = float(noise_truevals.get("log_Na", np.log10(DEFAULT_NOISE_NA)))
    return 10.0 ** np_level, 10.0 ** na_level


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


def compute_multipole_response(context, f0, tsegmid, multipole_l):
    response_method = get_response_method(context, context.params["tdi_lev"])
    response_basis = response_method(f0, tsegmid, set_almax=multipole_l)

    lm_indices = []
    for alm_idx in range(response_basis.shape[-1]):
        lval, _ = context.idxtoalm(multipole_l, alm_idx)
        if lval == multipole_l:
            lm_indices.append(alm_idx)

    expected_nmodes = 2 * multipole_l + 1
    if len(lm_indices) != expected_nmodes:
        raise ValueError(
            "Expected {} (l,m) coefficients for L={}, found {}.".format(
                expected_nmodes,
                multipole_l,
                len(lm_indices),
            )
        )

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
        finite_bins = np.where(np.isfinite(ac_min_bin), ac_min_bin, np.inf)
        best_idx = int(np.argmin(finite_bins))
        best_bin = {
            "frequency_hz": float(freqs[best_idx]),
            "ac_min_bin_h2": float(ac_min_bin[best_idx]),
            "omega_n_h2": float(omega_total[best_idx]),
        }
    else:
        best_bin = {"frequency_hz": np.nan, "ac_min_bin_h2": np.nan, "omega_n_h2": np.nan}

    return ac_min_bin, float(ac_min_band), best_bin


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


def plot_grid_corner(run_dir, alpha_grid, logac_grid, posterior_pdf, alpha_pdf, logac_pdf, alpha_true, logac_true, alpha_interval, logac_interval):
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
    fig.savefig(os.path.join(run_dir, "corners.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def sample_grid_posterior(alpha_grid, logac_grid, posterior_prob, nsamples, seed):
    rng = np.random.RandomState(seed)
    weights = posterior_prob.ravel().astype(float)
    if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0.0:
        return np.empty((0, 2))

    weights = weights / np.sum(weights)
    draw_idx = rng.choice(weights.size, size=nsamples, p=weights)
    alpha_idx, logac_idx = np.unravel_index(draw_idx, posterior_prob.shape)

    dalpha = alpha_grid[1] - alpha_grid[0] if alpha_grid.size > 1 else 0.0
    dlogac = logac_grid[1] - logac_grid[0] if logac_grid.size > 1 else 0.0
    alpha_samples = alpha_grid[alpha_idx] + rng.uniform(-0.5 * dalpha, 0.5 * dalpha, size=nsamples)
    logac_samples = logac_grid[logac_idx] + rng.uniform(-0.5 * dlogac, 0.5 * dlogac, size=nsamples)
    alpha_samples = np.clip(alpha_samples, alpha_grid[0], alpha_grid[-1])
    logac_samples = np.clip(logac_samples, logac_grid[0], logac_grid[-1])
    return np.column_stack([alpha_samples, logac_samples])


class MultipoleForecastSubmodel:
    def __init__(self, params, freqs, f0, omega_total, response_l, multipole_l, pivot_frequency):
        self.params = params
        self.fs = freqs
        self.f0 = f0
        self.omega_total = omega_total
        self.response_mat = response_l
        self.multipole_l = multipole_l
        self.pivot_frequency = pivot_frequency
        self.spectral_parameters = [r"$\alpha$", r"$\log_{10}(A_c)$"]
        self.spatial_parameters = []
        self.parameters = self.spectral_parameters
        self.Npar = len(self.parameters)
        self.fancyname = "Single-Multipole Power Law"
        self.color = "teal"
        self.has_map = False

    def omegaf(self, fs, alpha, log_ac):
        return 10.0 ** log_ac * (fs / self.pivot_frequency) ** alpha

    def compute_Sgw(self, fs, omegaf_args):
        H0 = 2.2e-18
        if isinstance(omegaf_args, (list, tuple)) and len(omegaf_args) == 2 and hasattr(omegaf_args[0], "__len__"):
            alpha = np.asarray(omegaf_args[0])
            log_ac = np.asarray(omegaf_args[1])
            omegaf = 10.0 ** log_ac[None, :] * (fs / self.pivot_frequency) ** alpha[None, :]
        else:
            alpha, log_ac = omegaf_args
            omegaf = self.omegaf(fs, alpha, log_ac)
        return omegaf * (3.0 / (4.0 * fs**3)) * (H0 / np.pi) ** 2

    def prior(self, unit_theta):
        alpha_bounds = self.params["prior_bounds"]["alpha"]
        log_ac_bounds = get_log_ac_prior_bounds(self.params)
        alpha = alpha_bounds[0] + unit_theta[0] * (alpha_bounds[1] - alpha_bounds[0])
        log_ac = log_ac_bounds[0] + unit_theta[1] * (log_ac_bounds[1] - log_ac_bounds[0])
        return [alpha, log_ac]


class MultipoleForecastModel:
    def __init__(self, params, submodel, data, sigma):
        self.params = params
        self.submodel_names = [params["model"]]
        self.submodels = {params["model"]: submodel}
        self.parameters = {
            params["model"]: submodel.parameters,
            "spectral": submodel.parameters,
            "spatial": [],
            "all": submodel.parameters,
        }
        self.Npar = submodel.Npar
        self.fs = submodel.fs
        self.data = data
        self.sigma = sigma

    def prior(self, unit_theta):
        return self.submodels[self.submodel_names[0]].prior(unit_theta)

    def likelihood(self, theta):
        alpha, log_ac = theta
        submodel = self.submodels[self.submodel_names[0]]
        model = submodel.omegaf(submodel.fs, alpha, log_ac)
        resid = self.data - model
        var = self.sigma ** 2
        logl = -0.5 * np.sum(resid ** 2 / var + np.log(2.0 * np.pi * var))
        return float(np.real(logl))


class MultipoleForecastInjection:
    def __init__(self, model_name, alpha_true, log_ac_true):
        self.component_names = [model_name]
        self.truevals = {
            model_name: {
                r"$\alpha$": alpha_true,
                r"$\log_{10}(A_c)$": log_ac_true,
            }
        }


class MultipoleForecastRun:
    def __init__(self, params, inj):
        if params.get("load_data", 0):
            raise ValueError("powerlaw_single_multipole is a forecast model and does not support load_data=1.")

        self.params = params
        self.inj = inj
        self.model_name = params["model"]
        self.multipole_l = int(params.get("multipole_l", 0))
        self.target_snr = float(params.get("target_snr", 1.0))
        self.bin_snr_target = float(params.get("bin_snr_target", 1.0))

        self.freqs, self.f0, self.tsegmid, self.delta_f = build_analysis_axes(params)
        self.context = MultipoleSensitivityContext(params, inj, self.freqs, self.f0, self.tsegmid)
        self.np_level, self.na_level = get_noise_levels(inj)
        self.noise_diag = get_noise_diagonal(self.context, self.freqs, self.f0, self.np_level, self.na_level)
        self.response_l = compute_multipole_response(self.context, self.f0, self.tsegmid, self.multipole_l)
        self.pair_curves, self.omega_total = compute_channel_sensitivities(self.freqs, self.response_l, self.noise_diag)
        self.ac_min_bin, self.ac_min_band, self.best_bin = compute_minimum_pivot_amplitudes(
            self.freqs,
            self.omega_total,
            params.get("forecast_alpha_default", 0.0),
            params["fref"],
            params["dur"],
            self.delta_f,
            self.target_snr,
            self.bin_snr_target,
        )

        component_truevals = inj.get("truevals", {}).get(self.model_name, {})
        self.alpha_true = float(component_truevals.get("alpha", params.get("forecast_alpha_default", 0.0)))
        if "log_A_c" in component_truevals:
            self.log10_ac_true = float(component_truevals["log_A_c"])
        elif "log_omega0" in component_truevals:
            self.log10_ac_true = float(component_truevals["log_omega0"])
        else:
            self.log10_ac_true = float(np.log10(self.ac_min_band))

        self.sigma = self.omega_total / np.sqrt(params["dur"] * self.delta_f)
        self.signal = 10.0 ** self.log10_ac_true * (self.freqs / params["fref"]) ** self.alpha_true
        if params.get("forecast_noisy", 0):
            rng = np.random.RandomState(params.get("seed", 0) + self.multipole_l)
            self.data = self.signal + rng.normal(scale=self.sigma, size=self.signal.size)
        else:
            self.data = self.signal.copy()

        submodel = MultipoleForecastSubmodel(
            params,
            self.freqs,
            self.f0,
            self.omega_total,
            self.response_l,
            self.multipole_l,
            params["fref"],
        )
        self.Model = MultipoleForecastModel(params, submodel, self.data, self.sigma)
        self.Injection = MultipoleForecastInjection(self.model_name, self.alpha_true, self.log10_ac_true)

    def write_sensitivity_products(self, out_dir):
        curves_path = os.path.join(out_dir, "multipole_sensitivity_curves.txt")
        summary_path = os.path.join(out_dir, "multipole_sensitivity_summary.txt")
        forecast_data_path = os.path.join(out_dir, "forecast_data.txt")
        snr_summary_path = os.path.join(out_dir, "snr_summary.txt")

        np.savetxt(
            curves_path,
            np.column_stack([self.freqs, self.omega_total, self.ac_min_bin]),
            header="freq_hz omega_n_L_h2 ac_min_bin_h2",
        )
        np.savetxt(
            summary_path,
            np.asarray(
                [
                    [
                        self.multipole_l,
                        self.best_bin["frequency_hz"],
                        self.best_bin["omega_n_h2"],
                        self.best_bin["ac_min_bin_h2"],
                        self.ac_min_band,
                        self.target_snr,
                        self.bin_snr_target,
                        self.params["dur"],
                        self.delta_f,
                        self.np_level,
                        self.na_level,
                        self.params["fref"],
                    ]
                ]
            ),
            header=(
                "multipole_l best_bin_frequency_hz best_bin_omega_n_h2 best_bin_ac_min_h2 "
                "ac_min_band_h2 target_snr bin_snr_target duration_s delta_f_hz noise_Np noise_Na pivot_frequency_hz"
            ),
        )
        np.savetxt(
            forecast_data_path,
            np.column_stack([self.freqs, self.data, self.sigma, self.signal]),
            header="freq_hz data sigma signal",
        )

        with open(snr_summary_path, "w") as outfile:
            outfile.write("multipole_l {}\n".format(self.multipole_l))
            outfile.write("target_band_snr {:.12e}\n".format(self.target_snr))
            outfile.write("target_bin_snr {:.12e}\n".format(self.bin_snr_target))
            outfile.write("pivot_frequency_hz {:.12e}\n".format(self.params["fref"]))
            outfile.write("duration_s {:.12e}\n".format(self.params["dur"]))
            outfile.write("delta_f_hz {:.12e}\n".format(self.delta_f))
            outfile.write("best_bin_frequency_hz {:.12e}\n".format(self.best_bin["frequency_hz"]))
            outfile.write("best_bin_omega_n_h2 {:.12e}\n".format(self.best_bin["omega_n_h2"]))
            outfile.write("best_bin_Ac_min_h2 {:.12e}\n".format(self.best_bin["ac_min_bin_h2"]))
            outfile.write("band_Ac_min_h2 {:.12e}\n".format(self.ac_min_band))
            outfile.write("band_log10_Ac_min {:.12e}\n".format(np.log10(self.ac_min_band)))

        fig1, ax1 = plt.subplots()
        ax1.plot(self.freqs, self.omega_total, color="black")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel(r"$\Omega_{{\rm GW},n}^{L}(f)\,h^2$")
        ax1.set_title("Single-L sensitivity kernel (L={})".format(self.multipole_l))
        fig1.tight_layout()
        fig1.savefig(os.path.join(out_dir, "multipole_omega_sensitivity.png"), dpi=200)
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(self.freqs, self.ac_min_bin, color="black")
        ax2.axhline(self.ac_min_band, color="tab:red", ls="--", lw=1.2)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel(r"$A_{{c,{\rm min}}}^{\rm bin}(f)\,h^2$")
        ax2.set_title(r"Minimum pivot amplitudes ($L={}$, $\alpha={:.3g}$)".format(self.multipole_l, self.alpha_true))
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "multipole_ac_min_bin.png"), dpi=200)
        plt.close(fig2)

        fig3, ax3 = plt.subplots()
        ax3.scatter([self.multipole_l], [self.ac_min_band], color="black")
        ax3.set_yscale("log")
        ax3.set_xlabel("Multipole L")
        ax3.set_ylabel(r"$A_{{c,{\rm min}}}^{\rm band}\,h^2$")
        ax3.set_title(r"Band-integrated threshold at SNR={:.3g}".format(self.target_snr))
        fig3.tight_layout()
        fig3.savefig(os.path.join(out_dir, "multipole_ac_min_band.png"), dpi=200)
        plt.close(fig3)

        print("Wrote single-multipole forecast products to {}".format(out_dir))
        print(
            "L = {:2d} | best-bin f = {:.6e} Hz | best-bin A_c,min h^2 = {:.6e} | band A_c,min h^2 = {:.6e}".format(
                self.multipole_l,
                self.best_bin["frequency_hz"],
                self.best_bin["ac_min_bin_h2"],
                self.ac_min_band,
            )
        )
        print("Sensitivity curves: {}".format(curves_path))
        print("Sensitivity summary: {}".format(summary_path))
        print("SNR summary: {}".format(snr_summary_path))
        print("Forecast data: {}".format(forecast_data_path))

    def run_grid_posterior(self, out_dir, grid_size=201, posterior_samples=20000):
        alpha_bounds = self.params["prior_bounds"]["alpha"]
        logac_bounds = get_log_ac_prior_bounds(self.params)
        alpha_grid = np.linspace(alpha_bounds[0], alpha_bounds[1], grid_size)
        logac_grid = np.linspace(logac_bounds[0], logac_bounds[1], grid_size)

        var = self.sigma ** 2
        log_norm = np.sum(np.log(2.0 * np.pi * var))
        loglike = np.empty((alpha_grid.size, logac_grid.size), dtype=float)
        amp_grid = 10.0 ** logac_grid
        scaled_freq = self.freqs / self.params["fref"]

        for alpha_idx, alpha_val in enumerate(alpha_grid):
            model_shape = scaled_freq ** alpha_val
            model_grid = amp_grid[:, None] * model_shape[None, :]
            resid = self.data[None, :] - model_grid
            loglike[alpha_idx, :] = -0.5 * np.sum(resid ** 2 / var[None, :], axis=1) - 0.5 * log_norm

        max_loglike = np.max(loglike)
        like_shifted = np.exp(loglike - max_loglike)
        dalpha = alpha_grid[1] - alpha_grid[0] if alpha_grid.size > 1 else 1.0
        dlogac = logac_grid[1] - logac_grid[0] if logac_grid.size > 1 else 1.0
        prior_area = (alpha_bounds[1] - alpha_bounds[0]) * (logac_bounds[1] - logac_bounds[0])
        logz = max_loglike + np.log(np.sum(like_shifted) * dalpha * dlogac) - np.log(prior_area)

        posterior_prob = like_shifted / np.sum(like_shifted)
        posterior_pdf = posterior_prob / (dalpha * dlogac)
        alpha_pdf = trapezoid_compat(posterior_pdf, logac_grid, axis=1)
        logac_pdf = trapezoid_compat(posterior_pdf, alpha_grid, axis=0)
        alpha_interval = density_interval(alpha_grid, alpha_pdf)
        logac_interval = density_interval(logac_grid, logac_pdf)

        post_samples = sample_grid_posterior(
            alpha_grid,
            logac_grid,
            posterior_prob,
            posterior_samples,
            self.params.get("seed", 0) + 2000 + self.multipole_l,
        )

        np.savetxt(os.path.join(out_dir, "post_samples.txt"), post_samples)
        np.savetxt(os.path.join(out_dir, "logz.txt"), np.atleast_1d(logz))
        np.savetxt(os.path.join(out_dir, "logzerr.txt"), np.atleast_1d(np.nan))
        np.savez(
            os.path.join(out_dir, "posterior_grid.npz"),
            alpha_grid=alpha_grid,
            log10_Ac_grid=logac_grid,
            posterior_pdf=posterior_pdf,
            alpha_pdf=alpha_pdf,
            log10_Ac_pdf=logac_pdf,
            loglike=loglike,
        )

        plot_grid_corner(
            out_dir,
            alpha_grid,
            logac_grid,
            posterior_pdf,
            alpha_pdf,
            logac_pdf,
            self.alpha_true,
            self.log10_ac_true,
            alpha_interval,
            logac_interval,
        )

        with open(os.path.join(out_dir, "posterior_summary.txt"), "w") as outfile:
            outfile.write("method grid\n")
            outfile.write("multipole_l {}\n".format(self.multipole_l))
            outfile.write("target_snr {:.12e}\n".format(self.target_snr))
            outfile.write("alpha_true {:.12e}\n".format(self.alpha_true))
            outfile.write("log10_Ac_true {:.12e}\n".format(self.log10_ac_true))
            outfile.write("threshold_log10_Ac {:.12e}\n".format(np.log10(self.ac_min_band)))
            outfile.write("alpha_q025 {:.12e}\n".format(alpha_interval[0]))
            outfile.write("alpha_median {:.12e}\n".format(alpha_interval[1]))
            outfile.write("alpha_q975 {:.12e}\n".format(alpha_interval[2]))
            outfile.write("log10_Ac_q025 {:.12e}\n".format(logac_interval[0]))
            outfile.write("log10_Ac_median {:.12e}\n".format(logac_interval[1]))
            outfile.write("log10_Ac_q975 {:.12e}\n".format(logac_interval[2]))
            outfile.write("logz {:.12e}\n".format(float(logz)))
            outfile.write("logzerr nan\n")

        print("Posterior summary: {}".format(os.path.join(out_dir, "posterior_summary.txt")))
        print("Corner plot: {}".format(os.path.join(out_dir, "corners.png")))
        print("Posterior samples: {}".format(os.path.join(out_dir, "post_samples.txt")))

        return {
            "multipole_l": self.multipole_l,
            "logz": float(logz),
            "logzerr": np.nan,
            "alpha_q025": alpha_interval[0],
            "alpha_median": alpha_interval[1],
            "alpha_q975": alpha_interval[2],
            "log10_Ac_q025": logac_interval[0],
            "log10_Ac_median": logac_interval[1],
            "log10_Ac_q975": logac_interval[2],
            "alpha_true": self.alpha_true,
            "log10_Ac_true": self.log10_ac_true,
        }
