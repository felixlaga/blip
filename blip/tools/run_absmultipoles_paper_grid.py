#!/usr/bin/env python3

import argparse
import ast
import configparser
import copy
import logging
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from blip.src.utils import catch_color_duplicates, log_manager

RUN_BLIP_PATH = REPO_ROOT / "blip" / "run_blip"


class CompositeInjection:
    '''
    Lightweight injection container for composite absolute-amplitude datasets.

    The shared dataset is assembled from separately generated BLIP component runs, but
    the downstream recovery plotting expects an Injection-like object in
    shared_data/injection.pickle. This class preserves the small subset of the Injection
    API that the plotting utilities use.
    '''

    def __init__(self, params, frange, components, truevals):
        self.params = copy.deepcopy(params)
        self.frange = np.asarray(frange, dtype=float)
        self.components = components
        self.component_names = list(components.keys())
        self.sgwb_component_names = [name for name in self.component_names if name != 'noise']
        self.truevals = truevals
        catch_color_duplicates(self)

    def compute_convolved_spectra(self, component_name, fs_new=None, channels='11', return_fs=False, imaginary=False):
        cm = self.components[component_name]
        c1_idx, c2_idx = int(channels[0]) - 1, int(channels[1]) - 1

        if not imaginary:
            PSD = np.abs(np.real(cm.frozen_convolved_spectra[c1_idx, c2_idx, :]))
        else:
            PSD = 1j * np.abs(np.imag(cm.frozen_convolved_spectra[c1_idx, c2_idx, :]))

        fs = self.frange
        if fs_new is not None:
            with log_manager(logging.ERROR):
                PSD_interp = interp1d(fs, np.log10(PSD))
                PSD = 10 ** PSD_interp(fs_new)
                fs = fs_new

        if return_fs:
            return fs, PSD
        return PSD

    def plot_injected_spectra(
        self,
        component_name,
        fs_new=None,
        ax=None,
        convolved=False,
        legend=False,
        channels='11',
        return_PSD=False,
        scale='log',
        flim=None,
        ymins=None,
        **plt_kwargs,
    ):
        cm = self.components[component_name]

        if ax is None:
            ax = plt.gca()

        if flim is not None:
            fmin, fmax = flim
        else:
            fmin, fmax = self.params['fmin'], self.params['fmax']

        if convolved:
            if component_name == 'noise':
                raise ValueError(
                    "Cannot convolve noise spectra with the detector GW response - this is not physical. "
                    "(Set convolved=False in the function call!)"
                )
            fs, PSD = self.compute_convolved_spectra(component_name, channels=channels, return_fs=True, fs_new=fs_new)
        else:
            PSD = cm.frozen_spectra
            if (len(PSD.shape) == 3) and (PSD.shape[0] == PSD.shape[1] == 3):
                i_idx, j_idx = int(channels[0]) - 1, int(channels[1]) - 1
                PSD = PSD[i_idx, j_idx, :]

            if fs_new is not None:
                with log_manager(logging.ERROR):
                    PSD_interp = interp1d(self.frange, np.log10(PSD))
                    PSD = 10 ** PSD_interp(fs_new)
                    fs = fs_new
            else:
                fs = self.frange

        filt = (fs > fmin) * (fs < fmax)

        if legend and ('label' not in plt_kwargs):
            plt_kwargs['label'] = cm.fancyname

        if scale == 'log':
            ax.loglog(fs[filt], PSD[filt], **plt_kwargs)
        elif scale == 'linear':
            ax.plot(fs[filt], PSD[filt], **plt_kwargs)
        else:
            raise ValueError("We only support linear and log plots, there is no secret third option!")

        if ymins is not None:
            ymins.append(PSD.min())

        if return_PSD:
            return PSD
        return None


# Make composite shared-data injections pickle-compatible across later BLIP runs.
CompositeInjection.__module__ = "blip.tools.run_absmultipoles_paper_grid"


def load_config(path):
    config = configparser.ConfigParser()
    read_files = config.read(path)
    if len(read_files) == 0:
        raise FileNotFoundError("Could not read config '{}'.".format(path))
    return config


def parse_literal(config, section, option, fallback=None):
    if not config.has_section(section) or not config.has_option(section, option):
        return fallback
    return ast.literal_eval(str(config.get(section, option)))


def parse_integer_sequence(config, section, option, fallback=None):
    values = parse_literal(config, section, option, fallback=fallback)
    if values is None:
        return fallback
    if isinstance(values, (int, np.integer)):
        return [int(values)]
    return [int(value) for value in values]


def parse_float_sequence(config, section, option, fallback=None):
    values = parse_literal(config, section, option, fallback=fallback)
    if values is None:
        return fallback
    if isinstance(values, (int, float, np.integer, np.floating)):
        return [float(values)]
    return [float(value) for value in values]


def parse_bool_option(config, section, option, fallback=False):
    if not config.has_section(section) or not config.has_option(section, option):
        return bool(fallback)
    return bool(int(config.get(section, option)))


def set_option(config, section, option, value):
    if not config.has_section(section):
        config.add_section(section)
    config.set(section, option, str(value))


def remove_option(config, section, option):
    if config.has_section(section) and config.has_option(section, option):
        config.remove_option(section, option)


def enable_full_plot_products(config):
    set_option(config, "run_params", "skip_diagnostics", 0)
    set_option(config, "run_params", "skip_postprocessing", 0)


def write_config(config, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as outfile:
        config.write(outfile)


def resolve_out_dir(config, config_path):
    out_dir = config.get("run_params", "out_dir")
    out_path = Path(out_dir)
    if out_path.is_absolute():
        return out_path
    return (config_path.parent / out_path).resolve()


def run_blip(config_path):
    subprocess.run(
        [sys.executable, str(RUN_BLIP_PATH), str(config_path.resolve())],
        cwd=str(REPO_ROOT),
        check=True,
    )


def read_logz(run_dir):
    logz = np.loadtxt(run_dir / "logz.txt")
    logz = np.asarray(logz, dtype=float).reshape(-1)
    if logz.size == 0:
        raise ValueError("No log-evidence samples found in '{}'.".format(run_dir))
    return float(logz[-1])


def find_absmultipoles_component_name(config):
    injection_components = config.get("inj", "injection").split("+")
    matches = [
        component
        for component in injection_components
        if component.split("-")[0].split("_")[-1] == "absmultipoles"
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one '*_absmultipoles' injection component, found {} in '{}'.".format(
                len(matches),
                config.get("inj", "injection"),
            )
        )
    return matches[0]


def build_analysis_aliases(model_string, injection_component_name):
    aliases = {}
    for component_name in model_string.split("+"):
        if component_name == "noise":
            continue
        aliases[component_name] = injection_component_name
    return aliases


def format_amplitude_label(amplitude):
    mantissa, exponent = "{:.3e}".format(float(amplitude)).split("e")
    mantissa_slug = mantissa.replace(".", "p").replace("-", "m")
    exponent_slug = exponent.replace("+", "p").replace("-", "m")
    return "A2_{}e{}".format(mantissa_slug, exponent_slug)


def normalize_absolute_amplitude_dict(component_amplitudes, multipole_ls):
    if "log_A_L_by_L" in component_amplitudes:
        log_lookup = {int(key): float(value) for key, value in component_amplitudes["log_A_L_by_L"].items()}
        return {multipole_l: float(10 ** log_lookup[multipole_l]) for multipole_l in multipole_ls}

    if "A_L_by_L" in component_amplitudes:
        amp_lookup = {int(key): float(value) for key, value in component_amplitudes["A_L_by_L"].items()}
        return {multipole_l: amp_lookup[multipole_l] for multipole_l in multipole_ls}

    raise ValueError(
        "Absolute paper-grid configs must provide [absolute_injection] A_L_by_L or log_A_L_by_L."
    )


def normalize_absolute_log_amplitude_dict(component_truevals, multipole_ls):
    if "log_A_L_by_L" in component_truevals:
        amplitude_dict = component_truevals["log_A_L_by_L"]
        return {int(key): float(value) for key, value in amplitude_dict.items()}

    if "A_L_by_L" in component_truevals:
        amplitude_dict = component_truevals["A_L_by_L"]
        return {int(key): float(np.log10(value)) for key, value in amplitude_dict.items()}

    if "log_A_Ls" in component_truevals:
        values = np.asarray(component_truevals["log_A_Ls"], dtype=float).reshape(-1)
        if values.size != len(multipole_ls):
            raise ValueError(
                "Configured 'log_A_Ls' has length {}, expected {} from multipole_ls."
                .format(values.size, len(multipole_ls))
            )
        return {int(multipole_l): float(value) for multipole_l, value in zip(multipole_ls, values)}

    if "A_Ls" in component_truevals:
        values = np.asarray(component_truevals["A_Ls"], dtype=float).reshape(-1)
        if values.size != len(multipole_ls):
            raise ValueError(
                "Configured 'A_Ls' has length {}, expected {} from multipole_ls."
                .format(values.size, len(multipole_ls))
            )
        if np.any(values <= 0):
            raise ValueError("Injected shared-spectrum absolute multipole A_L values must all be > 0.")
        return {int(multipole_l): float(np.log10(value)) for multipole_l, value in zip(multipole_ls, values)}

    if len(multipole_ls) == 1 and ("log_A_L" in component_truevals or "A_L" in component_truevals):
        if "log_A_L" in component_truevals:
            return {int(multipole_ls[0]): float(component_truevals["log_A_L"])}
        amplitude = float(component_truevals["A_L"])
        if amplitude <= 0:
            raise ValueError("Injected single absolute multipole A_L must be > 0.")
        return {int(multipole_ls[0]): float(np.log10(amplitude))}

    amplitude_dict = {}
    for multipole_l in multipole_ls:
        log_key = "log_A_L_{}".format(multipole_l)
        linear_key = "A_L_{}".format(multipole_l)
        if log_key in component_truevals:
            amplitude_dict[int(multipole_l)] = float(component_truevals[log_key])
        elif linear_key in component_truevals:
            linear_value = float(component_truevals[linear_key])
            if linear_value <= 0:
                raise ValueError("Injected shared-spectrum absolute multipole A_L values must all be > 0.")
            amplitude_dict[int(multipole_l)] = float(np.log10(linear_value))

    if len(amplitude_dict) == len(multipole_ls):
        return amplitude_dict

    raise ValueError(
        "The absmultipoles injection truevals must provide one of 'log_A_L_by_L', 'A_L_by_L', "
        "'log_A_Ls', 'A_Ls', or per-L keys such as 'log_A_L_2'."
    )


def build_dataset_specs(full_multipole_ls, target_multipole_l, absolute_a_l_grid, include_zero_baseline, seed_start):
    nuisance_multipole_ls = [multipole_l for multipole_l in full_multipole_ls if multipole_l != target_multipole_l]
    specs = []
    dataset_index = 0

    if include_zero_baseline:
        specs.append(
            {
                "dataset_label": "baseline_no_L2",
                "target_a_l": 0.0,
                "seed": seed_start + dataset_index,
                "active_multipole_ls": nuisance_multipole_ls,
            }
        )
        dataset_index += 1

    for amplitude in absolute_a_l_grid:
        specs.append(
            {
                "dataset_label": format_amplitude_label(amplitude),
                "target_a_l": float(amplitude),
                "seed": seed_start + dataset_index,
                "active_multipole_ls": list(full_multipole_ls),
            }
        )
        dataset_index += 1

    return specs


def run_full_dataset_absolute_grid(base_config_path, base_config, output_root, args):
    if not base_config.has_section("paper_grid"):
        raise ValueError("Absolute paper-grid configs must define a [paper_grid] section.")

    target_multipole_l = int(base_config.get("paper_grid", "target_multipole_l"))
    absolute_a_l_grid = parse_float_sequence(base_config, "paper_grid", "A_L_grid", fallback=[])
    include_zero_baseline = bool(int(base_config.get("paper_grid", "include_zero_baseline", fallback="1")))
    seed_start = int(base_config.get("paper_grid", "seed_start", fallback="200"))
    final_dataset_full_plots = parse_bool_option(
        base_config,
        "paper_grid",
        "final_dataset_full_plots",
        fallback=False,
    )

    if len(absolute_a_l_grid) == 0 and not include_zero_baseline:
        raise ValueError("The absolute paper grid must include at least one injected dataset.")
    if np.any(np.asarray(absolute_a_l_grid, dtype=float) <= 0):
        raise ValueError("All configured A_L_grid values must be > 0.")

    injection_component_name = find_absmultipoles_component_name(base_config)
    injection_multipole_ls = parse_integer_sequence(base_config, "inj", "multipole_ls", fallback=None)
    if injection_multipole_ls is None:
        injection_multipole_ls = parse_integer_sequence(base_config, "params", "multipole_ls", fallback=None)
    if injection_multipole_ls is None or len(injection_multipole_ls) == 0:
        raise ValueError("Paper-grid absmultipoles injections require [inj] multipole_ls or [params] multipole_ls.")
    if target_multipole_l not in injection_multipole_ls:
        raise ValueError(
            "Target multipole L={} is not present in the injected multipole list {}.".format(
                target_multipole_l,
                injection_multipole_ls,
            )
        )

    raw_truevals = parse_literal(base_config, "inj", "truevals", fallback=None)
    if raw_truevals is None:
        raise ValueError("Absolute paper-grid configs must provide [inj] truevals.")
    component_truevals = copy.deepcopy(raw_truevals[injection_component_name])
    if "omega0" not in component_truevals:
        raise ValueError("The absmultipoles injection component must specify an isotropic amplitude 'omega0'.")
    base_log_a_by_l = normalize_absolute_log_amplitude_dict(component_truevals, injection_multipole_ls)
    for multipole_l in injection_multipole_ls:
        if multipole_l not in base_log_a_by_l:
            raise ValueError("Missing injected A_L value for L={}.".format(multipole_l))

    dataset_specs = build_dataset_specs(
        injection_multipole_ls,
        target_multipole_l,
        absolute_a_l_grid,
        include_zero_baseline,
        seed_start,
    )

    summary_rows = []
    input_spectrum_name = Path(base_config.get("run_params", "input_spectrum", fallback="data_spectrum.npz")).name

    last_dataset_label = dataset_specs[-1]["dataset_label"]

    for dataset in dataset_specs:
        dataset_label = dataset["dataset_label"]
        injected_a2 = float(dataset["target_a_l"])
        is_final_dataset = dataset_label == last_dataset_label
        dataset_root = output_root / dataset_label
        shared_dir = dataset_root / "shared_data"
        null_dir = dataset_root / "null_model"
        alt_dir = dataset_root / "alt_model"
        shared_spectrum_path = shared_dir / input_spectrum_name

        shared_config = copy.deepcopy(base_config)
        shared_truevals = copy.deepcopy(raw_truevals)
        shared_component_truevals = copy.deepcopy(component_truevals)
        active_multipole_ls = list(dataset["active_multipole_ls"])
        active_log_a_by_l = {
            int(multipole_l): float(base_log_a_by_l[multipole_l])
            for multipole_l in active_multipole_ls
        }
        if target_multipole_l in active_log_a_by_l:
            active_log_a_by_l[target_multipole_l] = float(np.log10(injected_a2))

        if len(active_multipole_ls) == 0:
            isotropic_truevals = copy.deepcopy(component_truevals)
            for multipole_key in [
                "log_A_L",
                "A_L",
                "log_A_Ls",
                "A_Ls",
                "log_A_L_by_L",
                "A_L_by_L",
            ]:
                isotropic_truevals.pop(multipole_key, None)
            shared_truevals = {
                "noise": copy.deepcopy(raw_truevals["noise"]),
                "powerlaw_isgwb": isotropic_truevals,
            }

            set_option(shared_config, "params", "model", "noise+powerlaw_isgwb")
            set_option(shared_config, "params", "alias", repr({}))
            set_option(shared_config, "params", "load_data", 0)
            set_option(shared_config, "params", "lmax", target_multipole_l)
            remove_option(shared_config, "params", "multipole_l")
            remove_option(shared_config, "params", "multipole_ls")
            set_option(shared_config, "inj", "doInj", 1)
            set_option(shared_config, "inj", "injection", "noise+powerlaw_isgwb")
            remove_option(shared_config, "inj", "inj_lmax")
            remove_option(shared_config, "inj", "multipole_l")
            remove_option(shared_config, "inj", "multipole_ls")
            set_option(shared_config, "inj", "truevals", repr(shared_truevals))
        else:
            shared_component_truevals["log_A_L_by_L"] = active_log_a_by_l
            shared_component_truevals.pop("A_L_by_L", None)
            shared_component_truevals.pop("log_A_Ls", None)
            shared_component_truevals.pop("A_Ls", None)
            shared_truevals[injection_component_name] = shared_component_truevals

            shared_lmax = max(active_multipole_ls)

            set_option(shared_config, "params", "model", "noise+powerlaw_absmultipoles")
            set_option(shared_config, "params", "alias", repr({"powerlaw_absmultipoles": injection_component_name}))
            set_option(shared_config, "params", "load_data", 0)
            set_option(shared_config, "params", "lmax", shared_lmax)
            set_option(shared_config, "params", "multipole_ls", repr(active_multipole_ls))
            set_option(shared_config, "inj", "doInj", 1)
            set_option(shared_config, "inj", "injection", "noise+powerlaw_absmultipoles")
            set_option(shared_config, "inj", "inj_lmax", shared_lmax)
            set_option(shared_config, "inj", "multipole_ls", repr(active_multipole_ls))
            set_option(shared_config, "inj", "truevals", repr(shared_truevals))
        set_option(shared_config, "run_params", "FixSeed", 1)
        set_option(shared_config, "run_params", "seed", dataset["seed"])
        set_option(shared_config, "run_params", "generate_only", 1)
        set_option(shared_config, "run_params", "doPreProc", 1)
        set_option(shared_config, "run_params", "out_dir", str(shared_dir))
        set_option(shared_config, "run_params", "input_spectrum", input_spectrum_name)
        if final_dataset_full_plots and is_final_dataset:
            enable_full_plot_products(shared_config)
        shared_config_path = dataset_root / "shared_data.ini"
        write_config(shared_config, shared_config_path)

        if not args.prepare_only:
            if (not args.reuse_existing) or (not shared_spectrum_path.exists()):
                print("Generating shared dataset '{}' in {}".format(dataset_label, shared_dir))
                run_blip(shared_config_path)

        null_config = copy.deepcopy(base_config)
        set_option(null_config, "params", "model", "noise+powerlaw_isgwb")
        set_option(null_config, "params", "alias", repr({"powerlaw_isgwb": injection_component_name}))
        set_option(null_config, "params", "load_data", 1)
        remove_option(null_config, "params", "multipole_l")
        remove_option(null_config, "params", "multipole_ls")
        set_option(null_config, "inj", "doInj", 0)
        remove_option(null_config, "inj", "injection")
        remove_option(null_config, "inj", "inj_lmax")
        remove_option(null_config, "inj", "multipole_l")
        remove_option(null_config, "inj", "multipole_ls")
        remove_option(null_config, "inj", "truevals")
        set_option(null_config, "run_params", "FixSeed", 1)
        set_option(null_config, "run_params", "seed", dataset["seed"])
        set_option(null_config, "run_params", "generate_only", 0)
        set_option(null_config, "run_params", "doPreProc", 0)
        set_option(null_config, "run_params", "out_dir", str(null_dir))
        set_option(null_config, "run_params", "input_spectrum", str(shared_spectrum_path.resolve()))
        if final_dataset_full_plots and is_final_dataset:
            enable_full_plot_products(null_config)
        null_config_path = dataset_root / "null_model.ini"
        write_config(null_config, null_config_path)

        alt_config = copy.deepcopy(base_config)
        set_option(alt_config, "params", "model", "noise+powerlaw_absmultipoles")
        set_option(
            alt_config,
            "params",
            "alias",
            repr(build_analysis_aliases("noise+powerlaw_absmultipoles", injection_component_name)),
        )
        set_option(alt_config, "params", "load_data", 1)
        set_option(alt_config, "params", "lmax", target_multipole_l)
        set_option(alt_config, "params", "multipole_ls", repr([target_multipole_l]))
        set_option(alt_config, "inj", "doInj", 0)
        remove_option(alt_config, "inj", "injection")
        remove_option(alt_config, "inj", "inj_lmax")
        remove_option(alt_config, "inj", "multipole_l")
        remove_option(alt_config, "inj", "multipole_ls")
        remove_option(alt_config, "inj", "truevals")
        set_option(alt_config, "run_params", "FixSeed", 1)
        set_option(alt_config, "run_params", "seed", dataset["seed"])
        set_option(alt_config, "run_params", "generate_only", 0)
        set_option(alt_config, "run_params", "doPreProc", 0)
        set_option(alt_config, "run_params", "out_dir", str(alt_dir))
        set_option(alt_config, "run_params", "input_spectrum", str(shared_spectrum_path.resolve()))
        if final_dataset_full_plots and is_final_dataset:
            enable_full_plot_products(alt_config)
        alt_config_path = dataset_root / "alt_model.ini"
        write_config(alt_config, alt_config_path)

        if args.prepare_only:
            continue

        if (not args.reuse_existing) or (not (null_dir / "logz.txt").exists()):
            print("Running absolute null model for '{}' in {}".format(dataset_label, null_dir))
            run_blip(null_config_path)
        if (not args.reuse_existing) or (not (alt_dir / "logz.txt").exists()):
            print("Running absolute alt model for '{}' in {}".format(dataset_label, alt_dir))
            run_blip(alt_config_path)

        null_logz = read_logz(null_dir)
        alt_logz = read_logz(alt_dir)
        summary_rows.append(
            {
                "dataset_label": dataset_label,
                "injected_A2": injected_a2,
                "null_logz": null_logz,
                "alt_logz": alt_logz,
                "delta_logz": alt_logz - null_logz,
                "shared_dir": str(shared_dir.resolve()),
                "null_dir": str(null_dir.resolve()),
                "alt_dir": str(alt_dir.resolve()),
            }
        )

    if not args.prepare_only:
        write_summary_files(output_root, summary_rows)
        print("Finished native absolute paper-grid study in {}".format(output_root))


def ensure_matching_frequency_grid(reference_fdata, candidate_fdata, context):
    reference_fdata = np.asarray(reference_fdata, dtype=float)
    candidate_fdata = np.asarray(candidate_fdata, dtype=float)
    if reference_fdata.shape != candidate_fdata.shape or not np.allclose(reference_fdata, candidate_fdata):
        raise ValueError("Frequency grid mismatch while composing '{}': {}.".format(context, context))


def load_spectrum(path):
    data = np.load(path)
    return data["r1"], data["r2"], data["r3"], data["fdata"]


def write_summary_files(out_dir, rows):
    summary_path = out_dir / "bayes_factor_summary.txt"
    with open(summary_path, "w") as outfile:
        outfile.write(
            "# dataset_label injected_A2 null_logz alt_logz delta_logz shared_dir null_dir alt_dir\n"
        )
        for row in rows:
            outfile.write(
                "{} {:.16e} {:.16e} {:.16e} {:.16e} {} {} {}\n".format(
                    row["dataset_label"],
                    row["injected_A2"],
                    row["null_logz"],
                    row["alt_logz"],
                    row["delta_logz"],
                    row["shared_dir"],
                    row["null_dir"],
                    row["alt_dir"],
                )
            )

    labels = [row["dataset_label"] for row in rows]
    delta_logz = [row["delta_logz"] for row in rows]
    x_vals = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x_vals, delta_logz, color="royalblue", alpha=0.85)
    ax.axhline(0.0, color="0.5", lw=1, ls="--")
    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"$\Delta \log Z$")
    ax.set_title(r"Absolute Grid: $\log Z_{\mathrm{alt}} - \log Z_{\mathrm{null}}$")
    fig.tight_layout()
    fig.savefig(out_dir / "bayes_factor_summary.png", dpi=200)
    plt.close(fig)


def build_noise_only_config(base_config, noise_truevals, out_dir, seed, spectrum_name):
    config = copy.deepcopy(base_config)
    set_option(config, "params", "model", "noise")
    set_option(config, "params", "alias", repr({}))
    set_option(config, "params", "load_data", 0)
    remove_option(config, "params", "multipole_l")
    remove_option(config, "params", "multipole_ls")

    set_option(config, "inj", "doInj", 1)
    set_option(config, "inj", "injection", "noise")
    remove_option(config, "inj", "multipole_l")
    remove_option(config, "inj", "multipole_ls")
    remove_option(config, "inj", "inj_lmax")
    set_option(config, "inj", "truevals", repr({"noise": copy.deepcopy(noise_truevals)}))

    set_option(config, "run_params", "FixSeed", 1)
    set_option(config, "run_params", "seed", seed)
    set_option(config, "run_params", "generate_only", 1)
    set_option(config, "run_params", "doPreProc", 1)
    set_option(config, "run_params", "out_dir", str(out_dir))
    set_option(config, "run_params", "input_spectrum", spectrum_name)
    return config


def build_noise_plus_isotropic_config(base_config, noise_truevals, iso_truevals, out_dir, seed, spectrum_name):
    config = copy.deepcopy(base_config)
    set_option(config, "params", "model", "noise+powerlaw_isgwb")
    set_option(config, "params", "alias", repr({"powerlaw_isgwb": "powerlaw_isgwb"}))
    set_option(config, "params", "load_data", 0)
    remove_option(config, "params", "multipole_l")
    remove_option(config, "params", "multipole_ls")

    set_option(config, "inj", "doInj", 1)
    set_option(config, "inj", "injection", "noise+powerlaw_isgwb")
    remove_option(config, "inj", "multipole_l")
    remove_option(config, "inj", "multipole_ls")
    remove_option(config, "inj", "inj_lmax")
    set_option(
        config,
        "inj",
        "truevals",
        repr(
            {
                "noise": copy.deepcopy(noise_truevals),
                "powerlaw_isgwb": copy.deepcopy(iso_truevals),
            }
        ),
    )

    set_option(config, "run_params", "FixSeed", 1)
    set_option(config, "run_params", "seed", seed)
    set_option(config, "run_params", "generate_only", 1)
    set_option(config, "run_params", "doPreProc", 1)
    set_option(config, "run_params", "out_dir", str(out_dir))
    set_option(config, "run_params", "input_spectrum", spectrum_name)
    return config


def build_noise_plus_multipole_config(
    base_config,
    noise_truevals,
    absolute_alpha,
    multipole_l,
    multipole_amplitude,
    out_dir,
    seed,
    spectrum_name,
):
    config = copy.deepcopy(base_config)
    set_option(config, "params", "model", "noise+powerlaw_multipole")
    set_option(config, "params", "alias", repr({"powerlaw_multipole": "powerlaw_multipole"}))
    set_option(config, "params", "load_data", 0)
    set_option(config, "params", "multipole_l", multipole_l)
    remove_option(config, "params", "multipole_ls")

    set_option(config, "inj", "doInj", 1)
    set_option(config, "inj", "injection", "noise+powerlaw_multipole")
    set_option(config, "inj", "multipole_l", multipole_l)
    set_option(config, "inj", "inj_lmax", 0)
    remove_option(config, "inj", "multipole_ls")
    set_option(
        config,
        "inj",
        "truevals",
        repr(
            {
                "noise": copy.deepcopy(noise_truevals),
                "powerlaw_multipole": {"alpha": float(absolute_alpha), "A_L": float(multipole_amplitude)},
            }
        ),
    )

    set_option(config, "run_params", "FixSeed", 1)
    set_option(config, "run_params", "seed", seed)
    set_option(config, "run_params", "generate_only", 1)
    set_option(config, "run_params", "doPreProc", 1)
    set_option(config, "run_params", "out_dir", str(out_dir))
    set_option(config, "run_params", "input_spectrum", spectrum_name)
    return config


def build_null_config(base_config, shared_spectrum_path, out_dir, seed):
    config = copy.deepcopy(base_config)
    set_option(config, "params", "model", "noise+powerlaw_isgwb")
    set_option(config, "params", "alias", repr({"powerlaw_isgwb": "powerlaw_isgwb"}))
    set_option(config, "params", "load_data", 1)
    remove_option(config, "params", "multipole_l")
    remove_option(config, "params", "multipole_ls")

    set_option(config, "inj", "doInj", 0)
    remove_option(config, "inj", "injection")
    remove_option(config, "inj", "multipole_l")
    remove_option(config, "inj", "multipole_ls")
    remove_option(config, "inj", "inj_lmax")
    remove_option(config, "inj", "truevals")

    set_option(config, "run_params", "FixSeed", 1)
    set_option(config, "run_params", "seed", seed)
    set_option(config, "run_params", "generate_only", 0)
    set_option(config, "run_params", "doPreProc", 0)
    set_option(config, "run_params", "out_dir", str(out_dir))
    set_option(config, "run_params", "input_spectrum", str(shared_spectrum_path.resolve()))
    return config


def build_alt_config(base_config, target_multipole_l, shared_spectrum_path, out_dir, seed):
    config = copy.deepcopy(base_config)
    set_option(config, "params", "model", "noise+powerlaw_isgwb+powerlaw_multipole")
    set_option(
        config,
        "params",
        "alias",
        repr(
            {
                "powerlaw_isgwb": "powerlaw_isgwb",
                "powerlaw_multipole": "powerlaw_multipole_L{}".format(target_multipole_l),
            }
        ),
    )
    set_option(config, "params", "load_data", 1)
    set_option(config, "params", "multipole_l", target_multipole_l)
    remove_option(config, "params", "multipole_ls")

    set_option(config, "inj", "doInj", 0)
    remove_option(config, "inj", "injection")
    remove_option(config, "inj", "multipole_l")
    remove_option(config, "inj", "multipole_ls")
    remove_option(config, "inj", "inj_lmax")
    remove_option(config, "inj", "truevals")

    set_option(config, "run_params", "FixSeed", 1)
    set_option(config, "run_params", "seed", seed)
    set_option(config, "run_params", "generate_only", 0)
    set_option(config, "run_params", "doPreProc", 0)
    set_option(config, "run_params", "out_dir", str(out_dir))
    set_option(config, "run_params", "input_spectrum", str(shared_spectrum_path.resolve()))
    return config


def build_component_seed(base_seed, component_index, seed_stride):
    return int(base_seed + seed_stride * component_index)


def compose_shared_spectrum(base_spectrum_path, component_specs, output_spectrum_path):
    base_r1, base_r2, base_r3, base_fdata = load_spectrum(base_spectrum_path)
    total_r1 = np.asarray(base_r1, dtype=np.complex128)
    total_r2 = np.asarray(base_r2, dtype=np.complex128)
    total_r3 = np.asarray(base_r3, dtype=np.complex128)
    fdata = np.asarray(base_fdata, dtype=float)

    for component_spec in component_specs:
        noise_r1, noise_r2, noise_r3, noise_fdata = load_spectrum(component_spec["noise_only_spectrum"])
        signal_r1, signal_r2, signal_r3, signal_fdata = load_spectrum(component_spec["noise_plus_signal_spectrum"])
        ensure_matching_frequency_grid(fdata, noise_fdata, str(component_spec["noise_only_spectrum"]))
        ensure_matching_frequency_grid(fdata, signal_fdata, str(component_spec["noise_plus_signal_spectrum"]))
        total_r1 += np.asarray(signal_r1, dtype=np.complex128) - np.asarray(noise_r1, dtype=np.complex128)
        total_r2 += np.asarray(signal_r2, dtype=np.complex128) - np.asarray(noise_r2, dtype=np.complex128)
        total_r3 += np.asarray(signal_r3, dtype=np.complex128) - np.asarray(noise_r3, dtype=np.complex128)

    output_spectrum_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_spectrum_path,
        r1=total_r1,
        r2=total_r2,
        r3=total_r3,
        fdata=fdata,
    )


def build_composite_injection(base_injection_path, multipole_component_specs, output_path):
    with open(base_injection_path, "rb") as infile:
        base_injection = pickle.load(infile)

    components = {
        "noise": copy.deepcopy(base_injection.components["noise"]),
        "powerlaw_isgwb": copy.deepcopy(base_injection.components["powerlaw_isgwb"]),
    }
    truevals = {
        "noise": copy.deepcopy(base_injection.truevals["noise"]),
        "powerlaw_isgwb": copy.deepcopy(base_injection.truevals["powerlaw_isgwb"]),
    }

    for component_spec in multipole_component_specs:
        with open(component_spec["noise_plus_signal_injection"], "rb") as infile:
            signal_injection = pickle.load(infile)
        component_name = "powerlaw_multipole_L{}".format(component_spec["multipole_l"])
        component = copy.deepcopy(signal_injection.components["powerlaw_multipole"])
        component.name = component_name
        components[component_name] = component
        truevals[component_name] = copy.deepcopy(signal_injection.truevals["powerlaw_multipole"])

    composite_injection = CompositeInjection(base_injection.params, base_injection.frange, components, truevals)
    with open(output_path, "wb") as outfile:
        pickle.dump(composite_injection, outfile)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a paper-style absolute-amplitude multipole grid with matched null/alt recoveries."
    )
    parser.add_argument(
        "base_config",
        nargs="?",
        default=str(REPO_ROOT / "paper_absmultipoles_l2_grid_paper.ini"),
        help="Absolute paper-grid base config.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the generated configs; do not launch BLIP.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing shared-data component runs and recovery results when present.",
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config).resolve()
    base_config = load_config(str(base_config_path))
    output_root = resolve_out_dir(base_config, base_config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_config_path, output_root / base_config_path.name)

    workflow_mode = str(base_config.get("paper_grid", "workflow_mode", fallback="legacy_component")).strip().lower()
    if workflow_mode in ["full_dataset", "native_full_dataset", "native", "simple"]:
        run_full_dataset_absolute_grid(base_config_path, base_config, output_root, args)
        return

    if not base_config.has_section("paper_grid"):
        raise ValueError("Absolute paper-grid configs must define a [paper_grid] section.")
    if not base_config.has_section("absolute_injection"):
        raise ValueError("Absolute paper-grid configs must define an [absolute_injection] section.")

    target_multipole_l = int(base_config.get("paper_grid", "target_multipole_l"))
    absolute_a_l_grid = parse_float_sequence(base_config, "paper_grid", "A_L_grid", fallback=[])
    include_zero_baseline = bool(int(base_config.get("paper_grid", "include_zero_baseline", fallback="1")))
    seed_start = int(base_config.get("paper_grid", "seed_start", fallback="200"))
    seed_stride = int(base_config.get("absolute_injection", "component_seed_stride", fallback="100"))

    if len(absolute_a_l_grid) == 0 and not include_zero_baseline:
        raise ValueError("The absolute paper grid must include at least one injected dataset.")
    if np.any(np.asarray(absolute_a_l_grid, dtype=float) <= 0):
        raise ValueError("All configured A_L_grid values must be > 0.")

    injection_multipole_ls = parse_integer_sequence(base_config, "absolute_injection", "multipole_ls", fallback=None)
    if injection_multipole_ls is None or len(injection_multipole_ls) == 0:
        raise ValueError("Absolute paper-grid configs require [absolute_injection] multipole_ls.")
    if target_multipole_l not in injection_multipole_ls:
        raise ValueError(
            "Target multipole L={} is not present in the injected multipole list {}.".format(
                target_multipole_l,
                injection_multipole_ls,
            )
        )

    if base_config.has_option("absolute_injection", "A_L_by_L"):
        absolute_component_amplitudes = {
            "A_L_by_L": parse_literal(base_config, "absolute_injection", "A_L_by_L", fallback=None)
        }
    else:
        absolute_component_amplitudes = {
            "log_A_L_by_L": parse_literal(base_config, "absolute_injection", "log_A_L_by_L", fallback=None)
        }
    absolute_amplitudes_by_l = normalize_absolute_amplitude_dict(
        absolute_component_amplitudes,
        injection_multipole_ls,
    )

    noise_truevals = copy.deepcopy(parse_literal(base_config, "inj", "truevals", fallback={})["noise"])
    iso_truevals = copy.deepcopy(parse_literal(base_config, "inj", "truevals", fallback={})["powerlaw_isgwb"])
    absolute_alpha = float(base_config.get("absolute_injection", "alpha", fallback=iso_truevals["alpha"]))

    dataset_specs = build_dataset_specs(
        injection_multipole_ls,
        target_multipole_l,
        absolute_a_l_grid,
        include_zero_baseline,
        seed_start,
    )

    summary_rows = []
    spectrum_name = Path(base_config.get("run_params", "input_spectrum", fallback="data_spectrum.npz")).name

    for dataset in dataset_specs:
        dataset_label = dataset["dataset_label"]
        injected_a2 = float(dataset["target_a_l"])
        dataset_root = output_root / dataset_label
        shared_dir = dataset_root / "shared_data"
        component_root = dataset_root / "shared_components"
        null_dir = dataset_root / "null_model"
        alt_dir = dataset_root / "alt_model"
        shared_spectrum_path = shared_dir / spectrum_name
        composite_injection_path = shared_dir / "injection.pickle"

        active_amplitudes = {
            int(multipole_l): float(absolute_amplitudes_by_l[multipole_l])
            for multipole_l in dataset["active_multipole_ls"]
        }
        if target_multipole_l in active_amplitudes:
            active_amplitudes[target_multipole_l] = injected_a2

        dataset_manifest = base_config_path.parent / "unused"
        del dataset_manifest

        component_specs = []

        noise_iso_dir = component_root / "noise_plus_isotropic"
        noise_iso_config = build_noise_plus_isotropic_config(
            base_config,
            noise_truevals,
            iso_truevals,
            noise_iso_dir,
            dataset["seed"],
            spectrum_name,
        )
        noise_iso_config_path = dataset_root / "noise_plus_isotropic.ini"
        write_config(noise_iso_config, noise_iso_config_path)
        noise_iso_spectrum_path = noise_iso_dir / spectrum_name
        noise_iso_injection_path = noise_iso_dir / "injection.pickle"

        component_index = 1
        for multipole_l in sorted(active_amplitudes.keys()):
            component_seed = build_component_seed(dataset["seed"], component_index, seed_stride)
            component_index += 1

            noise_only_dir = component_root / "L{:02d}_noise_only".format(multipole_l)
            noise_plus_signal_dir = component_root / "L{:02d}_noise_plus_signal".format(multipole_l)

            noise_only_config = build_noise_only_config(
                base_config,
                noise_truevals,
                noise_only_dir,
                component_seed,
                spectrum_name,
            )
            noise_only_config_path = dataset_root / "L{:02d}_noise_only.ini".format(multipole_l)
            write_config(noise_only_config, noise_only_config_path)

            noise_plus_signal_config = build_noise_plus_multipole_config(
                base_config,
                noise_truevals,
                absolute_alpha,
                multipole_l,
                active_amplitudes[multipole_l],
                noise_plus_signal_dir,
                component_seed,
                spectrum_name,
            )
            noise_plus_signal_config_path = dataset_root / "L{:02d}_noise_plus_signal.ini".format(multipole_l)
            write_config(noise_plus_signal_config, noise_plus_signal_config_path)

            component_specs.append(
                {
                    "multipole_l": multipole_l,
                    "noise_only_config": noise_only_config_path,
                    "noise_only_spectrum": noise_only_dir / spectrum_name,
                    "noise_plus_signal_config": noise_plus_signal_config_path,
                    "noise_plus_signal_spectrum": noise_plus_signal_dir / spectrum_name,
                    "noise_plus_signal_injection": noise_plus_signal_dir / "injection.pickle",
                }
            )

        null_config = build_null_config(base_config, shared_spectrum_path, null_dir, dataset["seed"])
        null_config_path = dataset_root / "null_model.ini"
        write_config(null_config, null_config_path)

        alt_config = build_alt_config(
            base_config,
            target_multipole_l,
            shared_spectrum_path,
            alt_dir,
            dataset["seed"],
        )
        alt_config_path = dataset_root / "alt_model.ini"
        write_config(alt_config, alt_config_path)

        if args.prepare_only:
            continue

        if (not args.reuse_existing) or (not noise_iso_spectrum_path.exists()):
            print("Generating absolute shared base '{}' in {}".format(dataset_label, noise_iso_dir))
            run_blip(noise_iso_config_path)

        for component_spec in component_specs:
            if (not args.reuse_existing) or (not component_spec["noise_only_spectrum"].exists()):
                print(
                    "Generating matched noise helper for L={} in {}".format(
                        component_spec["multipole_l"],
                        component_spec["noise_only_config"].parent,
                    )
                )
                run_blip(component_spec["noise_only_config"])
            if (not args.reuse_existing) or (not component_spec["noise_plus_signal_spectrum"].exists()):
                print(
                    "Generating absolute multipole L={} component in {}".format(
                        component_spec["multipole_l"],
                        component_spec["noise_plus_signal_config"].parent,
                    )
                )
                run_blip(component_spec["noise_plus_signal_config"])

        if (not args.reuse_existing) or (not shared_spectrum_path.exists()):
            print("Composing absolute shared dataset '{}' in {}".format(dataset_label, shared_dir))
            compose_shared_spectrum(noise_iso_spectrum_path, component_specs, shared_spectrum_path)

        if (not args.reuse_existing) or (not composite_injection_path.exists()):
            build_composite_injection(noise_iso_injection_path, component_specs, composite_injection_path)

        if (not args.reuse_existing) or (not (null_dir / "logz.txt").exists()):
            print("Running absolute null model for '{}' in {}".format(dataset_label, null_dir))
            run_blip(null_config_path)
        if (not args.reuse_existing) or (not (alt_dir / "logz.txt").exists()):
            print("Running absolute alt model for '{}' in {}".format(dataset_label, alt_dir))
            run_blip(alt_config_path)

        null_logz = read_logz(null_dir)
        alt_logz = read_logz(alt_dir)
        summary_rows.append(
            {
                "dataset_label": dataset_label,
                "injected_A2": injected_a2,
                "null_logz": null_logz,
                "alt_logz": alt_logz,
                "delta_logz": alt_logz - null_logz,
                "shared_dir": str(shared_dir.resolve()),
                "null_dir": str(null_dir.resolve()),
                "alt_dir": str(alt_dir.resolve()),
            }
        )

    if not args.prepare_only:
        write_summary_files(output_root, summary_rows)
        print("Finished absolute paper-grid study in {}".format(output_root))


if __name__ == "__main__":
    main()
