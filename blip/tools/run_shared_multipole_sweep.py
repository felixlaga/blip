#!/usr/bin/env python3

import argparse
import configparser
import copy
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_BLIP_PATH = REPO_ROOT / "blip" / "run_blip"


def parse_multipoles(spec):
    values = []
    for token in spec.split(','):
        token = token.strip()
        if token == '':
            continue
        if '-' in token:
            start_str, end_str = token.split('-', 1)
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


def set_option(config, section, option, value):
    if not config.has_section(section):
        config.add_section(section)
    config.set(section, option, str(value))


def write_config(config, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as outfile:
        config.write(outfile)


def resolve_out_dir(config, config_path):
    out_dir = config.get('run_params', 'out_dir')
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


def summarize_parameter(samples, column):
    qlo, qmed, qhi = np.quantile(samples[:, column], [0.025, 0.5, 0.975])
    return float(qlo), float(qmed), float(qhi)


def summarize_recovery_run(run_dir, multipole_l):
    with open(run_dir / 'model.pickle', 'rb') as infile:
        model = pickle.load(infile)
    samples = np.atleast_2d(np.loadtxt(run_dir / 'post_samples.txt'))
    logz = np.loadtxt(run_dir / 'logz.txt')
    logzerr = np.loadtxt(run_dir / 'logzerr.txt')

    parameter_lookup = {name: idx for idx, name in enumerate(model.parameters['all'])}
    noise_submodel = next((sm for sm in model.submodels.values() if sm.name.split('-')[0] == 'noise'), None)
    multipole_submodel = next((sm for sm in model.submodels.values() if sm.name.split('-')[0].split('_')[-1] == 'multipole'), None)
    if multipole_submodel is None:
        raise ValueError("Run '{}' does not contain a fixed-L multipole model.".format(run_dir))

    def percentile_summary(parameter_name):
        if parameter_name is None or parameter_name not in parameter_lookup:
            return np.nan, np.nan, np.nan
        return summarize_parameter(samples, parameter_lookup[parameter_name])

    noise_parameters = noise_submodel.spectral_parameters if noise_submodel is not None else []
    alpha_name = next((name for name in multipole_submodel.spectral_parameters if r'$\alpha' in name), None)
    amplitude_name = multipole_submodel.get_single_multipole_amplitude_parameter()

    _, log_np_med, _ = percentile_summary(noise_parameters[0] if len(noise_parameters) > 0 else None)
    _, log_na_med, _ = percentile_summary(noise_parameters[1] if len(noise_parameters) > 1 else None)
    _, alpha_med, _ = percentile_summary(alpha_name)
    amp_q025, amp_med, amp_q975 = percentile_summary(amplitude_name)

    return {
        'multipole_l': multipole_l,
        'logz': float(np.ravel(logz)[-1]),
        'logzerr': float(np.ravel(logzerr)[-1]),
        'median_log_Np': log_np_med,
        'median_log_Na': log_na_med,
        'median_alpha': alpha_med,
        'median_log_A_L': amp_med,
        'q025_log_A_L': amp_q025,
        'q975_log_A_L': amp_q975,
    }


def write_summary(out_dir, summary_rows):
    if len(summary_rows) == 0:
        return

    import matplotlib.pyplot as plt

    summary_rows = sorted(summary_rows, key=lambda row: row['multipole_l'])
    best_logz = max(row['logz'] for row in summary_rows)

    header = "multipole_l logz logzerr delta_logz median_log_Np median_log_Na median_alpha median_log_A_L q025_log_A_L q975_log_A_L"
    table = []
    for row in summary_rows:
        delta_logz = row['logz'] - best_logz
        table.append([
            row['multipole_l'],
            row['logz'],
            row['logzerr'],
            delta_logz,
            row['median_log_Np'],
            row['median_log_Na'],
            row['median_alpha'],
            row['median_log_A_L'],
            row['q025_log_A_L'],
            row['q975_log_A_L'],
        ])

    np.savetxt(out_dir / 'multipole_sweep_summary.txt', np.asarray(table), header=header)

    multipoles = [row['multipole_l'] for row in summary_rows]
    delta_logz = [row['logz'] - best_logz for row in summary_rows]
    med_logA = [row['median_log_A_L'] for row in summary_rows]
    q025_logA = [row['q025_log_A_L'] for row in summary_rows]
    q975_logA = [row['q975_log_A_L'] for row in summary_rows]
    err_low = np.asarray(med_logA) - np.asarray(q025_logA)
    err_high = np.asarray(q975_logA) - np.asarray(med_logA)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(multipoles, delta_logz, marker='o', color='slateblue')
    axes[0].axhline(0.0, color='0.5', ls='--', lw=1)
    axes[0].set_xlabel('Recovery multipole L')
    axes[0].set_ylabel(r'$\Delta \log Z$')
    axes[0].set_title('Relative evidence')

    axes[1].errorbar(multipoles, med_logA, yerr=[err_low, err_high], marker='o', color='royalblue', lw=1.5, capsize=3)
    axes[1].set_xlabel('Recovery multipole L')
    axes[1].set_ylabel(r'$\log_{10}(A_L)$')
    axes[1].set_title('Recovered total-power amplitude')

    plt.tight_layout()
    fig.savefig(out_dir / 'multipole_sweep_summary.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run the individual paper multipole configs against one shared injected dataset.")
    parser.add_argument(
        'base_config',
        nargs='?',
        default=str(REPO_ROOT / 'params_all_sweep_paper.ini'),
        help="Base config used to generate the shared dataset.",
    )
    parser.add_argument(
        '--multipoles',
        default='0-10',
        help="Comma-separated list and/or ranges of recovery multipoles, e.g. '0-10' or '0,2,4-6'.",
    )
    parser.add_argument(
        '--reuse-shared-data',
        action='store_true',
        help="Reuse an existing shared_data/data_spectrum.npz instead of regenerating it.",
    )
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help="Only write the generated configs; do not launch BLIP.",
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config).resolve()
    base_config = load_config(str(base_config_path))
    output_root = resolve_out_dir(base_config, base_config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_config_path, output_root / base_config_path.name)

    injected_l = base_config.getint('inj', 'multipole_l', fallback=base_config.getint('params', 'multipole_l'))
    spectrum_name = os.path.basename(base_config.get('run_params', 'input_spectrum', fallback='data_spectrum.npz'))
    multipoles = parse_multipoles(args.multipoles)

    shared_dir = output_root / 'shared_data'
    shared_config_path = output_root / 'shared_data.ini'
    shared_spectrum_path = (shared_dir / spectrum_name).resolve()

    shared_config = copy.deepcopy(base_config)
    set_option(shared_config, 'run_params', 'multipole_sweep', 0)
    set_option(shared_config, 'run_params', 'generate_only', 1)
    set_option(shared_config, 'run_params', 'doPreProc', 1)
    set_option(shared_config, 'run_params', 'out_dir', str(shared_dir))
    set_option(shared_config, 'run_params', 'input_spectrum', spectrum_name)
    write_config(shared_config, shared_config_path)

    if not args.prepare_only and (not args.reuse_shared_data or not shared_spectrum_path.exists()):
        print("Preparing shared dataset in {}".format(shared_dir))
        run_blip(shared_config_path)

    summary_rows = []

    for multipole_l in multipoles:
        template_path = REPO_ROOT / 'params_single_multipole_paper_l{}.ini'.format(multipole_l)
        if template_path.exists():
            run_config = load_config(str(template_path))
        else:
            run_config = copy.deepcopy(base_config)

        run_dir = output_root / 'recover_l{:02d}'.format(multipole_l)
        run_config_path = output_root / 'recover_l{:02d}.ini'.format(multipole_l)

        set_option(run_config, 'params', 'multipole_l', multipole_l)
        set_option(run_config, 'inj', 'multipole_l', injected_l)
        set_option(run_config, 'run_params', 'multipole_sweep', 0)
        set_option(run_config, 'run_params', 'generate_only', 0)
        set_option(run_config, 'run_params', 'doPreProc', 0)
        set_option(run_config, 'run_params', 'out_dir', str(run_dir))
        set_option(run_config, 'run_params', 'input_spectrum', str(shared_spectrum_path))
        write_config(run_config, run_config_path)

        if args.prepare_only:
            continue

        print("Recovering L={} in {}".format(multipole_l, run_dir))
        run_blip(run_config_path)
        summary_rows.append(summarize_recovery_run(run_dir, multipole_l))

    if not args.prepare_only:
        write_summary(output_root, summary_rows)
        print("Finished shared-data sweep in {}".format(output_root))


if __name__ == '__main__':
    main()
