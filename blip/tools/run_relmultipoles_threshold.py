#!/usr/bin/env python3

import argparse
import configparser
import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_BLIP_PATH = REPO_ROOT / "blip" / "run_blip"


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


def remove_option(config, section, option):
    if config.has_section(section) and config.has_option(section, option):
        config.remove_option(section, option)


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


def parse_integer_sequence(config, section, option, fallback=None):
    if not config.has_section(section) or not config.has_option(section, option):
        return fallback
    values = eval(str(config.get(section, option)))
    if values is None:
        return fallback
    if isinstance(values, (int, np.integer)):
        return [int(values)]
    return [int(value) for value in values]


def parse_float_sequence(config, section, option, fallback=None):
    if not config.has_section(section) or not config.has_option(section, option):
        return fallback
    values = eval(str(config.get(section, option)))
    if values is None:
        return fallback
    if isinstance(values, (int, float, np.integer, np.floating)):
        return [float(values)]
    return [float(value) for value in values]


def read_logz(run_dir):
    logz = np.loadtxt(run_dir / 'logz.txt')
    logz = np.asarray(logz, dtype=float).reshape(-1)
    if logz.size == 0:
        raise ValueError("No log-evidence samples found in '{}'.".format(run_dir))
    return float(logz[-1])


def slugify_float(value):
    return "{:+.3f}".format(value).replace('+', 'p').replace('-', 'm').replace('.', 'p')


def find_relmultipoles_component_name(config):
    injection_components = config.get('inj', 'injection').split('+')
    matches = [component for component in injection_components if component.split('-')[0].split('_')[-1] == 'relmultipoles']
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one '*_relmultipoles' injection component, found {} in '{}'.".format(
                len(matches),
                config.get('inj', 'injection'),
            )
        )
    return matches[0]


def normalize_ratio_dict(component_truevals, multipole_ls):
    if 'log_A_ratio_by_L' in component_truevals:
        ratio_dict = component_truevals['log_A_ratio_by_L']
        return {int(key): float(value) for key, value in ratio_dict.items()}

    if 'A_ratio_by_L' in component_truevals:
        ratio_dict = component_truevals['A_ratio_by_L']
        return {int(key): float(np.log10(value)) for key, value in ratio_dict.items()}

    if 'log_A_ratios' in component_truevals:
        values = np.asarray(component_truevals['log_A_ratios'], dtype=float).reshape(-1)
        if values.size != len(multipole_ls):
            raise ValueError(
                "Configured 'log_A_ratios' has length {}, expected {} from multipole_ls."
                .format(values.size, len(multipole_ls))
            )
        return {int(multipole_l): float(value) for multipole_l, value in zip(multipole_ls, values)}

    if 'A_ratios' in component_truevals:
        values = np.asarray(component_truevals['A_ratios'], dtype=float).reshape(-1)
        if values.size != len(multipole_ls):
            raise ValueError(
                "Configured 'A_ratios' has length {}, expected {} from multipole_ls."
                .format(values.size, len(multipole_ls))
            )
        if np.any(values <= 0):
            raise ValueError("All configured A_ratio values must be > 0.")
        return {int(multipole_l): float(np.log10(value)) for multipole_l, value in zip(multipole_ls, values)}

    ratio_dict = {}
    for multipole_l in multipole_ls:
        log_key = 'log_A_ratio_{}'.format(multipole_l)
        linear_key = 'A_ratio_{}'.format(multipole_l)
        if log_key in component_truevals:
            ratio_dict[int(multipole_l)] = float(component_truevals[log_key])
        elif linear_key in component_truevals:
            linear_value = float(component_truevals[linear_key])
            if linear_value <= 0:
                raise ValueError("Configured {} must be > 0.".format(linear_key))
            ratio_dict[int(multipole_l)] = float(np.log10(linear_value))

    if len(ratio_dict) == len(multipole_ls):
        return ratio_dict

    raise ValueError(
        "The relmultipoles injection truevals must provide one of 'log_A_ratio_by_L', 'A_ratio_by_L', "
        "'log_A_ratios', 'A_ratios', or per-L keys like 'log_A_ratio_2'."
    )


def resolve_target_log_a_l_grid(config, component_truevals, target_multipole_l):
    log_omega0 = float(np.log10(component_truevals['omega0']))

    log_a_l_grid = parse_float_sequence(config, 'threshold', 'log_A_L_grid', fallback=None)
    if log_a_l_grid is not None:
        return [float(value) for value in log_a_l_grid]

    a_l_grid = parse_float_sequence(config, 'threshold', 'A_L_grid', fallback=None)
    if a_l_grid is not None:
        if np.any(np.asarray(a_l_grid) <= 0):
            raise ValueError("All configured A_L_grid values must be > 0.")
        return [float(np.log10(value)) for value in a_l_grid]

    log_a_ratio_grid = parse_float_sequence(config, 'threshold', 'log_A_ratio_grid', fallback=None)
    if log_a_ratio_grid is not None:
        return [float(log_omega0 + value) for value in log_a_ratio_grid]

    a_ratio_grid = parse_float_sequence(config, 'threshold', 'A_ratio_grid', fallback=None)
    if a_ratio_grid is not None:
        if np.any(np.asarray(a_ratio_grid) <= 0):
            raise ValueError("All configured A_ratio_grid values must be > 0.")
        return [float(log_omega0 + np.log10(value)) for value in a_ratio_grid]

    if target_multipole_l in normalize_ratio_dict(component_truevals, parse_integer_sequence(config, 'inj', 'multipole_ls', fallback=[])):
        default_log_a_l = log_omega0 + normalize_ratio_dict(
            component_truevals,
            parse_integer_sequence(config, 'inj', 'multipole_ls', fallback=[]),
        )[target_multipole_l]
        return [default_log_a_l]

    raise ValueError(
        "No target amplitude grid was configured. Provide one of [threshold] log_A_L_grid, A_L_grid, "
        "log_A_ratio_grid, or A_ratio_grid."
    )


def build_analysis_aliases(model_string, injection_component_name):
    aliases = {}
    for component_name in model_string.split('+'):
        if component_name == 'noise':
            continue
        aliases[component_name] = injection_component_name
    return aliases


def write_threshold_summaries(out_dir, detail_rows, aggregate_rows, evidence_threshold, efficiency_target):
    if len(detail_rows) > 0:
        detail_header = (
            "log_A_L target_log_A_ratio realization_index seed null_logz alt_logz "
            "delta_logz detected"
        )
        detail_table = np.asarray([
            [
                row['log_A_L'],
                row['target_log_A_ratio'],
                row['realization_index'],
                row['seed'],
                row['null_logz'],
                row['alt_logz'],
                row['delta_logz'],
                int(row['detected']),
            ]
            for row in detail_rows
        ])
        np.savetxt(out_dir / 'threshold_detail.txt', detail_table, header=detail_header)

    if len(aggregate_rows) == 0:
        return

    aggregate_header = (
        "log_A_L target_log_A_ratio detection_efficiency mean_delta_logz "
        "median_delta_logz min_delta_logz max_delta_logz"
    )
    aggregate_table = np.asarray([
        [
            row['log_A_L'],
            row['target_log_A_ratio'],
            row['detection_efficiency'],
            row['mean_delta_logz'],
            row['median_delta_logz'],
            row['min_delta_logz'],
            row['max_delta_logz'],
        ]
        for row in aggregate_rows
    ])
    np.savetxt(out_dir / 'threshold_summary.txt', aggregate_table, header=aggregate_header)

    recommended = next(
        (row for row in aggregate_rows if row['detection_efficiency'] >= efficiency_target),
        None,
    )
    with open(out_dir / 'threshold_recommendation.txt', 'w') as outfile:
        outfile.write("evidence_threshold {:.6f}\n".format(evidence_threshold))
        outfile.write("efficiency_target {:.6f}\n".format(efficiency_target))
        if recommended is None:
            outfile.write("recommended_log_A_L none\n")
            outfile.write("recommended_log_A_ratio none\n")
        else:
            outfile.write("recommended_log_A_L {:.16e}\n".format(recommended['log_A_L']))
            outfile.write("recommended_log_A_ratio {:.16e}\n".format(recommended['target_log_A_ratio']))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    log_a_l_values = [row['log_A_L'] for row in aggregate_rows]
    mean_delta = [row['mean_delta_logz'] for row in aggregate_rows]
    efficiency = [row['detection_efficiency'] for row in aggregate_rows]

    axes[0].plot(log_a_l_values, mean_delta, marker='o', color='royalblue')
    axes[0].axhline(evidence_threshold, color='0.5', ls='--', lw=1)
    axes[0].set_xlabel(r'Injected $\log_{10}(A_L)$')
    axes[0].set_ylabel(r'Mean $\Delta \log Z$')
    axes[0].set_title('Matched Null vs Alt')

    axes[1].plot(log_a_l_values, efficiency, marker='o', color='slateblue')
    axes[1].axhline(efficiency_target, color='0.5', ls='--', lw=1)
    axes[1].set_xlabel(r'Injected $\log_{10}(A_L)$')
    axes[1].set_ylabel('Detection efficiency')
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title(r'Fraction with $\Delta \log Z > {:.1f}$'.format(evidence_threshold))

    plt.tight_layout()
    fig.savefig(out_dir / 'threshold_summary.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate a relative-multipole detectability threshold from repeated shared-data realizations."
    )
    parser.add_argument(
        'base_config',
        nargs='?',
        default=str(REPO_ROOT / 'paper_relmultipoles_threshold_smoke.ini'),
        help="Threshold-study base config.",
    )
    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help="Only write the generated configs; do not launch BLIP.",
    )
    parser.add_argument(
        '--reuse-existing',
        action='store_true',
        help="Reuse existing shared-data and recovery results when the expected outputs are already present.",
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config).resolve()
    base_config = load_config(str(base_config_path))
    output_root = resolve_out_dir(base_config, base_config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_config_path, output_root / base_config_path.name)

    if not base_config.has_section('threshold'):
        raise ValueError("Threshold configs must define a [threshold] section.")

    target_multipole_l = int(base_config.get('threshold', 'target_multipole_l'))
    n_realizations = int(base_config.get('threshold', 'n_realizations', fallback=3))
    seed_start = int(base_config.get('threshold', 'seed_start', fallback=100))
    evidence_threshold = float(base_config.get('threshold', 'evidence_threshold', fallback=5.0))
    efficiency_target = float(base_config.get('threshold', 'efficiency_target', fallback=0.9))

    injection_component_name = find_relmultipoles_component_name(base_config)
    injection_multipole_ls = parse_integer_sequence(base_config, 'inj', 'multipole_ls', fallback=None)
    if injection_multipole_ls is None:
        injection_multipole_ls = parse_integer_sequence(base_config, 'params', 'multipole_ls', fallback=None)
    if injection_multipole_ls is None or len(injection_multipole_ls) == 0:
        raise ValueError("Threshold-study relmultipoles injections require [inj] multipole_ls or [params] multipole_ls.")
    if target_multipole_l not in injection_multipole_ls:
        raise ValueError("Target multipole L={} is not present in the injected multipole list {}.".format(
            target_multipole_l,
            injection_multipole_ls,
        ))

    raw_truevals = eval(str(base_config.get('inj', 'truevals')))
    component_truevals = copy.deepcopy(raw_truevals[injection_component_name])
    if 'omega0' not in component_truevals:
        raise ValueError("The relmultipoles injection component must specify an isotropic amplitude 'omega0'.")
    base_log_ratio_by_l = normalize_ratio_dict(component_truevals, injection_multipole_ls)
    for multipole_l in injection_multipole_ls:
        if multipole_l not in base_log_ratio_by_l:
            raise ValueError("The relmultipoles injection configuration is missing an A_L/A_0 value for L={}.".format(
                multipole_l
            ))

    log_a_l_grid = resolve_target_log_a_l_grid(base_config, component_truevals, target_multipole_l)
    null_multipole_ls = parse_integer_sequence(base_config, 'threshold', 'null_multipole_ls', fallback=None)
    if null_multipole_ls is None:
        null_multipole_ls = [multipole_l for multipole_l in injection_multipole_ls if multipole_l != target_multipole_l]
    alt_multipole_ls = parse_integer_sequence(base_config, 'threshold', 'alt_multipole_ls', fallback=None)
    if alt_multipole_ls is None:
        alt_multipole_ls = list(injection_multipole_ls)

    detail_rows = []
    aggregate_rows = []

    input_spectrum_name = os.path.basename(base_config.get('run_params', 'input_spectrum', fallback='data_spectrum.npz'))
    log_omega0 = float(np.log10(component_truevals['omega0']))

    for amplitude_index, log_a_l in enumerate(log_a_l_grid):
        target_log_a_ratio = float(log_a_l - log_omega0)
        delta_logz_values = []

        for realization_index in range(n_realizations):
            seed = seed_start + amplitude_index * 1000 + realization_index
            amplitude_slug = slugify_float(log_a_l)
            realization_root = output_root / 'logA_{}'.format(amplitude_slug) / 'realization_{:02d}'.format(realization_index)
            shared_dir = realization_root / 'shared_data'
            null_dir = realization_root / 'null_model'
            alt_dir = realization_root / 'alt_model'

            shared_config = copy.deepcopy(base_config)
            shared_truevals = eval(str(shared_config.get('inj', 'truevals')))
            shared_component_truevals = copy.deepcopy(shared_truevals[injection_component_name])
            updated_log_ratio_by_l = dict(base_log_ratio_by_l)
            updated_log_ratio_by_l[target_multipole_l] = target_log_a_ratio
            shared_component_truevals['log_A_ratio_by_L'] = updated_log_ratio_by_l
            shared_component_truevals.pop('A_ratio_by_L', None)
            shared_component_truevals.pop('log_A_ratios', None)
            shared_component_truevals.pop('A_ratios', None)
            shared_truevals[injection_component_name] = shared_component_truevals
            set_option(shared_config, 'inj', 'truevals', repr(shared_truevals))
            set_option(shared_config, 'run_params', 'FixSeed', 1)
            set_option(shared_config, 'run_params', 'seed', seed)
            set_option(shared_config, 'run_params', 'generate_only', 1)
            set_option(shared_config, 'run_params', 'doPreProc', 1)
            set_option(shared_config, 'run_params', 'out_dir', str(shared_dir))
            set_option(shared_config, 'run_params', 'input_spectrum', input_spectrum_name)
            shared_config_path = realization_root / 'shared_data.ini'
            write_config(shared_config, shared_config_path)

            shared_spectrum_path = shared_dir / input_spectrum_name
            if not args.prepare_only:
                if (not args.reuse_existing) or (not shared_spectrum_path.exists()):
                    print("Generating shared dataset for log10(A_{}) = {:.3f}, realization {} in {}".format(
                        target_multipole_l,
                        log_a_l,
                        realization_index,
                        shared_dir,
                    ))
                    run_blip(shared_config_path)

            null_config = copy.deepcopy(base_config)
            if len(null_multipole_ls) == 0:
                null_model = 'noise+powerlaw_isgwb'
            else:
                null_model = 'noise+powerlaw_relmultipoles'
                set_option(null_config, 'params', 'multipole_ls', repr(null_multipole_ls))
            set_option(null_config, 'params', 'model', null_model)
            set_option(null_config, 'params', 'load_data', 1)
            set_option(null_config, 'params', 'alias', repr(build_analysis_aliases(null_model, injection_component_name)))
            remove_option(null_config, 'params', 'multipole_l')
            if len(null_multipole_ls) == 0:
                remove_option(null_config, 'params', 'multipole_ls')
            set_option(null_config, 'inj', 'doInj', 0)
            remove_option(null_config, 'inj', 'multipole_l')
            remove_option(null_config, 'inj', 'multipole_ls')
            remove_option(null_config, 'inj', 'truevals')
            set_option(null_config, 'run_params', 'generate_only', 0)
            set_option(null_config, 'run_params', 'doPreProc', 0)
            set_option(null_config, 'run_params', 'out_dir', str(null_dir))
            set_option(null_config, 'run_params', 'input_spectrum', str(shared_spectrum_path.resolve()))
            null_config_path = realization_root / 'null_model.ini'
            write_config(null_config, null_config_path)

            alt_config = copy.deepcopy(base_config)
            alt_model = 'noise+powerlaw_relmultipoles'
            set_option(alt_config, 'params', 'model', alt_model)
            set_option(alt_config, 'params', 'load_data', 1)
            set_option(alt_config, 'params', 'multipole_ls', repr(alt_multipole_ls))
            set_option(alt_config, 'params', 'alias', repr(build_analysis_aliases(alt_model, injection_component_name)))
            remove_option(alt_config, 'params', 'multipole_l')
            set_option(alt_config, 'inj', 'doInj', 0)
            remove_option(alt_config, 'inj', 'multipole_l')
            remove_option(alt_config, 'inj', 'multipole_ls')
            remove_option(alt_config, 'inj', 'truevals')
            set_option(alt_config, 'run_params', 'generate_only', 0)
            set_option(alt_config, 'run_params', 'doPreProc', 0)
            set_option(alt_config, 'run_params', 'out_dir', str(alt_dir))
            set_option(alt_config, 'run_params', 'input_spectrum', str(shared_spectrum_path.resolve()))
            alt_config_path = realization_root / 'alt_model.ini'
            write_config(alt_config, alt_config_path)

            if not args.prepare_only:
                if (not args.reuse_existing) or (not (null_dir / 'logz.txt').exists()):
                    print("Running matched null model in {}".format(null_dir))
                    run_blip(null_config_path)
                if (not args.reuse_existing) or (not (alt_dir / 'logz.txt').exists()):
                    print("Running matched alt model in {}".format(alt_dir))
                    run_blip(alt_config_path)

                null_logz = read_logz(null_dir)
                alt_logz = read_logz(alt_dir)
                delta_logz = alt_logz - null_logz
                detected = bool(delta_logz > evidence_threshold)
                delta_logz_values.append(delta_logz)
                detail_rows.append({
                    'log_A_L': log_a_l,
                    'target_log_A_ratio': target_log_a_ratio,
                    'realization_index': realization_index,
                    'seed': seed,
                    'null_logz': null_logz,
                    'alt_logz': alt_logz,
                    'delta_logz': delta_logz,
                    'detected': detected,
                })

        if not args.prepare_only and len(delta_logz_values) > 0:
            delta_logz_values = np.asarray(delta_logz_values, dtype=float)
            aggregate_rows.append({
                'log_A_L': float(log_a_l),
                'target_log_A_ratio': float(target_log_a_ratio),
                'detection_efficiency': float(np.mean(delta_logz_values > evidence_threshold)),
                'mean_delta_logz': float(np.mean(delta_logz_values)),
                'median_delta_logz': float(np.median(delta_logz_values)),
                'min_delta_logz': float(np.min(delta_logz_values)),
                'max_delta_logz': float(np.max(delta_logz_values)),
            })

    if not args.prepare_only:
        aggregate_rows = sorted(aggregate_rows, key=lambda row: row['log_A_L'])
        write_threshold_summaries(output_root, detail_rows, aggregate_rows, evidence_threshold, efficiency_target)
        print("Finished threshold study in {}".format(output_root))


if __name__ == '__main__':
    main()
