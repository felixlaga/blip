#!/usr/bin/env python3

import argparse
import ast
import configparser
import copy
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


def set_option(config, section, option, value):
    if not config.has_section(section):
        config.add_section(section)
    config.set(section, option, str(value))


def remove_option(config, section, option):
    if config.has_section(section) and config.has_option(section, option):
        config.remove_option(section, option)


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


def find_relmultipoles_component_name(config):
    injection_components = config.get("inj", "injection").split("+")
    matches = [
        component
        for component in injection_components
        if component.split("-")[0].split("_")[-1] == "relmultipoles"
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one '*_relmultipoles' injection component, found {} in '{}'.".format(
                len(matches),
                config.get("inj", "injection"),
            )
        )
    return matches[0]


def normalize_ratio_dict(component_truevals, multipole_ls):
    if "log_A_ratio_by_L" in component_truevals:
        ratio_dict = component_truevals["log_A_ratio_by_L"]
        return {int(key): float(value) for key, value in ratio_dict.items()}

    if "A_ratio_by_L" in component_truevals:
        ratio_dict = component_truevals["A_ratio_by_L"]
        return {int(key): float(np.log10(value)) for key, value in ratio_dict.items()}

    raise ValueError(
        "The relmultipoles injection truevals must provide either 'log_A_ratio_by_L' or 'A_ratio_by_L'."
    )


def build_analysis_aliases(model_string, injection_component_name):
    aliases = {}
    for component_name in model_string.split("+"):
        if component_name == "noise":
            continue
        aliases[component_name] = injection_component_name
    return aliases


def format_ratio_label(ratio):
    return "ratio_{}".format("{:.3f}".format(ratio).replace(".", "p"))


def write_summary_files(out_dir, rows):
    summary_path = out_dir / "bayes_factor_summary.txt"
    with open(summary_path, "w") as outfile:
        outfile.write(
            "# dataset_label injected_A2_over_A0 null_logz alt_logz delta_logz shared_dir null_dir alt_dir\n"
        )
        for row in rows:
            outfile.write(
                "{} {:.16e} {:.16e} {:.16e} {:.16e} {} {} {}\n".format(
                    row["dataset_label"],
                    row["injected_A2_over_A0"],
                    row["null_logz"],
                    row["alt_logz"],
                    row["delta_logz"],
                    row["shared_dir"],
                    row["null_dir"],
                    row["alt_dir"],
                )
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["dataset_label"] for row in rows]
    delta_logz = [row["delta_logz"] for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, delta_logz, color="royalblue", alpha=0.85)
    ax.axhline(0.0, color="0.5", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"$\Delta \log Z$")
    ax.set_title(r"Paper Grid: $\log Z_{\mathrm{alt}} - \log Z_{\mathrm{null}}$")
    fig.tight_layout()
    fig.savefig(out_dir / "bayes_factor_summary.png", dpi=200)
    plt.close(fig)


def build_dataset_specs(full_multipole_ls, target_multipole_l, ratio_grid, include_zero_baseline, seed_start):
    nuisance_multipole_ls = [multipole_l for multipole_l in full_multipole_ls if multipole_l != target_multipole_l]
    specs = []
    dataset_index = 0

    if include_zero_baseline:
        specs.append(
            {
                "dataset_label": "baseline_no_L2",
                "ratio": 0.0,
                "seed": seed_start + dataset_index,
                "injection_multipole_ls": nuisance_multipole_ls,
            }
        )
        dataset_index += 1

    for ratio in ratio_grid:
        specs.append(
            {
                "dataset_label": format_ratio_label(ratio),
                "ratio": float(ratio),
                "seed": seed_start + dataset_index,
                "injection_multipole_ls": list(full_multipole_ls),
            }
        )
        dataset_index += 1

    return specs


def main():
    parser = argparse.ArgumentParser(
        description="Generate a fixed paper grid of relmultipoles datasets and matched null/alt recoveries."
    )
    parser.add_argument(
        "base_config",
        nargs="?",
        default=str(REPO_ROOT / "paper_relmultipoles_l2_grid.ini"),
        help="Paper-grid base config.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the generated configs; do not launch BLIP.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing shared-data and recovery results when the expected outputs are already present.",
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config).resolve()
    base_config = load_config(str(base_config_path))
    output_root = resolve_out_dir(base_config, base_config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_config_path, output_root / base_config_path.name)

    if not base_config.has_section("paper_grid"):
        raise ValueError("Paper-grid configs must define a [paper_grid] section.")

    target_multipole_l = int(base_config.get("paper_grid", "target_multipole_l"))
    ratio_grid = parse_float_sequence(base_config, "paper_grid", "a_ratio_grid", fallback=[])
    include_zero_baseline = bool(int(base_config.get("paper_grid", "include_zero_baseline", fallback="1")))
    seed_start = int(base_config.get("paper_grid", "seed_start", fallback="200"))

    if len(ratio_grid) == 0 and not include_zero_baseline:
        raise ValueError("The paper grid must include at least one injected dataset.")
    if np.any(np.asarray(ratio_grid, dtype=float) <= 0):
        raise ValueError("All configured A_2/A_0 grid values must be > 0.")

    injection_component_name = find_relmultipoles_component_name(base_config)
    injection_multipole_ls = parse_integer_sequence(base_config, "inj", "multipole_ls", fallback=None)
    if injection_multipole_ls is None:
        injection_multipole_ls = parse_integer_sequence(base_config, "params", "multipole_ls", fallback=None)
    if injection_multipole_ls is None or len(injection_multipole_ls) == 0:
        raise ValueError("Paper-grid relmultipoles injections require [inj] multipole_ls or [params] multipole_ls.")
    if target_multipole_l not in injection_multipole_ls:
        raise ValueError(
            "Target multipole L={} is not present in the injected multipole list {}.".format(
                target_multipole_l,
                injection_multipole_ls,
            )
        )

    raw_truevals = parse_literal(base_config, "inj", "truevals", fallback=None)
    if raw_truevals is None:
        raise ValueError("Paper-grid configs must provide [inj] truevals.")
    component_truevals = copy.deepcopy(raw_truevals[injection_component_name])
    if "omega0" not in component_truevals:
        raise ValueError("The relmultipoles injection component must specify an isotropic amplitude 'omega0'.")
    base_log_ratio_by_l = normalize_ratio_dict(component_truevals, injection_multipole_ls)
    for multipole_l in injection_multipole_ls:
        if multipole_l not in base_log_ratio_by_l:
            raise ValueError("Missing injected A_L/A_0 value for L={}.".format(multipole_l))

    dataset_specs = build_dataset_specs(
        injection_multipole_ls,
        target_multipole_l,
        ratio_grid,
        include_zero_baseline,
        seed_start,
    )

    summary_rows = []
    input_spectrum_name = Path(base_config.get("run_params", "input_spectrum", fallback="data_spectrum.npz")).name

    for dataset in dataset_specs:
        dataset_label = dataset["dataset_label"]
        injected_ratio = float(dataset["ratio"])
        dataset_root = output_root / dataset_label
        shared_dir = dataset_root / "shared_data"
        null_dir = dataset_root / "null_model"
        alt_dir = dataset_root / "alt_model"
        shared_spectrum_path = shared_dir / input_spectrum_name

        shared_config = copy.deepcopy(base_config)
        shared_truevals = copy.deepcopy(raw_truevals)
        shared_component_truevals = copy.deepcopy(component_truevals)
        active_multipole_ls = list(dataset["injection_multipole_ls"])
        active_log_ratio_by_l = {
            int(multipole_l): float(base_log_ratio_by_l[multipole_l])
            for multipole_l in active_multipole_ls
        }
        if target_multipole_l in active_log_ratio_by_l:
            active_log_ratio_by_l[target_multipole_l] = float(np.log10(injected_ratio))

        shared_component_truevals["log_A_ratio_by_L"] = active_log_ratio_by_l
        shared_component_truevals.pop("A_ratio_by_L", None)
        shared_truevals[injection_component_name] = shared_component_truevals

        set_option(shared_config, "params", "model", "noise+powerlaw_relmultipoles")
        set_option(shared_config, "params", "alias", repr({"powerlaw_relmultipoles": injection_component_name}))
        set_option(shared_config, "params", "load_data", 0)
        set_option(shared_config, "params", "lmax", max(active_multipole_ls))
        set_option(shared_config, "params", "multipole_ls", repr(active_multipole_ls))
        set_option(shared_config, "inj", "doInj", 1)
        set_option(shared_config, "inj", "injection", "noise+powerlaw_relmultipoles")
        set_option(shared_config, "inj", "inj_lmax", max(active_multipole_ls))
        set_option(shared_config, "inj", "multipole_ls", repr(active_multipole_ls))
        set_option(shared_config, "inj", "truevals", repr(shared_truevals))
        set_option(shared_config, "run_params", "FixSeed", 1)
        set_option(shared_config, "run_params", "seed", dataset["seed"])
        set_option(shared_config, "run_params", "generate_only", 1)
        set_option(shared_config, "run_params", "doPreProc", 1)
        set_option(shared_config, "run_params", "out_dir", str(shared_dir))
        set_option(shared_config, "run_params", "input_spectrum", input_spectrum_name)
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
        null_config_path = dataset_root / "null_model.ini"
        write_config(null_config, null_config_path)

        alt_config = copy.deepcopy(base_config)
        set_option(alt_config, "params", "model", "noise+powerlaw_relmultipoles")
        set_option(alt_config, "params", "alias", repr(build_analysis_aliases("noise+powerlaw_relmultipoles", injection_component_name)))
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
        alt_config_path = dataset_root / "alt_model.ini"
        write_config(alt_config, alt_config_path)

        if args.prepare_only:
            continue

        if (not args.reuse_existing) or (not (null_dir / "logz.txt").exists()):
            print("Running null model for '{}' in {}".format(dataset_label, null_dir))
            run_blip(null_config_path)
        if (not args.reuse_existing) or (not (alt_dir / "logz.txt").exists()):
            print("Running alt model for '{}' in {}".format(dataset_label, alt_dir))
            run_blip(alt_config_path)

        null_logz = read_logz(null_dir)
        alt_logz = read_logz(alt_dir)
        summary_rows.append(
            {
                "dataset_label": dataset_label,
                "injected_A2_over_A0": injected_ratio,
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
        print("Finished paper-grid study in {}".format(output_root))


if __name__ == "__main__":
    main()
