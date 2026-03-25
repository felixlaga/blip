#!/usr/bin/env python3

import argparse
import configparser
import os


def parse_prior_bounds(config, option, default):
    '''
    Parse a two-element prior interval from the optional [priors] section.
    '''
    if config.has_section('priors') and config.has_option('priors', option):
        bounds = eval(str(config.get('priors', option)))
    else:
        bounds = default
    if len(bounds) != 2:
        raise ValueError("Prior '{}' must have exactly two bounds.".format(option))
    bounds = [float(bounds[0]), float(bounds[1])]
    bounds.sort()
    return bounds


def build_prior_bounds(config):
    '''
    Build BLIP's prior-bounds dictionary exactly as run_blip does.
    '''
    return {
        'log10_Ac': parse_prior_bounds(config, 'log10_Ac', [-14, -8]),
        'log_Np': parse_prior_bounds(config, 'log_Np', [-44, -39]),
        'log_Na': parse_prior_bounds(config, 'log_Na', [-51, -46]),
        'alpha': parse_prior_bounds(config, 'alpha', [-5, 5]),
        'log_omega0': parse_prior_bounds(config, 'log_omega0', [-14, 8]),
        'log_A_L': parse_prior_bounds(config, 'log_A_L', [-14, 8]),
        'alpha1': parse_prior_bounds(config, 'alpha1', [-4, 6]),
        'alpha2': parse_prior_bounds(config, 'alpha2', [0, 40]),
        'log_fbreak': parse_prior_bounds(config, 'log_fbreak', [-4, -2]),
        'log_fcut': parse_prior_bounds(config, 'log_fcut', [-4, -2]),
        'log_fscale': parse_prior_bounds(config, 'log_fscale', [-4, -2]),
    }


def get_powerlaw_amplitude_prior_key(model_name):
    '''
    Mirror the current model-specific amplitude-prior selection for power-law models.
    '''
    if 'powerlaw_fixedLchannels' in model_name.split('+'):
        return 'log10_Ac'
    if 'powerlaw_multipole' in model_name.split('+'):
        return 'log_A_L'
    return 'log_omega0'


def main():
    parser = argparse.ArgumentParser(description='Inspect BLIP prior parsing without running the sampler.')
    parser.add_argument('paramsfile', help='Path to the BLIP .ini file to inspect')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    read_files = config.read(args.paramsfile)
    abs_path = os.path.abspath(args.paramsfile)

    print("[prior_debug] requested_path={}".format(args.paramsfile))
    print("[prior_debug] absolute_path={}".format(abs_path))
    print("[prior_debug] read_files={}".format(read_files))
    print("[prior_debug] sections={}".format(config.sections()))

    if not config.has_section('params'):
        raise ValueError("Missing [params] section in '{}'.".format(args.paramsfile))

    model_name = str(config.get('params', 'model'))
    prior_bounds = build_prior_bounds(config)
    raw_prior_keys = list(config['priors'].keys()) if config.has_section('priors') else []
    amplitude_prior_key = get_powerlaw_amplitude_prior_key(model_name)

    print("[prior_debug] model={}".format(model_name))
    print("[prior_debug] raw_priors_section_keys={}".format(raw_prior_keys))
    print("[prior_debug] parsed_prior_bounds={}".format(prior_bounds))
    print("[prior_debug] powerlaw_amplitude_prior_key={}".format(amplitude_prior_key))

    for key in ['log10_Ac', 'log_Np', 'log_Na', 'alpha', 'log_A_L']:
        print("[prior_debug] lookup {} -> {}".format(key, prior_bounds[key]))


if __name__ == '__main__':
    main()
