#!/usr/bin/env python3

import argparse
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl"))
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def flatten_complex(array):
    '''
    Flatten a complex array into concatenated real and imaginary parts.
    '''
    return np.concatenate([np.real(array).ravel(), np.imag(array).ravel()])


def cosine_similarity(array_a, array_b):
    '''
    Return the cosine similarity between two complex-valued templates.
    '''
    flat_a = flatten_complex(array_a)
    flat_b = flatten_complex(array_b)
    return float(np.dot(flat_a, flat_b) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b)))


def summarize_interval(samples):
    '''
    Return the 95% central interval and median for one posterior column.
    '''
    return np.quantile(samples, [0.025, 0.5, 0.975])


def get_matching_injection_component(model_submodel, injection):
    '''
    Return the injection component associated with a recovery submodel.
    '''
    if injection is None:
        return None
    if model_submodel.name in injection.components:
        return injection.components[model_submodel.name]
    model_alias = getattr(model_submodel, 'alias', None)
    if model_alias in injection.components:
        return injection.components[model_alias]
    return None


def extract_truth_ratios(multipoles_submodel, injection_component):
    '''
    Return injected relative total powers A_L / A_0 when available.
    '''
    if injection_component is None:
        return None
    if hasattr(injection_component, 'relative_multipole_truths'):
        truth_lmax = min(
            multipoles_submodel.multipole_lmax,
            injection_component.relative_multipole_truths.size - 1,
        )
        truths = np.full(multipoles_submodel.multipole_lmax + 1, np.nan)
        truths[:truth_lmax + 1] = injection_component.relative_multipole_truths[:truth_lmax + 1]
        return truths
    if not hasattr(injection_component, 'alms_inj'):
        return None
    available_lmax = int(np.sqrt(injection_component.alms_inj.size) - 1)
    truth_lmax = min(multipoles_submodel.multipole_lmax, available_lmax)
    relative_powers = injection_component.compute_angular_power_spectrum(
        injection_component.alms_inj,
        lmax=truth_lmax,
        relative_to_l0=True,
    )
    truths = np.full(multipoles_submodel.multipole_lmax + 1, np.nan)
    truths[:truth_lmax + 1] = relative_powers
    return truths


def relative_error(reference, test):
    '''
    Return the relative Frobenius error between two complex response stacks.
    '''
    numerator = np.linalg.norm(flatten_complex(reference - test))
    denominator = np.linalg.norm(flatten_complex(reference))
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def summarize_template_geometry(submodel):
    '''
    Summarize response-template norms, correlations, and conditioning.
    '''
    templates = [('iso', submodel.response_mat)]
    templates += [
        ('L{}'.format(multipole_l), submodel.relative_response_mats[..., idx])
        for idx, multipole_l in enumerate(submodel.relative_multipole_ls)
    ]

    iso_norm = np.linalg.norm(flatten_complex(submodel.response_mat))
    lines = []
    lines.append('[template norms]')
    for name, template in templates:
        norm = np.linalg.norm(flatten_complex(template))
        rel_norm = norm / iso_norm if iso_norm > 0 else np.nan
        rel_log10 = np.log10(rel_norm) if rel_norm > 0 else -np.inf
        lines.append(
            '{} norm={:.6e} rel_to_iso={:.6e} log10_rel={:.3f}'.format(
                name,
                norm,
                rel_norm,
                rel_log10,
            )
        )

    lines.append('[correlation to isotropic template]')
    for name, template in templates[1:]:
        lines.append('{} corr_to_iso={:.6f}'.format(name, cosine_similarity(submodel.response_mat, template)))

    pairwise = []
    for idx_a, (name_a, template_a) in enumerate(templates[1:], start=1):
        for name_b, template_b in templates[idx_a + 1:]:
            pairwise.append((cosine_similarity(template_a, template_b), name_a, name_b))

    lines.append('[largest pairwise template correlations]')
    for corr_value, name_a, name_b in sorted(pairwise, reverse=True)[:10]:
        lines.append('{} {} corr={:.6f}'.format(name_a, name_b, corr_value))

    template_matrix = np.stack([flatten_complex(template) for _, template in templates[1:]], axis=1)
    template_matrix = template_matrix / np.linalg.norm(template_matrix, axis=0, keepdims=True)
    gram = template_matrix.T @ template_matrix
    eigenvalues = np.linalg.eigvalsh(gram)
    lines.append('[relative-template Gram matrix]')
    lines.append('eigenvalues={}'.format(' '.join(['{:.6e}'.format(value) for value in eigenvalues])))
    lines.append('condition_number={:.6e}'.format(float(eigenvalues[-1] / eigenvalues[0])))

    return lines


def summarize_injection_mismatch(submodel, injection_component):
    '''
    Compare the total-power template bank against the actual injected response.
    '''
    if injection_component is None:
        return ['[injection mismatch]', 'no matching injection component found']

    lines = ['[injection mismatch]']
    truth_ratios = extract_truth_ratios(submodel, injection_component)
    if truth_ratios is not None:
        lines.append('truth ratios:')
        for multipole_l in submodel.relative_multipole_ls:
            if multipole_l < truth_ratios.size and np.isfinite(truth_ratios[multipole_l]):
                truth_value = truth_ratios[multipole_l]
                lines.append(
                    'L{} A_L/A_0={:.6e} log10={}'.format(
                        multipole_l,
                        truth_value,
                        '{:.3f}'.format(np.log10(truth_value)) if truth_value > 0 else '-inf',
                    )
                )
            else:
                lines.append('L{} not injected / unavailable'.format(multipole_l))

    if not hasattr(injection_component, 'alms_inj'):
        lines.append('matched total-power injection; no coherent-sky mismatch term to evaluate')
        return lines

    response_basis = injection_component.recompute_response(submodel.f0, submodel.tsegmid)
    coherent_response = np.einsum('ijklm,m->ijkl', response_basis, injection_component.alms_inj)
    coherent_response = 0.5 * (
        coherent_response + np.swapaxes(np.conj(coherent_response), 0, 1)
    )

    if truth_ratios is not None:
        weights = np.zeros(len(submodel.relative_multipole_ls))
        for idx, multipole_l in enumerate(submodel.relative_multipole_ls):
            if multipole_l < truth_ratios.size and np.isfinite(truth_ratios[multipole_l]):
                weights[idx] = truth_ratios[multipole_l]
        truth_projection = np.array(submodel.response_mat, copy=True)
        truth_projection += np.tensordot(submodel.relative_response_mats, weights, axes=([4], [0]))
        truth_projection = 0.5 * (
            truth_projection + np.swapaxes(np.conj(truth_projection), 0, 1)
        )
        lines.append(
            'relative_error(truth angular-power projection)={:.6f}'.format(
                relative_error(coherent_response, truth_projection)
            )
        )

    template_bank = [submodel.response_mat]
    template_bank += [
        submodel.relative_response_mats[..., idx]
        for idx in range(submodel.relative_response_mats.shape[-1])
    ]
    design = np.stack([flatten_complex(template) for template in template_bank], axis=1)
    target = flatten_complex(coherent_response)
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    least_squares_projection = sum(coeff * template for coeff, template in zip(coefficients, template_bank))
    least_squares_projection = 0.5 * (
        least_squares_projection + np.swapaxes(np.conj(least_squares_projection), 0, 1)
    )

    lines.append(
        'relative_error(best linear projection)={:.6f}'.format(
            relative_error(coherent_response, least_squares_projection)
        )
    )
    lines.append(
        'least_squares_coeff_range=[{:.6e}, {:.6e}]'.format(
            float(np.min(coefficients)),
            float(np.max(coefficients)),
        )
    )
    lines.append(
        'num_negative_least_squares_coefficients={}'.format(
            int(np.count_nonzero(coefficients < 0))
        )
    )

    return lines


def summarize_posterior(model, submodel, posterior_samples):
    '''
    Summarize posterior width, boundary pressure, and dominant correlations.
    '''
    parameter_lookup = {name: idx for idx, name in enumerate(model.parameters['all'])}
    prior_bounds = model.params.get('prior_bounds', {})
    lower, upper = prior_bounds.get('log_A_ratio', [-6.0, 6.0])
    prior_width = upper - lower
    lower_threshold = lower + 0.05 * prior_width
    upper_threshold = upper - 0.05 * prior_width

    lines = ['[posterior summary]']
    diagnostic_columns = []
    diagnostic_names = []

    alpha_name = next(
        (name for name in submodel.spectral_parameters if r'$\alpha' in name),
        None,
    )
    omega_name = next(
        (name for name in submodel.spectral_parameters if r'\Omega_0' in name),
        None,
    )

    for parameter_name in [alpha_name, omega_name]:
        if parameter_name is None:
            continue
        samples = posterior_samples[:, parameter_lookup[parameter_name]]
        q025, median, q975 = summarize_interval(samples)
        lines.append(
            '{} q025={:.6f} median={:.6f} q975={:.6f}'.format(
                parameter_name,
                q025,
                median,
                q975,
            )
        )
        diagnostic_columns.append(samples)
        diagnostic_names.append(parameter_name)

    for multipole_l in submodel.relative_multipole_ls:
        parameter_name = submodel.get_relative_multipole_amplitude_parameter(multipole_l)
        samples = posterior_samples[:, parameter_lookup[parameter_name]]
        q025, median, q975 = summarize_interval(samples)
        frac_near_lower = float(np.mean(samples <= lower_threshold))
        frac_near_upper = float(np.mean(samples >= upper_threshold))
        lines.append(
            '{} q025={:.6f} median={:.6f} q975={:.6f} frac_near_lower={:.3f} frac_near_upper={:.3f}'.format(
                parameter_name,
                q025,
                median,
                q975,
                frac_near_lower,
                frac_near_upper,
            )
        )
        diagnostic_columns.append(samples)
        diagnostic_names.append(parameter_name)

    if len(diagnostic_columns) >= 2:
        stacked = np.column_stack(diagnostic_columns)
        corr = np.corrcoef(stacked, rowvar=False)
        pairwise = []
        for idx_a, name_a in enumerate(diagnostic_names):
            for idx_b, name_b in enumerate(diagnostic_names[idx_a + 1:], start=idx_a + 1):
                pairwise.append((abs(float(corr[idx_a, idx_b])), name_a, name_b, float(corr[idx_a, idx_b])))
        lines.append('[largest posterior correlations]')
        for _, name_a, name_b, corr_value in sorted(pairwise, reverse=True)[:10]:
            lines.append('{} {} corr={:.6f}'.format(name_a, name_b, corr_value))

    return lines


def main():
    parser = argparse.ArgumentParser(description='Diagnose relative-multipole identifiability for a BLIP run directory.')
    parser.add_argument('rundir', help='Path to a BLIP run directory containing model.pickle')
    args = parser.parse_args()

    run_dir = Path(args.rundir).resolve()
    with open(run_dir / 'model.pickle', 'rb') as infile:
        model = pickle.load(infile)

    injection = None
    injection_path = run_dir / 'injection.pickle'
    if injection_path.exists():
        with open(injection_path, 'rb') as infile:
            injection = pickle.load(infile)

    posterior_samples = None
    post_path = run_dir / 'post_samples.txt'
    if post_path.exists():
        posterior_samples = np.atleast_2d(np.loadtxt(post_path))

    for submodel_name in model.submodel_names:
        submodel = model.submodels[submodel_name]
        if getattr(submodel, 'spatial_model_name', None) != 'multipoles':
            continue

        report_lines = []
        report_lines.append('# {}'.format(submodel_name))
        report_lines.append('# {}'.format(submodel.fancyname))
        report_lines.extend(summarize_template_geometry(submodel))
        report_lines.extend(summarize_injection_mismatch(submodel, get_matching_injection_component(submodel, injection)))
        if posterior_samples is not None:
            report_lines.extend(summarize_posterior(model, submodel, posterior_samples))

        report = '\n'.join(report_lines) + '\n'
        report_path = run_dir / '{}_diagnostics.txt'.format(submodel_name.replace('/', '_'))
        with open(report_path, 'w') as outfile:
            outfile.write(report)

        print(report, end='')
        print('wrote {}'.format(report_path))


if __name__ == '__main__':
    main()
