#!/usr/bin/env python3

from pathlib import Path
import textwrap
import math


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = "./Storage/fixedl_iso_plus_one_from_all_fixedmultipoles_l10"
SHARED_SPECTRUM = OUTPUT_ROOT + "/shared_data/shared_all_fixedmultipoles_l10_spectrum.npz"

# Match BLIP's internal H0 convention when mapping the requested h^2 Omega_GW
# benchmark onto the plain Omega parameters used by the code.
H0_BLIP = 2.2e-18
H100 = 100.0 * 1000.0 / 3.085677581491367e22
LISA_FREF = 2.5e-2
INJECTION_ALPHA = 0.0
TARGET_H2_OMEGA0 = 5.0e-12
TARGET_OMEGA0 = TARGET_H2_OMEGA0 / ((H0_BLIP / H100) ** 2)
ANISOTROPIC_POWER_FRACTION_PER_L = 0.07
ALL_MULTIPOLES = list(range(1, 11))
ALL_ABSOLUTE_LS = "[" + ",".join(str(lval) for lval in ALL_MULTIPOLES) + "]"
ALL_ABSOLUTE_AS = [
    ANISOTROPIC_POWER_FRACTION_PER_L * TARGET_OMEGA0 / (2 * lval + 1)
    for lval in ALL_MULTIPOLES
]
ALL_ABSOLUTE_AS_STRING = "[" + ",".join("{:.16e}".format(value) for value in ALL_ABSOLUTE_AS) + "]"
LOG10_OMEGA0 = math.log10(TARGET_OMEGA0)
LOG10_A_MIN = math.log10(min(ALL_ABSOLUTE_AS))
LOG10_A_MAX = math.log10(max(ALL_ABSOLUTE_AS))
LOG_OMEGA0_PRIOR = "[{:.1f},{:.1f}]".format(math.floor(LOG10_OMEGA0 - 1.0), math.ceil(LOG10_OMEGA0 + 1.0))
LOG_A_L_PRIOR = "[{:.1f},{:.1f}]".format(math.floor(LOG10_A_MIN - 1.0), math.ceil(LOG10_A_MAX + 1.0))


def iso_only_filename():
    return "params_fixedl_iso_only_from_all_fixedmultipoles_l10.ini"


def iso_plus_one_filename(multipole_l):
    return "params_fixedl_iso_plus_l{multipole_l:02d}_from_all_fixedmultipoles_l10.ini".format(
        multipole_l=multipole_l
    )


def build_iso_only_config():
    return textwrap.dedent(
        f"""\
        [params]
        ## Inject one shared all-multipole absolute-amplitude dataset and recover only L=0.
        ## Run this config first so the L>0 configs can reuse the same cached spectrum.

        fmin=4e-4
        fmax=3e-3
        duration=2e6
        seglen=1e5
        fs=0.25
        Shfile=LISA_2017_PSD_M.npy
        nside=4
        tstart = 0
        lisa_config=orbiting
        tdi_lev=aet

        load_data=0
        datafile=mldc_tdi_withnoise.txt
        datatype=strain
        ## Reference frequency fixed to 25 mHz as requested for the paper benchmark.
        fref = {LISA_FREF:.1e}

        model=noise+powerlaw_isgwb
        alias={{'powerlaw_isgwb':'powerlaw_fixedmultipoles'}}

        lmax = 10
        absolute_multipole_ls = {ALL_ABSOLUTE_LS}


        [inj]
        doInj=1
        injection=noise+powerlaw_fixedmultipoles

        ## Injection benchmark:
        ## log10(h^2 Omega_GW) = -11.30 at 25 mHz, mapped to BLIP's plain Omega convention.
        ## The anisotropic power satisfies (2L+1) A_L = 0.07 Omega_0 for every L=1..10,
        ## so the summed anisotropic power is 0.7 Omega_0 across the ten injected multipoles.
        truevals = {{'noise':{{'Np':9e-42,'Na':3.6e-49}},
                    'powerlaw_fixedmultipoles':{{'alpha':{INJECTION_ALPHA:.1f},
                                                'omega0':{TARGET_OMEGA0:.16e},
                                                'A_ls':{ALL_ABSOLUTE_AS_STRING}}}
                   }}


        [priors]
        log_Np=[-43.0,-40.0]
        log_Na=[-50.0,-47.0]
        alpha=[-1.0,2.5]
        log_omega0={LOG_OMEGA0_PRIOR}


        [run_params]
        sampler=dynesty
        verbose=1
        doPreProc=0
        projection=E
        colormap=magma
        nlive=1000
        Nthreads=1
        FixSeed=1
        seed=42

        input_spectrum={SHARED_SPECTRUM}

        checkpoint=1
        checkpoint_interval=1800
        sample_method=rslice

        out_dir={OUTPUT_ROOT}/recover_l00/
        """
    )


def build_iso_plus_one_config(multipole_l):
    return textwrap.dedent(
        f"""\
        [params]
        ## Reuse one shared all-multipole absolute-amplitude dataset and recover only L=0 plus L={multipole_l}.

        fmin=4e-4
        fmax=3e-3
        duration=2e6
        seglen=1e5
        fs=0.25
        Shfile=LISA_2017_PSD_M.npy
        nside=4
        tstart = 0
        lisa_config=orbiting
        tdi_lev=aet

        load_data=1
        datafile=mldc_tdi_withnoise.txt
        datatype=strain
        fref = {LISA_FREF:.1e}

        model=noise+powerlaw_isgwb+powerlaw_multipole
        alias={{'powerlaw_isgwb':'powerlaw_fixedmultipoles',
                'powerlaw_multipole':'powerlaw_fixedmultipoles'}}

        lmax = 10
        multipole_l = {multipole_l}


        [inj]
        ## Reuse the shared dataset written by {iso_only_filename()}.
        doInj=0


        [priors]
        log_Np=[-43.0,-40.0]
        log_Na=[-50.0,-47.0]
        alpha=[-1.0,2.5]
        log_omega0={LOG_OMEGA0_PRIOR}
        log_A_L={LOG_A_L_PRIOR}


        [run_params]
        sampler=dynesty
        verbose=1
        doPreProc=0
        projection=E
        colormap=magma
        nlive=1000
        Nthreads=1
        FixSeed=1
        seed=42

        input_spectrum={SHARED_SPECTRUM}

        checkpoint=1
        checkpoint_interval=1800
        sample_method=rslice

        out_dir={OUTPUT_ROOT}/recover_l{multipole_l:02d}/
        """
    )


def main():
    iso_only_path = REPO_ROOT / iso_only_filename()
    iso_only_path.write_text(build_iso_only_config())
    print("wrote", iso_only_path)

    for multipole_l in range(1, 11):
        output_path = REPO_ROOT / iso_plus_one_filename(multipole_l)
        output_path.write_text(build_iso_plus_one_config(multipole_l))
        print("wrote", output_path)


if __name__ == "__main__":
    main()
