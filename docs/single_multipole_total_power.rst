Single-Multipole Total-Power Mode
=================================

This mode adds a fixed-``L`` anisotropic search that infers only one multipole's total power amplitude and slope. It does not sample individual ``b_lm`` coefficients and it does not reconstruct a sky map.

Use the standard BLIP model strings

``model=noise+powerlaw_multipole``

``injection=noise+powerlaw_multipole``

Set ``multipole_l`` in the ``[params]`` section to choose the multipole for the run. If ``[inj] multipole_l`` is omitted, the injection uses the same value.

The sampled parameters in this first-pass mode are

``log_Np``, ``log_Na``, ``alpha``, and ``log_A_L``.

Run the fast example with

``python blip/run_blip params_single_multipole_fast.ini``

Run the production example with

``python blip/run_blip params_single_multipole_production.ini``

BLIP writes the posterior samples to ``post_samples.txt`` in the run directory. The corner plot in ``corners.png`` includes ``alpha`` and the run-specific amplitude label, for example ``log10(A_2)`` when ``multipole_l = 2``.


Shared-Injection Multipole Sweep
--------------------------------

To compare recovery multipoles against the same simulated dataset, enable

``multipole_sweep=1``

and provide a recovery list with

``sweep_multipoles=[0,1,2,...]``

in the ``[run_params]`` section.

In this mode, BLIP generates or loads the dataset once, then runs one fixed-``L`` recovery analysis for each listed multipole against that same data. The injected dataset is still controlled by the usual ``[inj]`` settings, including ``[inj] multipole_l`` for fixed-``L`` injections.

The sweep mode is currently intended for the single fixed-``L`` total-power model, so the recovery model should contain exactly one ``powerlaw_multipole`` component.

Run the paper-style shared-dataset sweep example with

``python blip/run_blip params_single_multipole_paper_sweep.ini``

The sweep writes

``shared_data/``

plus one recovery directory per multipole,

``recover_l00/``, ``recover_l01/``, ...

inside the configured ``out_dir``. It also writes

``multipole_sweep_summary.txt``

and

``multipole_sweep_summary.png``

to summarize the evidences and recovered amplitudes across the tested multipoles.
