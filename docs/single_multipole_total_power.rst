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


Single Relative-Multipole Recovery
----------------------------------

The second supported multipole mode keeps one shared isotropic spectrum and
adds one fixed relative multipole amplitude under that same spectral index,

``model=noise+powerlaw_relmultipole``

with

``log10(A_L / A_0)``

sampled for the selected ``multipole_l``.

The covariance is built as

``C_gw = S_gw(f; alpha, log10 Omega_0) [R_iso + (A_L/A_0) R^(L)]``

where ``R^(L)`` is the effective total-power response for the chosen multipole
after marginalizing over its ``m`` modes.

Use ``multipole_l`` in the ``[params]`` section to choose the recovered
multipole. For injected data, provide the shared monopole amplitude through
``omega0`` together with either ``A_ratio`` or ``log_A_ratio`` for the selected
relative multipole.
