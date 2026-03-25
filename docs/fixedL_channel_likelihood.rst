Fixed-L Channel Likelihood
==========================

BLIP's dedicated fixed-``L`` channel-resolved mode uses the model strings

``model=noise+powerlaw_fixedLchannels``

``injection=noise+powerlaw_fixedLchannels``

This mode samples exactly

``log10_Ac``, ``alpha``, ``log_Np``, and ``log_Na``.

The signal model is

``Omega_GW^L(f) h^2 = 10^(log10_Ac) * (f / fixedL_fc)^alpha``

with:

- ``fixedL`` selecting the multipole ``L``
- ``fixedL_fc`` defaulting to ``2.5e-3`` Hz

The likelihood is evaluated over the unique A/E/T channel pairs

``AA, EE, TT, AE, AT, ET``

using channel-resolved fixed-``L`` response curves and instrumental-noise curves written in ``Omega h^2`` units.

For BLIP-generated benchmark injections, the example configs use

``fixedL_injection_mode=mean_rmat``

which builds the averaged fixed-``L`` data product directly inside BLIP's native ``LISA`` / ``Injection`` /
``Model`` pipeline so the smoke test exercises the new likelihood against a self-consistent fixed-``L`` benchmark.

Run the smoke example with

``python blip/run_blip fixedL_channel_smoke.ini``

Run the production example with

``python blip/run_blip fixedL_channel_production.ini``

In addition to the posterior samples and corner plot, BLIP writes fixed-``L`` diagnostics to the run directory:

- ``fixedL_signal_omega.npz``
- ``fixedL_channel_response.npz``
- ``fixedL_channel_noise_omega.npz``
- ``fixedL_channel_data.npz``
- ``fixedL_channel_theory.npz``

and corresponding summary plots.
