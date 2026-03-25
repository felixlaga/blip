#  BLIP: Bayesian LISA Inference Package

This is a fully Bayesian Python package for detecting/characterizing stochastic gravitational wave backgrounds and foregrounds with LISA.


1) We recommend creating a dedicated conda environment for BLIP. Conda is a common python virtual environment manager; if you already have Conda, start at step 2; otherwise [install conda.](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

2) Create an environment. We require Python 3.10.0:

`conda create --name blip-env python=3.10.0`


3) Activate it via

`conda activate blip-env`

4) You can now install the package via pip by running

`pip install -e .`

in this directory.

You should now be ready to go! To run BLIP, you only need to provide a configuration file. In this directory, you will find params_default.ini, a pre-constructed config file with reasonable settings and accompanying parameter explanations.

To run, call

`run_blip params_default.ini`

This will (by default) inject and recover a power law isotropic SGWB, with LISA detector noise at the level specified in the LISA proposal (Amaro-Seoane et al., 2017), for 3 months of data.

Two other helpful parameter files are also included: test_params.ini, which has settings ideal for (more) rapid code testing, and minimal_params.ini, which only includes the bare bones, minimal necessary settings for BLIP to run.

Posterior plots will be automatically created in the specified output directory, along with some diagnostics. All statistical model information is saved in Model.pickle; all information used to perform the injection is likewise saved in Injection.pickle. The posterior samples are saved to post_samples.txt.

More details can be found in [the code documentation](https://blip.readthedocs.io/en/latest/).

## Fixed-L Channel Likelihood

BLIP also includes a dedicated fixed-`L` channel-resolved likelihood mode:

`model=noise+powerlaw_fixedLchannels`

`injection=noise+powerlaw_fixedLchannels`

The new mode samples exactly

`log10_Ac, alpha, log_Np, log_Na`

and evaluates a Gaussian likelihood over the unique A/E/T channel pairs `AA, EE, TT, AE, AT, ET` in frequency bins. It is configured with the example files:

`fixedL_channel_smoke.ini`

`fixedL_channel_production.ini`

The included benchmark injections use `fixedL_injection_mode=mean_rmat`, which keeps the new mode inside BLIP's
native `run_blip` / `Injection` / `Model` / sampler flow while constructing a fixed-`L` averaged data product that
matches the dedicated channel likelihood.

The older `powerlaw_multipole` single-multipole mode is still present for legacy/experimental use, but it is not the new fixed-`L` channel likelihood.
