# Code and data manifest

This repository snapshot supports the manuscript *SeaVarinv: An Open-Source,
GUI-Driven Forward-Inverse Stratigraphic Simulation Tool for Geoscience
Education*.

## Code

- `zonghe.py` implements the experiment configuration, synthetic sea-level
  truth, Sedapp wrapper, observation and noise models, damped Gauss-Newton and
  total-variation inversions, L-curve selection, Laplace uncertainty
  quantification, GUI, command-line modes, and all ten figure generators.
- `sedapp_py.py` is the Sedapp Python forward-model kernel imported by
  `zonghe.py`.

## Deposited numerical data

The `checkpoints/` directory contains NumPy `.npz` archives generated with the
fixed random seed `SEED = 42`. Their mapping to manuscript figures is described
in `FIGURE_DATA_MAP.md`. The principal groups are:

- reference forward simulation: `truth.npz`;
- non-uniqueness experiment: `nonuniq_*.npz`;
- regularisation-weight sweep: `lam_*.npz`;
- objective landscape and iteration history: `jlandscape.npz` and
  `iter_hist.npz`;
- noise experiment: `noise_*.npz` and `noise_ensemble.npz`;
- step-recovery experiment: `step_*.npz` and `tv_irls.npz`;
- uncertainty experiment: `laplace.npz`.

## Generated figures

The `figures/` directory contains the ten 300-dpi PNG files used in the
manuscript. These are derived outputs; the source code and checkpoint archives
are the reproducibility inputs.

## Integrity and provenance

`SHA256SUMS.txt` lists SHA-256 hashes for all deposited files except itself.
All stochastic results use seed 42. The full workflow can be regenerated with
`python zonghe.py --nogui --figures`; `python zonghe.py --smoketest` performs a
fast logic check.
