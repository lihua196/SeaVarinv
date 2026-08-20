# SeaVarinv — A Forward–Inverse Stratigraphic Modelling Teaching Tool

Repository: https://github.com/joj656582-bot/SeaVarinv

`zonghe.py` is a single self-contained script that packages a complete
computational-stratigraphy workflow for the classroom:

* a **forward model** — Sedapp 1-D finite-volume diffusion stratigraphy
  (`sedapp_py.py`, copied next to the script);
* a **synthetic truth generator** — parameterised relative sea-level curves
  (sinusoids + trend + optional abrupt step);
* an **observation operator** — water depth sampled at 8 pseudo-wells;
* a **noise model** — seeded additive Gaussian noise;
* a **variational inversion** — damped Gauss–Newton (Levenberg–Marquardt)
  with a finite-difference Jacobian, Tikhonov (H¹) or total-variation (IRLS)
  regularisation;
* **L-curve λ selection**, **Laplace uncertainty quantification**, metrics
  (RMSE, Pearson r);
* a **tkinter GUI** with live matplotlib canvases;
* a **figure factory** producing 10 education figures at 300 dpi.

## Install / dependencies

Python ≥ 3.9 with:

```bash
pip install numpy scipy matplotlib   # tkinter ships with CPython (python3-tk on Debian/Ubuntu)
```

No other dependencies. `sedapp_py.py` must sit in the same folder as
`zonghe.py` (it is imported directly).

For a reproducible installation, use:

```bash
python -m pip install -r requirements.txt
```

## GUI usage

```bash
python3 zonghe.py            # opens the GUI (needs a display)
```

The left control panel exposes the full experiment configuration:

| control | meaning |
|---|---|
| Amplitude 1/2, Period 1/2 | two sinusoids of the sea-level truth e(t) |
| Trend | linear sea-level trend (m, peak-to-peak) |
| Step height | abrupt mid-run sea-level step (m) — for the Tikhonov-vs-TV lesson |
| Noise level | σ = fraction × std of the clean data |
| lambda | Tikhonov regularisation weight |
| No. of knots | piecewise-linear control knots (6–20) |

Buttons: **Run Forward** (Sedapp, ~0.5 s), **Generate Observations** (seeded
noise), **Run Inversion** (Gauss–Newton), **L-curve** (quick λ sweep),
**Reset**. Long runs execute in a worker thread — the GUI never freezes, and
the two canvases (sea-level curve; basin fill / L-curve) update live when a
run finishes.

## CLI usage (headless)

```bash
python3 zonghe.py --nogui --figures   # full pipeline + all 10 figures (300 dpi)
python3 zonghe.py --nogui             # pipeline only (uses checkpoints)
python3 zonghe.py --smoketest         # headless GUI/logic smoke test
```

Checkpoints (`.npz`) accumulate under `checkpoints/`; deleting them forces a
full recompute (~15–20 min from scratch; <1 min with checkpoints).

## Pedagogy notes — learning objectives mapped to figures

| Figure | Learning objective | Suggested exercise |
|---|---|---|
| fig1 architecture/GUI | see the whole workflow as data flow: forcing → stratigraphy → data → inversion | trace one loop iteration on the diagram |
| fig2 forward demo | understand the forward map F: sea level + subsidence + supply → basin fill | GUI: change amplitude, predict the change in the fill, then run |
| fig3 non-uniqueness | grasp why the inverse problem is ill-posed | find a third curve by hand that leaves the wells nearly unchanged |
| fig4 variational principle | J = misfit + λ·roughness; optimisation walks downhill on J | GUI: run inversion from different initial guesses |
| fig5 iteration frames | watch Gauss–Newton converge; connect curve evolution to J decrease | count forward calls per iteration (Jacobian cost!) |
| fig6 noise levels | more noise → worse recovery, but with diminishing returns past the misfit floor | GUI: noise slider 1%→5%, record RMSE |
| fig7 λ / L-curve | regularisation weight trades fit against smoothness; the knee is objective | GUI: L-curve button, then set λ manually |
| fig8 Tikhonov vs TV | regulariser choice encodes prior belief; with sparse data (2 wells), TV preserves the 25 m step (RMSE 3.35 m) while H¹ smears it (7.41 m); with the full 8 wells both coincide — the data, not the prior, then dominate | set Step height = 25 m and invert with both; discuss when the prior matters |
| fig9 uncertainty | a MAP estimate without error bars is only half an answer; linearised UQ can be optimistic | check the coverage diagnostic — why is it < 95%? |
| fig10 exercise workflow | a complete 45-min lab session script | follow stages 1–4 in the GUI |

## Reproducibility checklist

* **Seed**: `SEED = 42` (`np.random.default_rng`, fresh per noise call — order-independent).
* **Versions used**: Python 3.x, numpy 2.2.5, scipy, matplotlib (Agg for CLI).
* **Forward config**: L = 15 km, nx = 120, dt = 4 Ma, t1 = 200 Ma (50 steps),
  8 pseudo-wells (2–13 km), 12 knots, FD step 3 m, bounds ±100 m.
* **Regenerate everything**: `rm -rf checkpoints figures && python3 zonghe.py
  --nogui --figures`.
* **Checkpoints**: every inversion / forward run / UQ ensemble is stored in
  `checkpoints/*.npz`; reruns skip finished compute.

## Repository contents

* `zonghe.py`: GUI, command-line workflow, inversion, uncertainty analysis,
  and figure factory.
* `sedapp_py.py`: one-dimensional Sedapp forward-model kernel.
* `checkpoints/`: precomputed NumPy archives underlying the quantitative
  results and Figures 2–10.
* `figures/`: the ten manuscript figures generated by the workflow.
* `FIGURE_DATA_MAP.md`: figure-to-script and figure-to-checkpoint mapping.
* `SHA256SUMS.txt`: integrity checksums for the deposited files.

The smoke test was verified on 20 August 2026 with Python 3, NumPy 1.26.4,
SciPy 1.11.4, and Matplotlib 3.8.0. The manuscript reference environment is
Python 3.12.12, NumPy 2.2.5, SciPy 1.16.2, and Matplotlib 3.10.3.

## Licence status

No software or data licence was supplied with the source materials used to
prepare this manuscript snapshot. Copyright therefore remains with the
respective authors and original Sedapp contributors. Users should obtain the
authors' permission before redistribution or reuse beyond inspection and
reproduction of the reported results.

