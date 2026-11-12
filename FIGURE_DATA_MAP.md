# zonghe figure map

All figures are regenerated end-to-end (with checkpoint reuse) by:

```bash
cd /mnt/agents/output/edu
python3 zonghe.py --nogui --figures
```

To force a full recompute, delete `checkpoints/` first. Seed is fixed at 42.
All figures are saved at 300 dpi (matplotlib, Agg) into `figures/`.

| File | Generator (zonghe.py) | Line range* | Data sources (checkpoints) | Panels |
|---|---|---|---|---|
| fig1_architecture_gui.png | `fig1_architecture` | 860–984 | none (schematic of workflow + GUI widget tree) | 2 |
| fig2_forward_demo.png | `fig2_forward_demo` | 988–1041 | `truth.npz` (sf, wd, z, shoreline), subsidence from `subsidence_rate_field` (280) | 5 |
| fig3_nonuniqueness.png | `fig3_nonuniqueness` | 1045–1106 | `nonuniq_truth.npz`, `nonuniq_shifted.npz`, `nonuniq_rescaled.npz` via `run_nonuniqueness` (697) | 4 |
| fig4_variational_principle.png | `fig4_variational` | 1110–1158 | `jlandscape.npz` (11×11 J grid), `iter_hist.npz` (GN path), `lam_*.npz` | 2 |
| fig5_iteration_frames.png | `fig5_iterations` | 1162–1203 | `iter_hist.npz` (iterate vectors + J history) | 5 |
| fig6_noise_levels.png | `fig6_noise` | 1207–1254 | `lam_<knee>.npz`, `noise_0.01.npz`, `noise_0.05.npz`, `noise_ensemble.npz` (3 seeds × 3 levels) | 3 |
| fig7_lambda_lcurve.png | `fig7_lambda` | 1258–1340 | `lam_1e-05.npz` … `lam_1000.npz` (9-point λ sweep), knee from `lcurve_knee` (475) | 3 |
| fig8_tikhonov_vs_tv_step.png | `fig8_tikhonov_vs_tv` | 1344–1397 | `step_truth.npz`, `step_tik.npz`, `step_tv.npz` via `run_step_experiment` (768); 25 m step truth, 2-well sparse data | 2 |
| fig9_uncertainty.png | `fig9_uq` | 1401–1453 | `laplace.npz` (400 posterior samples, post_std), `lam_<knee>.npz` | 3 |
| fig10_exercise_workflow.png | `fig10_exercise` | 1457–1521 | `truth.npz`, `lam_<knee>.npz`, `laplace.npz`, λ sweep | 4 |

*Line ranges refer to `zonghe.py` as delivered (1871 lines); all figure
functions are orchestrated by `make_all_figures`.

Supporting sections: forward wrapper `forward` (240), truth generator
`sea_level_truth` (188), observation operator `observe` (290) + pseudo-well
constants, noise model `add_noise` (298), inversion `Inversion` (324) +
`tv_irls_inversion` (426), L-curve `lcurve_knee` (475), Laplace UQ
`laplace_posterior` (496), metrics (312), pipeline `run_experiments` (532),
GUI `SedappInversionGUI`, CLI `main`.
