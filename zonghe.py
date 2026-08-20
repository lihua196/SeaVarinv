#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zonghe.py — Consolidated forward–inverse stratigraphic modelling teaching tool
==============================================================================

A single self-contained script that packages a complete computational
stratigraphy workflow for EDUCATION:

    (a) forward model wrapper      -> Sedapp 1-D finite-volume diffusion model
                                      (``sedapp_py.run_sedapp``, copied next to
                                      this file and imported from the same folder)
    (b) synthetic truth generator  -> parameterised sea-level curves
                                      (sinusoids + trend, optional abrupt step)
    (c) observation operator       -> water depth sampled at 8 pseudo-wells
    (d) noise model                -> additive Gaussian noise, seeded rng
    (e) variational inversion      -> J = misfit + regularisation, FD Jacobian,
                                      damped Gauss-Newton (Levenberg-Marquardt),
                                      Tikhonov (H1) and TV (IRLS) regularisers
    (f) L-curve lambda selection   -> maximum-curvature knee of the L-curve
    (g) uncertainty quantification -> Laplace (linearised Gaussian posterior)
    (h) metrics                    -> RMSE and Pearson r against the truth
    (i) plotting module            -> ALL education figures (fig1 ... fig10)
    (j) GUI                        -> tkinter control panel + live matplotlib
                                      canvas (threaded so the GUI never freezes)

Reproducibility
---------------
* Fixed seed SEED = 42 everywhere (``numpy.random.default_rng``).
* All expensive results are checkpointed as .npz files under
  ``edu/checkpoints/``; re-running with the same configuration skips compute.
* ``python3 zonghe.py --nogui --figures`` regenerates every figure in
  ``edu/figures/`` at 300 dpi with the Agg backend (no display needed).
* ``python3 zonghe.py --smoketest`` checks that the GUI class and the core
  logic instantiate headlessly (no X display required).

Forward-model configuration is deliberately modest so that one Sedapp call
takes ~0.5 s: L = 15 km, nx = 120 cells, dt = 4 Ma, t1 = 200 Ma (50 steps).

Authors: consolidated from ``sedapp_py.py`` (Sedapp V1.12 Python port,
original R by Jingzhe Li) and ``inversion.py`` (variational sea-level
inversion).  See zonghe_README.md for usage and pedagogy notes.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np

import matplotlib
# Agg is selected only in headless mode (see main()); the GUI switches to TkAgg.
if os.environ.get('ZONGHE_GUI', '0') != '1':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sedapp_py import run_sedapp, phi_mixed, jet_colors, best_color_function, heng_sum

SEED = 42
FIGDIR = os.path.join(HERE, 'figures')
CKPTDIR = os.path.join(HERE, 'checkpoints')
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(CKPTDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 8.5, 'figure.dpi': 110, 'savefig.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.3, 'font.family': 'DejaVu Sans',
})


def log(msg):
    """Timestamped log line (flushed so GUI threads stream progress)."""
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
# Fixed forward-model configuration (shared by GUI and CLI).
L = 15000.0     # domain length (m)
DT = 4.0        # time step (Ma)
T1 = 200.0      # total time (Ma)
NX = 120        # finite-volume cells
K = int(np.ceil(T1 / DT))              # 50 time steps
TIME = np.arange(1, K + 1) * DT        # step times (Ma), matches sl_series

# Initial topography control points (from example_run.py)
TOPO = np.array([[0, 9.0], [1890, 0.0], [3900, -86.0], [L, -6.8]])

# Tectonic subsidence control points (example_run.py), rescaled by t1/400 so
# the *rate* (m/Ma) matches the original 400 Ma teaching example.
SUBS = np.array([
    [0, 0.0], [1890.03, 0.0], [2044.67, 28.0], [2216.49, 441.7],
    [2405.5, 722.0], [2731.96, 1107.0], [3250, 1409.0], [3402.06, 1530.0],
    [3436.43, 1537.0], [4673.5, 1530.0], [5910.65, 1373.0],
    [7285.22, 1239.0], [8556.7, 1000.0], [11271.5, 530.0],
    [13505.2, 77.0], [14003.4, 0.0], [15000, 0.0],
])
SUBS = SUBS.copy()
SUBS[:, 1] *= T1 / 400.0

BASE_PARAMS = dict(
    nx=NX, L=L, t=0.0, t1=T1, dt=DT,
    topo_samples=TOPO, subsidence_samples=SUBS,
    df_dco=168e6, df_EA=968e2, df_zhishu=2.0, env_energy=100.0,
    df_wdscale_sand=0.1, df_wdscale_mud=0.075,
    expected_sedimentation_sand=0.135, expected_sedimentation_mud=0.675,
    dep_ero_ratio=1e6, overlying_layer_thickness=0.0, color_mode=2,
)

# Pseudo-well positions: 8 wells spread across the basin (observation operator)
X_VALUE = np.linspace(L / (2 * NX), L - L / (2 * NX), NX)
WELL_X = np.linspace(2000.0, 13000.0, 8)
IW = np.array([np.argmin(np.abs(X_VALUE - w)) for w in WELL_X])
N_WELL = len(IW)
N_DATA = N_WELL * K                     # 8 wells x 50 steps = 400 data

FD_EPS = 3.0      # finite-difference Jacobian step (m); bridges plateaus
E_BOUNDS = (-100.0, 100.0)              # box bounds on every knot value (m)

FORWARD_STATS = {'calls': 0, 'time': 0.0}   # global runtime bookkeeping


@dataclass
class Config:
    """User-adjustable experiment configuration (mirrors the GUI controls).

    Attributes
    ----------
    amp1, amp2 : float
        Amplitudes (m) of the long- and short-period sea-level sinusoids.
    period1, period2 : float
        Periods (Ma) of the two sinusoids.
    trend : float
        Peak-to-peak linear trend (m) across the model run.
    step_height : float
        Height (m) of an abrupt mid-run sea-level step (0 = smooth curve).
    noise : float
        Noise level as a fraction of the clean-data standard deviation.
    lam : float
        Tikhonov regularisation weight lambda.
    n_knots : int
        Number of piecewise-linear sea-level control knots.
    well_idx : tuple of int or None
        Subset of pseudo-well indices to use as the observation operator
        (None = all 8 wells).  Sparse-well settings make the regularisation
        carry more weight — used by the Tikhonov-vs-TV step experiment.
    """
    amp1: float = 22.0
    period1: float = 85.0
    amp2: float = 10.0
    period2: float = 34.0
    trend: float = 18.0
    step_height: float = 0.0
    noise: float = 0.03
    lam: float = 1.0
    n_knots: int = 12
    well_idx: tuple = None

    def signature(self):
        """Short string uniquely identifying this configuration (for ckpts)."""
        return (f'a{self.amp1:g}_p{self.period1:g}_b{self.amp2:g}'
                f'_q{self.period2:g}_tr{self.trend:g}_st{self.step_height:g}'
                f'_n{self.noise:g}_l{self.lam:g}_k{self.n_knots}')


DEFAULT_CFG = Config()


# ═════════════════════════════════════════════════════════════════════════════
# (b) SYNTHETIC TRUTH GENERATOR — parameterised sea-level curves
# ═════════════════════════════════════════════════════════════════════════════
def sea_level_truth(t, cfg=DEFAULT_CFG):
    """Synthetic relative sea-level curve e_true(t)  [m].

    e(t) = amp1 * sin(2 pi t / period1)
         + amp2 * sin(2 pi t / period2 + 0.7)
         + trend * (t / T1 - 0.5)
         + step_height * H(t - T1/2)

    where H is the Heaviside function: setting ``step_height`` adds an abrupt
    mid-run step (used to teach the Tikhonov-vs-TV regularisation choice).
    """
    t = np.asarray(t, dtype=float)
    e = (cfg.amp1 * np.sin(2 * np.pi * t / cfg.period1)
         + cfg.amp2 * np.sin(2 * np.pi * t / cfg.period2 + 0.7)
         + cfg.trend * (t / T1 - 0.5))
    if cfg.step_height != 0.0:
        e = e + cfg.step_height * (t >= T1 / 2.0)
    return e


# ═════════════════════════════════════════════════════════════════════════════
# (e-support) KNOT PARAMETERISATION AND ROUGHNESS MATRIX
# ═════════════════════════════════════════════════════════════════════════════
def knot_times(cfg=DEFAULT_CFG):
    """Time positions of the piecewise-linear control knots (Ma)."""
    return np.linspace(TIME[0], TIME[-1], cfg.n_knots)


def knots_to_sl(e_knots, cfg=DEFAULT_CFG):
    """Interpolate knot values to the model step times (piecewise linear)."""
    return np.interp(TIME, knot_times(cfg), np.asarray(e_knots, dtype=float))


def roughness_matrix(cfg=DEFAULT_CFG):
    """Matrix R with  integral (de/dt)^2 dt = e^T R e  for piecewise-linear e."""
    n = cfg.n_knots
    D = np.diff(np.eye(n), axis=0)          # (n-1, n) first-difference operator
    dt_k = np.diff(knot_times(cfg))         # (n-1,)
    W = np.diag(1.0 / dt_k)
    return D.T @ W @ D


def roughness(e, cfg=DEFAULT_CFG):
    """Roughness functional  integral (de/dt)^2 dt  of a knot vector."""
    R = roughness_matrix(cfg)
    e = np.asarray(e, dtype=float)
    return float(e @ R @ e)


# ═════════════════════════════════════════════════════════════════════════════
# (a) FORWARD MODEL WRAPPER
# ═════════════════════════════════════════════════════════════════════════════
def forward(e_knots, cfg=DEFAULT_CFG, keep_full=False, quiet=True):
    """Run the Sedapp forward model for a knot vector.

    Parameters
    ----------
    e_knots : array-like, length cfg.n_knots
        Sea-level control-knot values (m).
    cfg : Config
        Experiment configuration.
    keep_full : bool
        If True, also return the full Sedapp result dictionary.
    quiet : bool
        Sedapp prints one line per time step; suppress it by default.

    Returns
    -------
    d : ndarray, shape (N_DATA,)
        Observable vector: water depth at the pseudo-wells, all time steps
        (flattened as (well, time)).
    res : dict, optional
        Full Sedapp output when ``keep_full=True``.
    """
    p = dict(BASE_PARAMS)
    p['sl_series'] = knots_to_sl(e_knots, cfg)
    iw = IW if cfg.well_idx is None else IW[np.asarray(cfg.well_idx)]
    t0 = time.time()
    if quiet:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = run_sedapp(p)
    else:
        res = run_sedapp(p)
    FORWARD_STATS['calls'] += 1
    FORWARD_STATS['time'] += time.time() - t0
    d = res['wd_matrix'][iw, :].ravel()
    if keep_full:
        return d, res
    return d


def subsidence_rate_field():
    """Tectonic subsidence-rate field DS(x) in m/Ma (as used inside Sedapp)."""
    sub = SUBS.copy()
    sub[:, 1] = sub[:, 1] * 0.68 * (4.0 / 3.0)     # compaction correction
    return np.interp(X_VALUE, sub[:, 0], sub[:, 1]) / T1


# ═════════════════════════════════════════════════════════════════════════════
# (c) OBSERVATION OPERATOR  +  (d) NOISE MODEL
# ═════════════════════════════════════════════════════════════════════════════
def observe(res):
    """Observation operator: water depth at the pseudo-wells from a Sedapp run.

    Returns an (N_WELL, K) matrix.
    """
    return res['wd_matrix'][IW, :]


def add_noise(d_clean, frac, seed=SEED):
    """Additive Gaussian noise with sigma = frac * std(d_clean).

    A fresh ``np.random.default_rng(seed)`` is created per call so the noise
    realisation is reproducible regardless of call order.
    """
    rng = np.random.default_rng(seed)
    sigma = frac * float(np.std(d_clean))
    return d_clean + rng.normal(0.0, sigma, d_clean.shape), sigma


# ═════════════════════════════════════════════════════════════════════════════
# (h) METRICS
# ═════════════════════════════════════════════════════════════════════════════
def metrics(e_inv, e_true_knots):
    """RMSE (m) and Pearson correlation r between inverted and true knots."""
    e_inv = np.asarray(e_inv, dtype=float)
    e_true_knots = np.asarray(e_true_knots, dtype=float)
    rmse = float(np.sqrt(np.mean((e_inv - e_true_knots) ** 2)))
    r = float(np.corrcoef(e_inv, e_true_knots)[0, 1])
    return rmse, r


# ═════════════════════════════════════════════════════════════════════════════
# (e) VARIATIONAL INVERSION — objective, FD Jacobian, LM solver, Tikhonov/TV
# ═════════════════════════════════════════════════════════════════════════════
class Inversion:
    """Variational sea-level inversion.

    Minimises

        J(e) = 1/2 ||(F(e) - d_obs)/sigma||^2  +  lam/2 * e^T R e

    with a damped Gauss-Newton (Levenberg-Marquardt) scheme using a forward
    finite-difference Jacobian.  Passing a reweighted ``Rmat`` implements
    total-variation regularisation via IRLS (see ``tv_irls_inversion``).

    Attributes
    ----------
    history : list of (n_forward_calls, J, misfit, reg)
        One entry per accepted iteration.
    iterates : list of ndarray
        Knot vector at every accepted iteration (for teaching animations).
    """

    def __init__(self, d_obs, sigma, lam, cfg=DEFAULT_CFG, Rmat=None,
                 tag='inv', maxiter=14, callback=None):
        self.d_obs = np.asarray(d_obs, dtype=float)
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.cfg = cfg
        self.R = roughness_matrix(cfg) if Rmat is None else Rmat
        self.tag = tag
        self.maxiter = int(maxiter)
        self.callback = callback          # optional GUI progress hook
        self.history = []
        self.iterates = []
        self.n_data = len(self.d_obs)

    def jacobian(self, e, F0):
        """Whitened forward-FD Jacobian G (n_data x n_knots): d(F/sigma)/de_j."""
        n = self.cfg.n_knots
        G = np.empty((self.n_data, n))
        for j in range(n):
            ep = e.copy()
            ep[j] += FD_EPS
            G[:, j] = (forward(ep, self.cfg) - F0) / (FD_EPS * self.sigma)
        return G

    def solve(self, e0=None):
        """Run damped Gauss-Newton from ``e0`` (default: constant 0 m)."""
        t0 = time.time()
        e = np.zeros(self.cfg.n_knots) if e0 is None else np.asarray(e0, float).copy()
        F0 = forward(e, self.cfg)
        r = (F0 - self.d_obs) / self.sigma
        misfit = 0.5 * r @ r
        reg = 0.5 * self.lam * (e @ self.R @ e)
        J = misfit + reg
        mu = 1e-2
        self.history.append((FORWARD_STATS['calls'], J, misfit, reg))
        self.iterates.append(e.copy())
        nit = 0
        for it in range(self.maxiter):
            G = self.jacobian(e, F0)
            A = G.T @ G + self.lam * self.R
            g = G.T @ r + self.lam * (self.R @ e)
            accepted = False
            for _try in range(6):
                M = A + mu * np.diag(np.maximum(np.diag(A), 1e-12))
                try:
                    delta = np.linalg.solve(M, -g)
                except np.linalg.LinAlgError:
                    mu *= 10.0
                    continue
                e_trial = np.clip(e + delta, E_BOUNDS[0], E_BOUNDS[1])
                Ft = forward(e_trial, self.cfg)
                rt = (Ft - self.d_obs) / self.sigma
                misfit_t = 0.5 * rt @ rt
                reg_t = 0.5 * self.lam * (e_trial @ self.R @ e_trial)
                J_t = misfit_t + reg_t
                if J_t < J:
                    accepted = True
                    break
                mu *= 8.0
            nit = it + 1
            if not accepted:
                log(f'    [{self.tag}] GN stalled at iter {it} (mu={mu:.1e})')
                break
            rel = (J - J_t) / max(abs(J), 1e-30)
            e, F0, r = e_trial, Ft, rt
            J, misfit, reg = J_t, misfit_t, reg_t
            mu = max(mu * 0.4, 1e-8)
            self.history.append((FORWARD_STATS['calls'], J, misfit, reg))
            self.iterates.append(e.copy())
            if self.callback is not None:
                self.callback(it + 1, e.copy(), J, misfit, reg)
            if rel < 1e-5:
                break

        class Out:
            pass
        out = Out()
        out.x, out.fun, out.nit, out.success = e, J, nit, True
        out.misfit, out.reg = misfit, reg
        out.runtime = time.time() - t0
        return out


def tv_irls_inversion(d_obs, sigma, lam, cfg=DEFAULT_CFG, n_outer=4,
                      maxiter=8, beta=1.0, tag='tv'):
    """Total-variation inversion via iteratively reweighted least squares.

    The TV functional  sum_i sqrt((de_i/dt)^2 + beta)  is approximated at each
    outer iteration by a weighted Tikhonov functional with
    w_i = 1/sqrt((de_i/dt)^2 + beta), which keeps each inner problem quadratic
    so the same LM solver applies unchanged.
    """
    e_tv = np.zeros(cfg.n_knots)
    hist = None
    kt = knot_times(cfg)
    D = np.diff(np.eye(cfg.n_knots), axis=0)
    for outer in range(n_outer):
        de = np.diff(e_tv) / np.diff(kt)
        w = 1.0 / np.sqrt(de ** 2 + beta)
        R_tv = D.T @ np.diag(w / np.diff(kt)) @ D
        inv = Inversion(d_obs, sigma, lam, cfg=cfg, Rmat=R_tv,
                        tag=f'{tag}_{outer}', maxiter=maxiter)
        out = inv.solve(e_tv)
        e_tv = out.x
        hist = np.array(inv.history)
        log(f'  IRLS outer {outer}: TV={np.sum(np.sqrt(np.diff(e_tv)**2 / np.diff(kt) + beta)):.1f}')
    return e_tv, hist


# ═════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING — .npz files under edu/checkpoints/; reruns skip compute
# ═════════════════════════════════════════════════════════════════════════════
def ckpt_path(tag):
    return os.path.join(CKPTDIR, f'{tag}.npz')


def save_ckpt(tag, **kw):
    """Save a checkpoint atomically (write-then-rename)."""
    tmp = ckpt_path(tag) + '.tmp.npz'
    np.savez(tmp, **kw)
    os.replace(tmp, ckpt_path(tag))


def load_ckpt(tag):
    """Load a checkpoint, or None if it does not exist."""
    p = ckpt_path(tag)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


# ═════════════════════════════════════════════════════════════════════════════
# (f) L-CURVE LAMBDA SELECTION
# ═════════════════════════════════════════════════════════════════════════════
def lcurve_knee(lambdas, misfits, roughs):
    """Pick lambda at the L-curve knee (max curvature of log-log curve).

    Returns (lambda_knee, index_into_lambdas).  Only interior points are
    considered so that endpoints never win by default.
    """
    order = np.argsort(np.log10(roughs))
    xs = np.log10(np.asarray(roughs))[order]
    ys = np.log10(np.asarray(misfits))[order]
    d1 = np.gradient(ys, xs)
    d2 = np.gradient(d1, xs)
    curv = np.abs(d2) / (1 + d1 ** 2) ** 1.5
    i_best = int(np.argmax(curv[1:-1]) + 1)
    lam = np.asarray(lambdas)[order][i_best]
    idx = list(lambdas).index(lam)
    return lam, idx


# ═════════════════════════════════════════════════════════════════════════════
# (g) UNCERTAINTY QUANTIFICATION — Laplace approximation
# ═════════════════════════════════════════════════════════════════════════════
def laplace_posterior(e_opt, d_obs, sigma, lam, misfit, cfg=DEFAULT_CFG,
                      n_samples=400, seed=SEED):
    """Linearised Gaussian (Laplace) posterior around an optimum.

    H = G^T G / sigma^2 + lam R  (Gauss-Newton Hessian),  C = chi2_red * H^-1

    The covariance is inflated by the reduced chi^2 (misfit / (n_data/2)) to
    account for the discretisation-error misfit floor; without it the band is
    badly overconfident — itself a useful teaching point.

    Returns (samples, post_std) with samples of shape (n_samples, n_knots).
    """
    n = cfg.n_knots
    F0 = forward(e_opt, cfg)
    G = np.empty((N_DATA, n))
    for j in range(n):
        ep = e_opt.copy()
        ep[j] += FD_EPS
        G[:, j] = (forward(ep, cfg) - F0) / FD_EPS
    Gw = G / sigma
    H = Gw.T @ Gw + lam * roughness_matrix(cfg)
    C = np.linalg.inv(H)
    chi2_red = misfit / (len(d_obs) / 2.0)
    C = C * chi2_red
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(e_opt, C, size=n_samples)
    return samples, np.sqrt(np.diag(C)), chi2_red


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER — produces every datum the figures need (checkpointed)
# ═════════════════════════════════════════════════════════════════════════════
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
NOISE_LEVELS = [0.01, 0.03, 0.05]


def run_experiments(cfg=DEFAULT_CFG, progress=log):
    """Run (or load from checkpoint) the full teaching experiment suite.

    Returns a dictionary with everything the plotting module needs.  Every
    expensive piece is checkpointed, so a second call is nearly instantaneous.
    """
    R = dict(cfg=cfg)
    kt = knot_times(cfg)
    e_true_knots = sea_level_truth(kt, cfg)

    # ── Truth forward run ────────────────────────────────────────────────────
    ck = load_ckpt('truth')
    if ck is None:
        progress('TRUTH forward run ...')
        d_clean, res_truth = forward(e_true_knots, cfg, keep_full=True)
        save_ckpt('truth', d_clean=d_clean,
                  sf=res_truth['sand_fraction_matrix'],
                  wd=res_truth['wd_matrix'],
                  z=res_truth['z_matrix'],
                  por=res_truth['porosity_matrix'],
                  lt=res_truth['layer_thickness_matrix'],
                  shoreline=res_truth['shoreline_x'],
                  x_value=res_truth['x_value'],
                  time=res_truth['time_series'])
    else:
        d_clean = ck['d_clean']
    R['truth_ck'] = load_ckpt('truth')
    data_std = float(np.std(d_clean))
    R['data_std'] = data_std

    # Seeded observations at each noise level
    d_obs, sigmas = {}, {}
    for f in NOISE_LEVELS:
        d_obs[f], sigmas[f] = add_noise(d_clean, f, seed=SEED)
    R['d_obs'], R['sigmas'], R['d_clean'] = d_obs, sigmas, d_clean
    R['e_true_knots'] = e_true_knots

    # ── Lambda sweep (3% noise), cold-started per lambda ────────────────────
    lam_results = {}
    for lam in LAMBDAS:
        tag = f'lam_{lam:g}'
        ck = load_ckpt(tag)
        if ck is None:
            progress(f'lambda sweep: lam={lam:g} ...')
            inv = Inversion(d_obs[0.03], sigmas[0.03], lam, cfg=cfg, tag=tag,
                            maxiter=14)
            out = inv.solve()
            save_ckpt(tag, e_opt=out.x, J=out.fun, hist=np.array(inv.history),
                      iterates=np.array(inv.iterates), runtime=out.runtime,
                      nit=out.nit, misfit=out.misfit, reg=out.reg)
            ck = load_ckpt(tag)
        rmse, r = metrics(ck['e_opt'], e_true_knots)
        lam_results[lam] = dict(e=ck['e_opt'], J=float(ck['J']),
                                hist=ck['hist'], runtime=float(ck['runtime']),
                                misfit=float(ck['hist'][-1, 2]),
                                reg=float(ck['hist'][-1, 3]),
                                rmse=rmse, corr=r)
        progress(f'  lam={lam:g}: J={ck["J"]:.1f} rmse={rmse:.2f} m r={r:.3f}')
    R['lam_results'] = lam_results

    misf = np.array([lam_results[l]['misfit'] for l in LAMBDAS])
    rough = np.array([roughness(lam_results[l]['e'], cfg) for l in LAMBDAS])
    lam_knee, i_knee = lcurve_knee(LAMBDAS, misf, rough)
    lam_minrmse = LAMBDAS[int(np.argmin([lam_results[l]['rmse']
                                         for l in LAMBDAS]))]
    progress(f'L-curve knee lambda = {lam_knee:g}; min-RMSE lambda = {lam_minrmse:g}')
    R['best_lam'], R['lam_minrmse'] = lam_knee, lam_minrmse
    best = lam_results[lam_knee]
    R['best'] = best

    # ── Noise levels 1 % and 5 % (3 % already done) ─────────────────────────
    noise_results = {0.03: dict(e=best['e'], rmse=best['rmse'],
                                corr=best['corr'], hist=best['hist'])}
    for f in (0.01, 0.05):
        tag = f'noise_{f:g}'
        ck = load_ckpt(tag)
        if ck is None:
            progress(f'noise variant f={f:g} ...')
            inv = Inversion(d_obs[f], sigmas[f], lam_knee, cfg=cfg, tag=tag,
                            maxiter=12)
            out = inv.solve()
            save_ckpt(tag, e_opt=out.x, J=out.fun, hist=np.array(inv.history),
                      runtime=out.runtime)
            ck = load_ckpt(tag)
        rmse, r = metrics(ck['e_opt'], e_true_knots)
        noise_results[f] = dict(e=ck['e_opt'], rmse=rmse, corr=r,
                                hist=ck['hist'])
        progress(f'  noise {f:g}: rmse={rmse:.2f} m r={r:.3f}')
    R['noise_results'] = noise_results

    # ── Multiple noise realisations (for error bars in fig6) ────────────────
    ens_tag = 'noise_ensemble'
    ck = load_ckpt(ens_tag)
    if ck is None:
        progress('Noise-realisation ensemble (3 levels x 3 seeds) ...')
        rows = []
        for f in NOISE_LEVELS:
            for iseed in range(3):
                d_i, s_i = add_noise(d_clean, f, seed=SEED + 101 + iseed)
                inv = Inversion(d_i, s_i, lam_knee, cfg=cfg,
                                tag=f'ens_{f}_{iseed}', maxiter=8)
                out = inv.solve()
                rmse_i, r_i = metrics(out.x, e_true_knots)
                rows.append([f, iseed, rmse_i, r_i])
                progress(f'    f={f:g} seed#{iseed}: rmse={rmse_i:.2f} r={r_i:.3f}')
        save_ckpt(ens_tag, rows=np.array(rows))
        ck = load_ckpt(ens_tag)
    R['noise_ensemble'] = ck['rows']

    # ── Tikhonov vs TV on the SMOOTH truth (3% noise) ───────────────────────
    tag = 'tv_irls'
    ck = load_ckpt(tag)
    if ck is None:
        progress('TV inversion (IRLS, smooth truth) ...')
        e_tv, hist = tv_irls_inversion(d_obs[0.03], sigmas[0.03], lam_knee,
                                       cfg=cfg, tag='tv')
        save_ckpt(tag, e_opt=e_tv, hist=hist)
        ck = load_ckpt(tag)
    R['e_tv'] = ck['e_opt']
    R['tv_metrics'] = metrics(ck['e_opt'], e_true_knots)

    # ── Iteration history WITH iterates (for fig4 path and fig5 frames) ─────
    tag = 'iter_hist'
    ck = load_ckpt(tag)
    if ck is None:
        progress('Recording iteration history (iterate vectors) ...')
        inv = Inversion(d_obs[0.03], sigmas[0.03], lam_knee, cfg=cfg, tag=tag,
                        maxiter=14)
        out = inv.solve()
        save_ckpt(tag, iterates=np.array(inv.iterates),
                  hist=np.array(inv.history), e_opt=out.x)
        ck = load_ckpt(tag)
    R['iterates'], R['iter_hist'] = ck['iterates'], ck['hist']

    # ── Laplace approximation at the best optimum ───────────────────────────
    tag = 'laplace'
    ck = load_ckpt(tag)
    if ck is None:
        progress('Laplace approximation: Jacobian at optimum ...')
        samples, post_std, chi2 = laplace_posterior(
            best['e'], d_obs[0.03], sigmas[0.03], lam_knee,
            best['misfit'], cfg=cfg)
        save_ckpt(tag, samples=samples, post_std=post_std, chi2=chi2)
        ck = load_ckpt(tag)
    R['samples'] = ck['samples']
    R['post_std'] = ck['post_std']
    R['chi2_red'] = float(ck['chi2']) if 'chi2' in ck.files else np.nan
    lo = np.percentile(ck['samples'], 2.5, axis=0)
    hi = np.percentile(ck['samples'], 97.5, axis=0)
    R['lo'], R['hi'] = lo, hi
    R['coverage'] = float(np.mean((e_true_knots >= lo) & (e_true_knots <= hi)))
    progress(f'Laplace 95% band coverage of truth: {R["coverage"]*100:.0f}% of knots')

    # ── Non-uniqueness demo: different e(t), similar stratigraphy ───────────
    R['nonuniq'] = run_nonuniqueness(cfg, progress)

    # ── J landscape over two knot values (fig4) ─────────────────────────────
    R['jlandscape'] = run_jlandscape(d_obs[0.03], sigmas[0.03], lam_knee,
                                     cfg, progress)

    # ── Step-truth experiment: Tikhonov vs TV on an abrupt step (fig8) ──────
    R['step'] = run_step_experiment(progress)
    return R


def run_nonuniqueness(cfg, progress=log):
    """Forward runs with 3 different sea-level curves whose stratigraphic
    outputs are nearly identical: the core non-uniqueness teaching point."""
    kt = knot_times(cfg)
    base = sea_level_truth(kt, cfg)
    variants = {
        'truth': base,
        'shifted': base + 6.0 * np.sin(2 * np.pi * kt / 55.0 + 1.3),
        'rescaled': 1.08 * base - 4.0,
    }
    out = {'knots': {k: v for k, v in variants.items()}}
    for name, e in variants.items():
        tag = f'nonuniq_{name}'
        ck = load_ckpt(tag)
        if ck is None:
            progress(f'non-uniqueness forward run: {name} ...')
            d, res = forward(e, cfg, keep_full=True)
            save_ckpt(tag, d=d, z=res['z_matrix'], sf=res['sand_fraction_matrix'],
                      wd=res['wd_matrix'], shoreline=res['shoreline_x'])
            ck = load_ckpt(tag)
        out[name] = dict(d=ck['d'], z=ck['z'], sf=ck['sf'], wd=ck['wd'],
                         shoreline=ck['shoreline'])
    # Pairwise rms differences of the observable and of final topography
    d0 = out['truth']['d']
    out['obs_rms'] = {k: float(np.sqrt(np.mean((out[k]['d'] - d0) ** 2)))
                      for k in variants}
    z0 = out['truth']['z'][:, -1]
    out['z_rms'] = {k: float(np.sqrt(np.mean((out[k]['z'][:, -1] - z0) ** 2)))
                    for k in variants}
    progress(f'  non-uniqueness: observable rms diffs = '
             f'{ {k: round(v, 3) for k, v in out["obs_rms"].items()} } m')
    return out


def run_jlandscape(d_obs, sigma, lam, cfg, progress=log,
                   i_knot=4, j_knot=8, n_grid=11):
    """Evaluate J on a 2-D grid over two knot values around the optimum.

    All other knots are frozen at the inverted optimum.  n_grid x n_grid
    forward calls (~1 min for 11 x 11).  Checkpointed as 'jlandscape'.
    """
    tag = 'jlandscape'
    ck = load_ckpt(tag)
    if ck is not None:
        return dict(grid_i=ck['grid_i'], grid_j=ck['grid_j'], J=ck['Jgrid'],
                    misfit=ck['Mgrid'], reg=ck['Rgrid'],
                    i_knot=int(ck['i_knot']), j_knot=int(ck['j_knot']),
                    e_base=ck['e_base'])
    progress(f'J landscape grid ({n_grid}x{n_grid}) ...')
    e_base = load_ckpt('iter_hist')['e_opt']
    gi = np.linspace(e_base[i_knot] - 30, e_base[i_knot] + 30, n_grid)
    gj = np.linspace(e_base[j_knot] - 30, e_base[j_knot] + 30, n_grid)
    R_mat = roughness_matrix(cfg)
    Jg = np.zeros((n_grid, n_grid))
    Mg = np.zeros((n_grid, n_grid))
    Rg = np.zeros((n_grid, n_grid))
    for a in range(n_grid):
        for b in range(n_grid):
            e = e_base.copy()
            e[i_knot], e[j_knot] = gi[a], gj[b]
            F = forward(e, cfg)
            r = (F - d_obs) / sigma
            Mg[a, b] = 0.5 * r @ r
            Rg[a, b] = 0.5 * lam * (e @ R_mat @ e)
            Jg[a, b] = Mg[a, b] + Rg[a, b]
    save_ckpt(tag, grid_i=gi, grid_j=gj, Jgrid=Jg, Mgrid=Mg, Rgrid=Rg,
              i_knot=i_knot, j_knot=j_knot, e_base=e_base)
    return dict(grid_i=gi, grid_j=gj, J=Jg, misfit=Mg, reg=Rg,
                i_knot=i_knot, j_knot=j_knot, e_base=e_base)


def run_step_experiment(progress=log):
    """Tikhonov vs TV on a sea-level truth WITH AN ABRUPT STEP (fig8).

    Uses a dedicated Config with step_height=25 m and a SPARSE observation
    operator (only 2 of the 8 pseudo-wells).  With sparse data the
    regulariser carries most of the prior information, which is exactly the
    setting where the choice of regulariser matters: H1 Tikhonov smears the
    step, edge-preserving TV recovers it sharply.
    Checkpointed as step_truth / step_tik / step_tv.
    """
    cfg = Config(step_height=25.0, well_idx=(1, 5))
    kt = knot_times(cfg)
    e_true = sea_level_truth(kt, cfg)

    ck = load_ckpt('step_truth')
    if ck is None:
        progress('STEP-truth forward run ...')
        d_clean = forward(e_true, cfg)
        save_ckpt('step_truth', d_clean=d_clean, e_true=e_true)
        ck = load_ckpt('step_truth')
    d_clean, e_true = ck['d_clean'], ck['e_true']
    d_obs, sigma = add_noise(d_clean, 0.03, seed=SEED)

    ck = load_ckpt('step_tik')
    if ck is None:
        progress('STEP truth: Tikhonov inversion ...')
        inv = Inversion(d_obs, sigma, 1.0, cfg=cfg, tag='step_tik', maxiter=12)
        out = inv.solve()
        save_ckpt('step_tik', e_opt=out.x, hist=np.array(inv.history))
        ck = load_ckpt('step_tik')
    e_tik = ck['e_opt']

    ck = load_ckpt('step_tv')
    if ck is None:
        progress('STEP truth: TV-IRLS inversion ...')
        e_tv, hist = tv_irls_inversion(d_obs, sigma, 1.0, cfg=cfg,
                                       tag='step_tv')
        save_ckpt('step_tv', e_opt=e_tv, hist=hist)
        ck = load_ckpt('step_tv')
    e_tv = ck['e_opt']
    rmse_tik, r_tik = metrics(e_tik, e_true)
    rmse_tv, r_tv = metrics(e_tv, e_true)
    progress(f'  STEP: Tikhonov rmse={rmse_tik:.2f} m; TV rmse={rmse_tv:.2f} m')
    return dict(e_true=e_true, e_tik=e_tik, e_tv=e_tv,
                rmse_tik=rmse_tik, r_tik=r_tik, rmse_tv=rmse_tv, r_tv=r_tv)


# ═════════════════════════════════════════════════════════════════════════════
# (i) PLOTTING MODULE — all education figures (300 dpi, Agg)
# ═════════════════════════════════════════════════════════════════════════════
def savefig(fig, name):
    """Save *fig* into FIGDIR at 300 dpi and close it."""
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log(f'  saved figures/{name}')


def dense_truth(cfg=DEFAULT_CFG, n=300):
    """(t_dense, e_dense) for smooth plotting of the truth curve."""
    td = np.linspace(TIME[0], TIME[-1], n)
    return td, sea_level_truth(td, cfg)


def draw_strat_section(ax, z_mat, color_mat, x_value, cmap, vmin, vmax,
                       clabel, sl_last=None, n_skip=1, wells=None):
    """Draw a Sedapp-style stratigraphic cross-section on *ax*.

    Layers are drawn from the z_matrix; colour encodes ``color_mat``
    (water depth, sand fraction or porosity).  Returns the ScalarMappable.
    """
    k = z_mat.shape[1]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for ii in range(0, k, n_skip):
        ax.scatter(x_value / 1000, z_mat[:, ii], s=1.0, linewidths=0,
                   c=[cmap(norm(v)) for v in color_mat[:, ii]])
    ax.plot(x_value / 1000, z_mat[:, -1], 'k-', lw=0.8)
    if sl_last is not None:
        ax.axhline(sl_last, color='blue', lw=0.8, ls='--', alpha=0.7,
                   label=f'sea level = {sl_last:.0f} m')
    if wells is not None:
        for w in wells / 1000:
            ax.axvline(w, color='grey', lw=0.5, alpha=0.5)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Elevation (m)')
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=clabel)
    return sm


# ── fig1: software architecture + GUI schematic ─────────────────────────────
def fig1_architecture(R):
    """2-panel: (a) workflow diagram; (b) annotated GUI mock drawn from the
    actual widget layout of SedappInversionGUI (see section (j))."""
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6.2))

    # ── (a) workflow ─────────────────────────────────────────────────────────
    ax = axs[0]
    ax.axis('off')

    def box(x, y, w, h, text, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec='k', lw=1.4,
                                   zorder=2, joinstyle='round'))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=8, zorder=3)

    def arrow(x1, y1, x2, y2, text='', rad=0.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', lw=1.6, color='k',
                                    connectionstyle=f'arc3,rad={rad}'))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.035, text, ha='center',
                    fontsize=8, style='italic')

    box(0.03, 0.74, 0.26, 0.20, 'Sea-level knots $e$\n(initial guess: 0 m)\n'
        '[GUI sliders]', '#dfeffb')
    box(0.38, 0.74, 0.26, 0.20, 'Sedapp forward kernel\n$F(e)$: FV diffusion\n'
        'stratigraphy (~0.5 s)', '#fde9c8')
    box(0.74, 0.74, 0.23, 0.20, 'Stratigraphy\nwater depth at\n8 pseudo-wells',
        '#e3f4d7')
    box(0.74, 0.38, 0.23, 0.20, 'Misfit\n$\\frac{1}{2}\\|(F(e)-d_{obs})'
        '/\\sigma\\|^2$', '#f6d5d5')
    box(0.03, 0.38, 0.26, 0.20, 'Noisy observations\n$d_{obs} = F(e_{true})'
        '+\\epsilon$\n[seeded rng]', '#fff3b8')
    box(0.38, 0.38, 0.26, 0.20, 'Regularisation\n$\\frac{\\lambda}{2}'
        '\\int \\dot e^2 dt$ (or TV)', '#e8dff5')
    box(0.38, 0.03, 0.26, 0.20, 'Objective $J$ + FD Jacobian\n'
        'damped Gauss-Newton', '#f0f0f0')
    box(0.74, 0.03, 0.23, 0.20, 'Laplace UQ\n95% credible band\n+ checkpoints',
        '#d5ecef')
    arrow(0.29, 0.84, 0.38, 0.84)
    arrow(0.64, 0.84, 0.74, 0.84)
    arrow(0.855, 0.74, 0.855, 0.58, 'residual')
    arrow(0.16, 0.58, 0.40, 0.50, '', rad=-0.2)
    arrow(0.51, 0.58, 0.51, 0.23, '', rad=0.0)
    arrow(0.855, 0.38, 0.62, 0.20, '', rad=0.2)
    arrow(0.38, 0.13, 0.16, 0.74, '', rad=0.35)
    ax.text(0.055, 0.27, 'update $e$ (iterate)', fontsize=8, style='italic')
    arrow(0.64, 0.13, 0.74, 0.13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('(a) Software architecture: variational inversion loop\n'
                 'around the Sedapp forward kernel', fontsize=11)

    # ── (b) annotated GUI mock (matches SedappInversionGUI layout) ──────────
    ax = axs[1]
    ax.axis('off')
    # window frame
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.90, fc='white', ec='k',
                               lw=1.6))
    ax.text(0.5, 0.875, 'ZongHe — Computational Stratigraphy Teaching Tool',
            ha='center', fontsize=9.5, weight='bold')
    # control panel (left column of widgets)
    ax.add_patch(plt.Rectangle((0.05, 0.10), 0.28, 0.72, fc='#f4f4f4',
                               ec='k', lw=1.2))
    ax.text(0.19, 0.775, 'Control panel', ha='center', fontsize=9,
            weight='bold')
    labels = ['Sea-level amplitude 1', 'Period 1', 'Amplitude 2', 'Period 2',
              'Trend', 'Step height', 'Noise level', 'lambda', 'No. of knots']
    for i, lab in enumerate(labels):
        y = 0.725 - i * 0.062
        ax.text(0.065, y, lab, fontsize=6.4, va='center')
        ax.add_patch(plt.Rectangle((0.17, y - 0.012), 0.14, 0.024,
                                   fc='#cfe0ef', ec='k', lw=0.6))
        ax.plot([0.20 + 0.02 * (i % 3)], [y], marker='s', ms=4, color='k')
    buttons = ['Run Forward', 'Generate Observations', 'Run Inversion',
               'L-curve', 'Reset']
    for i, b in enumerate(buttons):
        y = 0.155 - i * 0.011 - 0.0
    by = 0.165
    for i, b in enumerate(buttons):
        y = by - i * 0.03
        ax.add_patch(plt.Rectangle((0.065, y), 0.25, 0.024, fc='#e2e2e2',
                                   ec='k', lw=0.7))
        ax.text(0.19, y + 0.012, b, ha='center', va='center', fontsize=6.2)
    # matplotlib canvas (right)
    ax.add_patch(plt.Rectangle((0.37, 0.42), 0.58, 0.40, fc='#fcfcff',
                               ec='k', lw=1.2))
    ax.text(0.66, 0.775, 'Live matplotlib canvas (top)', ha='center',
            fontsize=8, weight='bold')
    tt = np.linspace(0, 1, 100)
    ax.plot(0.40 + 0.52 * tt, 0.60 + 0.10 * np.sin(6 * tt), 'b-', lw=1.4)
    ax.plot(0.40 + 0.52 * tt, 0.60 + 0.10 * np.sin(6 * tt) + 0.03 *
            np.sin(23 * tt), 'r.', ms=1.5)
    ax.text(0.66, 0.45, 'sea-level curve: truth vs inversion', ha='center',
            fontsize=6.5, style='italic')
    ax.add_patch(plt.Rectangle((0.37, 0.10), 0.58, 0.28, fc='#fffdf5',
                               ec='k', lw=1.2))
    ax.text(0.66, 0.345, 'Live matplotlib canvas (bottom)', ha='center',
            fontsize=8, weight='bold')
    # fake strat section
    rngf = np.random.default_rng(1)
    for irow in range(14):
        y0 = 0.13 + irow * 0.013
        c = plt.cm.viridis(0.15 + 0.7 * irow / 13)
        ax.plot([0.40, 0.92], [y0, y0 + 0.05], lw=5, color=c,
                solid_capstyle='butt', alpha=0.85)
    ax.text(0.66, 0.115, 'basin fill cross-section / data fit', ha='center',
            fontsize=6.5, style='italic')
    # annotations
    ann = [('sliders / entries set the\nexperiment configuration', (0.19, 0.55),
            (0.03, -0.06)),
           ('buttons trigger runs in a\nworker thread (no freezing)',
            (0.19, 0.10), (0.35, -0.16)),
           ('FigureCanvasTkAgg updates\nlive as iterations converge',
            (0.66, 0.26), (0.68, -0.06))]
    for text, xy, xytext in ann:
        ax.annotate(text, xy=xy, xytext=xytext, fontsize=6.8, ha='left',
                    arrowprops=dict(arrowstyle='->', lw=0.9, color='tab:red'),
                    color='tab:red')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.24, 1)
    ax.set_title('(b) GUI layout (schematic of the actual widget tree)',
                 fontsize=11)
    fig.tight_layout()
    savefig(fig, 'fig1_architecture_gui.png')


# ── fig2: forward-model teaching demo ───────────────────────────────────────
def fig2_forward_demo(R):
    """5-panel forward-model tour: sea-level input, subsidence field,
    water-depth field, sand-fraction field, final stratigraphic section."""
    ck = R['truth_ck']
    xv, tvec = ck['x_value'], ck['time']
    fig = plt.figure(figsize=(12.5, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.25], hspace=0.38,
                          wspace=0.24)

    ax = fig.add_subplot(gs[0, 0])
    td, ed = dense_truth()
    ax.plot(td, ed, 'b-', lw=2)
    ax.plot(knot_times(), R['e_true_knots'], 'ko', ms=4, alpha=0.6,
            label='control knots (n=12)')
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Relative sea level (m)')
    ax.set_title('(a) Input: relative sea-level curve $e(t)$')
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    ds = subsidence_rate_field()
    ax.plot(xv / 1000, ds, 'k-', lw=2)
    ax.fill_between(xv / 1000, 0, ds, color='tab:brown', alpha=0.3)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Subsidence rate (m/Ma)')
    ax.set_title('(b) Input: tectonic subsidence field $DS(x)$')

    ax = fig.add_subplot(gs[1, 0])
    im = ax.pcolormesh(xv / 1000, tvec, ck['wd'].T, cmap='viridis',
                       shading='auto')
    ax.plot(ck['shoreline'] / 1000, tvec, 'r-', lw=0.8, label='shoreline')
    for w in WELL_X / 1000:
        ax.axvline(w, color='w', lw=0.6, alpha=0.6)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Time (Ma)')
    ax.set_title('(c) Output: water-depth field (white = pseudo-wells)')
    plt.colorbar(im, ax=ax, label='Water depth (m)')
    ax.legend(loc='lower right', fontsize=7.5)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.pcolormesh(xv / 1000, tvec, ck['sf'].T, cmap='YlOrBr',
                       shading='auto', vmin=0, vmax=1)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Time (Ma)')
    ax.set_title('(d) Output: deposited sand-fraction field')
    plt.colorbar(im, ax=ax, label='Sand fraction (-)')

    ax = fig.add_subplot(gs[2, :])
    draw_strat_section(ax, ck['z'], ck['sf'], xv, plt.cm.YlOrBr, 0, 1,
                       'Sand fraction (-)', sl_last=ed[-1], wells=WELL_X)
    ax.set_title('(e) Final stratigraphic architecture (layers coloured by '
                 'sand fraction)')
    ax.legend(loc='upper left', fontsize=8)
    savefig(fig, 'fig2_forward_demo.png')


# ── fig3: non-uniqueness demo ───────────────────────────────────────────────
def fig3_nonuniqueness(R):
    """Same stratigraphy, different sea levels: 3 e(t) curves whose Sedapp
    outputs are nearly indistinguishable — why inversion needs regularisation."""
    nu = R['nonuniq']
    kt = knot_times()
    td, _ = dense_truth()
    xv = R['truth_ck']['x_value']
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 7.6))

    ax = axs[0, 0]
    styles = {'truth': ('k-', 'Truth $e_{true}(t)$'),
              'shifted': ('b--', 'Variant A: + 6 m, 55 Ma ripple'),
              'rescaled': ('r-.', 'Variant B: 1.08$\\times$ - 4 m')}
    for name, (st, lab) in styles.items():
        e = nu['knots'][name]
        ax.plot(td, np.interp(td, kt, e), st, lw=1.8, label=lab)
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Relative sea level (m)')
    ax.set_title('(a) Three quite different sea-level histories')
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    for name, (st, lab) in styles.items():
        ax.plot(nu[name]['z'][:, -1] - nu['truth']['z'][:, -1], xv / 1000,
                st.replace('-', '-') if name == 'truth' else st, lw=1.5,
                label=f'{lab.split(":")[0]} (rms {nu["z_rms"][name]:.2f} m)'
                if name != 'truth' else 'reference')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Final-topography difference (m)')
    ax.set_ylabel('Distance (km)')
    ax.set_title('(b) ...produce nearly identical stratigraphy')
    ax.legend(fontsize=7.5)

    ax = axs[1, 0]
    tt = np.tile(TIME, N_WELL)
    ww = np.repeat(WELL_X / 1000, K)
    for name, c in [('shifted', 'b'), ('rescaled', 'r')]:
        dd = (nu[name]['d'] - nu['truth']['d'])
        ax.scatter(tt, dd, s=3, alpha=0.25, c=c,
                   label=f'{name}: rms {nu["obs_rms"][name]:.2f} m')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Water-depth difference at wells (m)')
    ax.set_title('(c) Observable differences are at the few-metre level')
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.axis('off')
    txt = ('Teaching point\n\n'
           'Many sea-level histories map to nearly the\n'
           'same stratigraphy: the forward operator $F$\n'
           'smooths and integrates. The inverse problem is\n'
           'therefore ILL-POSED — small data differences\n'
           'admit very different sea-level curves.\n\n'
           'This is why students must add prior information\n'
           '(regularisation) and why uncertainty\n'
           'quantification matters (fig4, fig7, fig9).')
    ax.text(0.05, 0.95, txt, va='top', fontsize=11, family='serif')
    fig.suptitle('Non-uniqueness: "same stratigraphy, different sea levels"',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, 'fig3_nonuniqueness.png')


# ── fig4: variational principle visualised ──────────────────────────────────
def fig4_variational(R):
    """J landscape over two knot values with the Gauss-Newton path overlaid,
    plus the misfit-vs-regularisation trade-off measured from the lambda
    sweep."""
    jl = R['jlandscape']
    GI, GJ = np.meshgrid(jl['grid_i'], jl['grid_j'], indexing='ij')
    it = R['iterates']
    lam = R['best_lam']

    fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.0))
    ax = axs[0]
    lv = np.linspace(jl['J'].min(), np.percentile(jl['J'], 92), 24)
    cs = ax.contourf(GI, GJ, jl['J'], levels=lv, cmap='viridis')
    ax.contour(GI, GJ, jl['J'], levels=lv[::4], colors='w', linewidths=0.4,
               alpha=0.6)
    plt.colorbar(cs, ax=ax, label='Objective $J$')
    ik, jk = jl['i_knot'], jl['j_knot']
    ax.plot(it[:, ik], it[:, jk], 'o-', color='tab:red', ms=4, lw=1.5,
            label='Gauss-Newton path', zorder=5)
    ax.plot(it[0, ik], it[0, jk], 'ws', ms=9, label='start', zorder=6)
    ax.plot(it[-1, ik], it[-1, jk], 'w*', ms=16, label='optimum', zorder=6)
    et = R['e_true_knots']
    ax.plot(et[ik], et[jk], marker='X', color='orange', ms=11,
            label='truth', zorder=6)
    ax.set_xlabel(f'knot {ik} value (m)')
    ax.set_ylabel(f'knot {jk} value (m)')
    ax.set_title(f'(a) Objective landscape $J(e)$ over two knots\n'
                 f'($\\lambda$={lam:g}; others frozen at optimum)')
    ax.legend(fontsize=8, loc='upper right')

    ax = axs[1]
    lr = R['lam_results']
    misf = np.array([lr[l]['misfit'] for l in LAMBDAS])
    roughv = np.array([roughness(lr[l]['e']) for l in LAMBDAS])
    ax.loglog(LAMBDAS, misf, 'o-', color='tab:blue', label='data misfit')
    ax.loglog(LAMBDAS, np.array(LAMBDAS) * roughv, 's-', color='tab:orange',
              label='$\\lambda \\times$ roughness')
    ax.axvline(lam, color='tab:red', ls='--', lw=1.2,
               label=f'L-curve knee $\\lambda$={lam:g}')
    ax.set_xlabel('Regularisation weight $\\lambda$')
    ax.set_ylabel('Term value at optimum')
    ax.set_title('(b) Misfit–regularisation trade-off (real runs)')
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    savefig(fig, 'fig4_variational_principle.png')


# ── fig5: inversion step-by-step iteration frames ───────────────────────────
def fig5_iterations(R):
    """Iterations 0, 2, 5, 10, final: sea-level curve evolution with a
    J-decrease inset — what a student watches converge in the GUI."""
    it = R['iterates']
    hist = R['iter_hist']
    calls = hist[:, 0] - hist[0, 0] + 1   # forward calls spent in this inversion
    kt = knot_times()
    td, ed = dense_truth()
    picks = [0, 2, 5, 10, len(it) - 1]
    picks = sorted(set(min(p, len(it) - 1) for p in picks))
    fig, axs = plt.subplots(1, len(picks), figsize=(14.5, 3.6), sharey=True)
    for ax, ip in zip(axs, picks):
        ax.plot(td, ed, 'k-', lw=2, label='truth' if ip == picks[0] else None)
        ax.plot(kt, it[ip], 'o-', color='tab:blue', ms=4, lw=1.5,
                label=f'iteration {ip}')
        ax.set_title(f'iteration {ip}', fontsize=10)
        ax.set_xlabel('Time (Ma)')
        if ax is axs[0]:
            ax.set_ylabel('Relative sea level (m)')
        # J-decrease inset
        axin = ax.inset_axes([0.52, 0.55, 0.44, 0.40])
        axin.semilogy(calls, hist[:, 1], 'o-', ms=2.5, lw=1,
                      color='tab:blue')
        axin.axvline(calls[min(ip, len(hist) - 1)], color='tab:red', lw=1)
        axin.set_title('$J$ vs fwd calls', fontsize=6.5)
        axin.tick_params(labelsize=6)
        axin.grid(alpha=0.3)
    axs[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Inversion step by step: the sea-level curve converges while '
                 '$J$ decreases', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, 'fig5_iteration_frames.png')


# ── fig6: effect of noise levels ────────────────────────────────────────────
def fig6_noise(R):
    """1/3/5% noise recovery with error bars from 3 seeded realisations."""
    kt = knot_times()
    td, ed = dense_truth()
    nr = R['noise_results']
    ens = R['noise_ensemble']
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axs[0]
    ax.plot(td, ed, 'k-', lw=2.2, label='Truth')
    for f, c in zip(NOISE_LEVELS, ['tab:green', 'tab:blue', 'tab:red']):
        ax.plot(kt, nr[f]['e'], 'o-', color=c, ms=4, lw=1.5,
                label=f'{int(f*100)}% noise (RMSE {nr[f]["rmse"]:.2f} m)')
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Relative sea level (m)')
    ax.set_title('(a) Recovery at 1 / 3 / 5 % noise')
    ax.legend(fontsize=8)

    ax = axs[1]
    for f, c, xoff in zip(NOISE_LEVELS, ['tab:green', 'tab:blue', 'tab:red'],
                          [-0.12, 0.0, 0.12]):
        rows = ens[ens[:, 0] == f]
        mu, sd = rows[:, 2].mean(), rows[:, 2].std()
        ax.errorbar([f * 100 + xoff], [mu], yerr=[sd], fmt='o', color=c,
                    ms=7, capsize=5, lw=1.5,
                    label=f'{int(f*100)}% (n=3 realisations)')
        ax.plot([f * 100 + xoff] * len(rows), rows[:, 2], '.', color=c,
                alpha=0.5, ms=5)
    ax.set_xlabel('Noise level (% of data std)')
    ax.set_ylabel('Sea-level RMSE (m)')
    ax.set_title('(b) RMSE vs noise — mean ± 1 std over realisations')
    ax.legend(fontsize=8)

    ax = axs[2]
    for f, c in zip(NOISE_LEVELS, ['tab:green', 'tab:blue', 'tab:red']):
        rows = ens[ens[:, 0] == f]
        mu, sd = rows[:, 3].mean(), rows[:, 3].std()
        ax.errorbar([f * 100], [mu], yerr=[sd], fmt='s-', color=c, ms=7,
                    capsize=5, lw=1.5, label=f'{int(f*100)}%')
    ax.set_xlabel('Noise level (% of data std)')
    ax.set_ylabel('Pearson correlation r')
    ax.set_ylim(0.9, 1.005)
    ax.set_title('(c) Correlation vs noise')
    ax.legend(fontsize=8)
    fig.suptitle('Effect of observation noise on sea-level recovery '
                 '(seeded realisations)', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, 'fig6_noise_levels.png')


# ── fig7: effect of lambda + L-curve ────────────────────────────────────────
def fig7_lambda(R):
    """Under- and over-regularised examples plus the L-curve used to pick
    lambda objectively."""
    lr = R['lam_results']
    lam = R['best_lam']
    kt = knot_times()
    td, ed = dense_truth()
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axs[0]
    xs = np.array([roughness(lr[l]['e']) for l in LAMBDAS])
    ys = np.array([lr[l]['misfit'] for l in LAMBDAS])
    ax.loglog(xs, ys, 'o-', color='tab:blue')
    for l, x, y in zip(LAMBDAS, xs, ys):
        off = {0.1: (8, -2), 0.01: (8, 9), 0.001: (12, -9)}.get(l, (6, 5))
        ax.annotate(f'{l:g}', (x, y), textcoords='offset points', xytext=off,
                    fontsize=8)
    iB = LAMBDAS.index(lam)
    ax.plot(xs[iB], ys[iB], 'o', ms=13, mfc='none', mec='tab:red', mew=2)
    ax.set_xlabel(r'Roughness $\int \dot e^2 dt$  (m$^2$/Ma)')
    ax.set_ylabel(r'Misfit $\frac{1}{2}\|(F-d)/\sigma\|^2$')
    ax.set_title(f'(a) L-curve — knee at $\\lambda$={lam:g}')

    ax = axs[1]
    ax.plot(td, ed, 'k-', lw=2.2, label='Truth')
    iK = LAMBDAS.index(lam)
    picks = []
    if iK > 0:
        picks.append((LAMBDAS[max(0, iK - 1)], 'under-regularised'))
    picks.append((lam, 'L-curve pick'))
    picks.append((LAMBDAS[-1], 'over-regularised'))
    for (l, lab), c in zip(picks, ['tab:red', 'tab:blue', 'tab:green'][-len(picks):]):
        ax.plot(kt, lr[l]['e'], 'o-', color=c, ms=4, lw=1.6,
                label=f'{lab} $\\lambda$={l:g} (RMSE {lr[l]["rmse"]:.1f} m)')
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Relative sea level (m)')
    ax.set_title('(b) Under / optimal / over-regularised recoveries')
    ax.legend(fontsize=8)

    ax = axs[2]
    rmses = [lr[l]['rmse'] for l in LAMBDAS]
    ax.semilogx(LAMBDAS, rmses, 'o-', color='tab:purple', lw=1.8)
    ax.axvline(lam, color='tab:red', ls='--', lw=1.2)
    ax.axvline(R['lam_minrmse'], color='tab:green', ls=':', lw=1.2,
               label=f'min-RMSE $\\lambda$={R["lam_minrmse"]:g} (unknowable '
               'in practice)')
    ax.set_xlabel('$\\lambda$')
    ax.set_ylabel('Sea-level RMSE (m)')
    ax.set_title('(c) Recovery error vs $\\lambda$')
    ax.legend(fontsize=8)
    fig.suptitle('Choosing the regularisation weight: the L-curve criterion',
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, 'fig7_lambda_lcurve.png')


# ── fig8: Tikhonov vs TV on a step truth ────────────────────────────────────
def fig8_tikhonov_vs_tv(R):
    """Regularisation choice matters: on an abrupt step, TV (edge-preserving)
    beats H1 Tikhonov (smoothing) — the classic teaching example."""
    st = R['step']
    kt = knot_times()
    td = np.linspace(TIME[0], TIME[-1], 300)
    cfg_step = Config(step_height=25.0, well_idx=(1, 5))
    ed = sea_level_truth(td, cfg_step)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axs[0]
    ax.plot(td, ed, 'k-', lw=2.2, label='Truth (with 25 m step)')
    ax.plot(kt, st['e_tik'], 'o-', color='tab:blue', ms=4, lw=1.6,
            label=f'Tikhonov $H^1$ (RMSE {st["rmse_tik"]:.2f} m, '
            f'r={st["r_tik"]:.3f})')
    ax.plot(kt, st['e_tv'], 's-', color='tab:orange', ms=4, lw=1.6,
            label=f'TV / IRLS (RMSE {st["rmse_tv"]:.2f} m, '
            f'r={st["r_tv"]:.3f})')
    ax.axvline(T1 / 2, color='grey', ls=':', lw=1)
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Relative sea level (m)')
    ax.set_title('(a) Recovering an abrupt step from only 2 pseudo-wells\n'
                 '(sparse data → the regulariser IS the prior)')
    ax.legend(fontsize=8)

    ax = axs[1]
    x = np.arange(2)
    rms = [st['rmse_tik'], st['rmse_tv']]
    crs = [st['r_tik'], st['r_tv']]
    ax.bar(x - 0.18, rms, width=0.36, color='tab:blue', alpha=0.8,
           label='RMSE (m)')
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, crs, width=0.36, color='tab:red', alpha=0.8,
            label='Pearson r')
    ax2.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(['Tikhonov $H^1$', 'Total variation'])
    ax.set_ylabel('RMSE (m)', color='tab:blue')
    ax2.set_ylabel('Pearson r', color='tab:red')
    ax.grid(False)
    ax2.grid(False)
    for xi, v in zip(x - 0.18, rms):
        ax.text(xi, v + 0.05, f'{v:.2f}', ha='center', fontsize=9,
                color='tab:blue')
    for xi, v in zip(x + 0.18, crs):
        ax2.text(xi, v + 0.02, f'{v:.3f}', ha='center', fontsize=9,
                 color='tab:red')
    ax.set_title('(b) TV preserves the step; $H^1$ smears it')
    fig.suptitle('Teaching regularisation choice: with sparse data on a '
                 'stepped sea-level curve,\nedge-preserving TV beats '
                 'smoothing Tikhonov $H^1$', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, 'fig8_tikhonov_vs_tv_step.png')


# ── fig9: uncertainty quantification teaching panel ─────────────────────────
def fig9_uq(R):
    """Laplace band, posterior sample ensemble, and a coverage diagnostic."""
    kt = knot_times()
    td, ed = dense_truth()
    best = R['best']
    samples = R['samples']
    lo, hi = R['lo'], R['hi']
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axs[0]
    ax.plot(td, ed, 'k-', lw=2, label='Truth $e_{true}(t)$')
    ax.fill_between(kt, lo, hi, color='tab:blue', alpha=0.25,
                    label='Laplace 95% credible band')
    ax.plot(kt, best['e'], 'o-', color='tab:blue', ms=4, lw=1.5,
            label='Inversion (MAP)')
    ax.plot(kt, R['e_true_knots'], 'ks', ms=4, alpha=0.5, label='Truth knots')
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Relative sea level (m)')
    ax.set_title('(a) Linearised (Laplace) posterior band')
    ax.legend(fontsize=8)

    ax = axs[1]
    rng = np.random.default_rng(SEED + 7)
    idx = rng.choice(len(samples), size=60, replace=False)
    for i in idx:
        ax.plot(kt, samples[i], '-', color='tab:blue', alpha=0.10, lw=0.8)
    ax.plot(td, ed, 'k-', lw=2, label='Truth')
    ax.plot(kt, best['e'], '-', color='tab:red', lw=1.8, label='MAP')
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Relative sea level (m)')
    ax.set_title('(b) 60 posterior samples (ensemble view)')
    ax.legend(fontsize=8)

    ax = axs[2]
    inside = (R['e_true_knots'] >= lo) & (R['e_true_knots'] <= hi)
    ax.errorbar(kt, best['e'] - R['e_true_knots'],
                yerr=1.96 * R['post_std'], fmt='none', capsize=3,
                ecolor='grey', lw=1)
    ax.scatter(kt[inside], (best['e'] - R['e_true_knots'])[inside],
               color='tab:green', zorder=5, label='covered')
    ax.scatter(kt[~inside], (best['e'] - R['e_true_knots'])[~inside],
               color='tab:red', zorder=5, label='missed')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('MAP error (m) with 95% bars')
    cov = R['coverage'] * 100
    ax.set_title(f'(c) Coverage diagnostic: {cov:.0f}% of true knots inside '
                 'the 95% band\n(green = covered, red = missed — linearised '
                 'UQ is optimistic here)')
    fig.suptitle('Uncertainty quantification with the Laplace approximation',
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, 'fig9_uncertainty.png')


# ── fig10: GUI-guided student exercise workflow ─────────────────────────────
def fig10_exercise(R):
    """A 4-stage student exercise mapped onto the GUI buttons, with the
    expected output at each stage (all real results from this script)."""
    kt = knot_times()
    td, ed = dense_truth()
    best = R['best']
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 7.8))

    ax = axs[0, 0]
    ax.plot(td, ed, 'b-', lw=2)
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Relative sea level (m)')
    ax.set_title('Stage 1 — "Run Forward": set the sea-level curve with the '
                 'sliders\nand watch Sedapp build the basin fill')
    ax.text(0.03, 0.05, 'Learning goal: forward model = map from forcing to '
            'stratigraphy', transform=ax.transAxes, fontsize=8.5,
            style='italic')

    ax = axs[0, 1]
    iw_show = 3
    tvec = R['truth_ck']['time']
    d_clean = R['d_clean'].reshape(N_WELL, K)
    d_obs = R['d_obs'][0.03].reshape(N_WELL, K)
    ax.plot(tvec, d_clean[iw_show], 'k-', lw=2, label='clean')
    ax.plot(tvec, d_obs[iw_show], '.', color='tab:red', ms=4,
            label='3% noise (seeded)')
    for w in WELL_X / 1000:
        pass
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Water depth (m)')
    ax.set_title(f'Stage 2 — "Generate Observations": pseudo-well data\n'
                 f'(well at x = {WELL_X[iw_show]/1000:.1f} km) with seeded '
                 'noise')
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.plot(td, ed, 'k-', lw=2, label='Truth')
    ax.plot(kt, best['e'], 'o-', color='tab:blue', ms=4, lw=1.5,
            label=f'Inversion (RMSE {best["rmse"]:.2f} m)')
    ax.fill_between(kt, R['lo'], R['hi'], color='tab:blue', alpha=0.2,
                    label='95% band')
    ax.set_xlabel('Time (Ma)')
    ax.set_ylabel('Relative sea level (m)')
    ax.set_title('Stage 3 — "Run Inversion": recover e(t) and its uncertainty')
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    lr = R['lam_results']
    xs = np.array([roughness(lr[l]['e']) for l in LAMBDAS])
    ys = np.array([lr[l]['misfit'] for l in LAMBDAS])
    ax.loglog(xs, ys, 'o-', color='tab:blue')
    iB = LAMBDAS.index(R['best_lam'])
    ax.plot(xs[iB], ys[iB], 'o', ms=13, mfc='none', mec='tab:red', mew=2)
    ax.set_xlabel(r'Roughness $\int \dot e^2 dt$')
    ax.set_ylabel('Misfit')
    ax.set_title('Stage 4 — "L-curve": pick $\\lambda$ objectively, then '
                 '"Reset"\nand repeat with your own scenario')
    fig.suptitle('A GUI-guided student exercise (≈ 45 min lab session)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, 'fig10_exercise_workflow.png')


def make_all_figures(R):
    """Generate every education figure from the experiment results dict."""
    log('Drawing figures ...')
    fig1_architecture(R)
    fig2_forward_demo(R)
    fig3_nonuniqueness(R)
    fig4_variational(R)
    fig5_iterations(R)
    fig6_noise(R)
    fig7_lambda(R)
    fig8_tikhonov_vs_tv(R)
    fig9_uq(R)
    fig10_exercise(R)


# ═════════════════════════════════════════════════════════════════════════════
# (j) GUI — tkinter control panel + live matplotlib canvas (threaded)
# ═════════════════════════════════════════════════════════════════════════════
class AppCore:
    """GUI-independent application state and run logic.

    The GUI (SedappInversionGUI) is a thin shell around this class; the headless
    smoke test instantiates AppCore directly to verify the logic without a
    display.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.e_true_knots = sea_level_truth(knot_times(self.cfg), self.cfg)
        self.d_clean = None          # clean observable from the forward run
        self.d_obs = None            # noisy observations
        self.sigma = None
        self.res_truth = None        # full Sedapp result of last forward run
        self.inv_result = None       # last inversion Out object
        self.inv_history = None
        self.inv_iterates = None
        self.lcurve = None           # (lambdas, misfits, roughs, knee)
        self.busy = False            # True while a worker thread runs

    # ── workflow steps (each is safe to call from a worker thread) ──────────
    def run_forward(self):
        """(Re)generate the truth curve and run the Sedapp forward model."""
        self.e_true_knots = sea_level_truth(knot_times(self.cfg), self.cfg)
        d, res = forward(self.e_true_knots, self.cfg, keep_full=True)
        self.d_clean, self.res_truth = d, res
        self.d_obs = None
        return d, res

    def generate_observations(self):
        """Add seeded Gaussian noise to the clean forward data."""
        if self.d_clean is None:
            self.run_forward()
        self.d_obs, self.sigma = add_noise(self.d_clean, self.cfg.noise,
                                           seed=SEED)
        return self.d_obs

    def run_inversion(self, callback=None):
        """Run the damped Gauss-Newton inversion with the current lambda."""
        if self.d_obs is None:
            self.generate_observations()
        inv = Inversion(self.d_obs, self.sigma, self.cfg.lam, cfg=self.cfg,
                        tag='gui', maxiter=10, callback=callback)
        out = inv.solve()
        self.inv_result = out
        self.inv_history = np.array(inv.history)
        self.inv_iterates = inv.iterates
        return out

    def run_lcurve(self, lambdas=(0.01, 0.1, 1.0, 10.0, 100.0)):
        """Quick L-curve sweep (reduced lambda list + fewer iters for the GUI)."""
        if self.d_obs is None:
            self.generate_observations()
        misf, roughv = [], []
        for lam in lambdas:
            inv = Inversion(self.d_obs, self.sigma, lam, cfg=self.cfg,
                            tag=f'gui_lam{lam:g}', maxiter=8)
            out = inv.solve()
            misf.append(out.misfit)
            roughv.append(roughness(out.x, self.cfg))
        knee, _ = lcurve_knee(list(lambdas), misf, roughv)
        self.lcurve = (list(lambdas), misf, roughv, knee)
        return self.lcurve

    def reset(self):
        """Restore the default configuration and clear all results."""
        self.__init__(Config())


class SedappInversionGUI:
    """Tkinter GUI: control panel (left) + two live matplotlib canvases (right).

    Layout (mirrored in fig1 panel b):
      * sliders/entries for sea-level amplitudes, periods, trend, step,
        noise level, lambda, and number of knots;
      * buttons: Run Forward, Generate Observations, Run Inversion, L-curve,
        Reset;
      * top canvas: sea-level curve (truth / observations / inversion);
      * bottom canvas: basin-fill cross-section or data fit.
    Long runs execute in a worker thread; plot updates are marshalled back to
    the GUI thread with ``root.after``.
    """

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.tk = tk
        self.root = root
        self.core = AppCore()
        root.title('ZongHe — Computational Stratigraphy Teaching Tool')
        root.geometry('1180x720')

        main = ttk.Frame(root)
        main.pack(fill='both', expand=True)
        panel = ttk.Frame(main, width=280)
        panel.pack(side='left', fill='y', padx=6, pady=6)
        panel.pack_propagate(False)

        ttk.Label(panel, text='Control panel',
                  font=('TkDefaultFont', 11, 'bold')).pack(pady=(4, 8))

        self.vars = {}
        specs = [('amp1', 'Amplitude 1 (m)', 0, 50, 22.0),
                 ('period1', 'Period 1 (Ma)', 30, 200, 85.0),
                 ('amp2', 'Amplitude 2 (m)', 0, 30, 10.0),
                 ('period2', 'Period 2 (Ma)', 10, 100, 34.0),
                 ('trend', 'Trend (m)', -50, 50, 18.0),
                 ('step_height', 'Step height (m)', 0, 50, 0.0),
                 ('noise', 'Noise level (frac)', 0.0, 0.1, 0.03),
                 ('lam', 'lambda', 1e-3, 1000.0, 1.0),
                 ('n_knots', 'No. of knots', 6, 20, 12)]
        for key, lab, lo, hi, val in specs:
            ttk.Label(panel, text=lab).pack(anchor='w', padx=4)
            var = tk.DoubleVar(value=val)
            self.vars[key] = var
            ttk.Scale(panel, from_=lo, to=hi, variable=var,
                      orient='horizontal').pack(fill='x', padx=4)

        btn_frame = ttk.Frame(panel)
        btn_frame.pack(fill='x', pady=8)
        for text, cmd in [('Run Forward', self.on_run_forward),
                          ('Generate Observations', self.on_generate_obs),
                          ('Run Inversion', self.on_run_inversion),
                          ('L-curve', self.on_lcurve),
                          ('Reset', self.on_reset)]:
            ttk.Button(btn_frame, text=text, command=cmd).pack(fill='x',
                                                               pady=2, padx=4)
        self.status = tk.StringVar(value='Ready.')
        ttk.Label(panel, textvariable=self.status, wraplength=250,
                  foreground='navy').pack(pady=6, padx=4)

        # canvases (right)
        right = ttk.Frame(main)
        right.pack(side='right', fill='both', expand=True)
        self.fig_top, self.ax_top = plt.subplots(figsize=(7.4, 3.4))
        self.fig_bot, self.ax_bot = plt.subplots(figsize=(7.4, 3.4))
        self.canvas_top = FigureCanvasTkAgg(self.fig_top, master=right)
        self.canvas_top.get_tk_widget().pack(fill='both', expand=True)
        self.canvas_bot = FigureCanvasTkAgg(self.fig_bot, master=right)
        self.canvas_bot.get_tk_widget().pack(fill='both', expand=True)
        self.redraw_initial()

    # ── helpers ──────────────────────────────────────────────────────────────
    def read_cfg(self):
        """Build a Config from the current slider/entry values."""
        v = self.vars
        return Config(amp1=v['amp1'].get(), period1=v['period1'].get(),
                      amp2=v['amp2'].get(), period2=v['period2'].get(),
                      trend=v['trend'].get(),
                      step_height=v['step_height'].get(),
                      noise=v['noise'].get(), lam=v['lam'].get(),
                      n_knots=int(round(v['n_knots'].get())))

    def set_status(self, msg):
        self.root.after(0, lambda: self.status.set(msg))

    def start_worker(self, fn, done_msg):
        """Run *fn* in a daemon thread so the GUI never freezes."""
        if self.core.busy:
            self.set_status('Busy — please wait for the current run.')
            return
        self.core.busy = True
        self.set_status(done_msg + ' ...')

        def work():
            try:
                fn()
                self.set_status(done_msg + ' — done.')
            except Exception as exc:                # noqa: BLE001
                self.set_status(f'Error: {exc}')
            finally:
                self.core.busy = False
                self.root.after(0, self.refresh_plots)
        threading.Thread(target=work, daemon=True).start()

    # ── button handlers ──────────────────────────────────────────────────────
    def on_run_forward(self):
        self.core.cfg = self.read_cfg()
        self.start_worker(self.core.run_forward, 'Forward run')

    def on_generate_obs(self):
        self.core.cfg = self.read_cfg()
        self.start_worker(self.core.generate_observations,
                          'Generating observations')

    def on_run_inversion(self):
        self.core.cfg = self.read_cfg()
        self.start_worker(self.core.run_inversion, 'Inversion')

    def on_lcurve(self):
        self.core.cfg = self.read_cfg()
        self.start_worker(self.core.run_lcurve, 'L-curve sweep')

    def on_reset(self):
        self.core.reset()
        for k, v in [('amp1', 22.0), ('period1', 85.0), ('amp2', 10.0),
                     ('period2', 34.0), ('trend', 18.0), ('step_height', 0.0),
                     ('noise', 0.03), ('lam', 1.0), ('n_knots', 12)]:
            self.vars[k].set(v)
        self.set_status('Reset to defaults.')
        self.refresh_plots()

    # ── plotting ─────────────────────────────────────────────────────────────
    def redraw_initial(self):
        ax = self.ax_top
        ax.clear()
        td = np.linspace(TIME[0], TIME[-1], 300)
        cfg = self.core.cfg
        ax.plot(td, sea_level_truth(td, cfg), 'b-', lw=2,
                label='truth (current sliders)')
        ax.set_xlabel('Time (Ma)')
        ax.set_ylabel('Relative sea level (m)')
        ax.legend(fontsize=8)
        self.fig_top.tight_layout()
        self.canvas_top.draw_idle()
        self.ax_bot.clear()
        self.ax_bot.text(0.5, 0.5, 'Press "Run Forward" to build the basin',
                         ha='center', va='center', transform=self.ax_bot.transAxes)
        self.canvas_bot.draw_idle()

    def refresh_plots(self):
        c = self.core
        cfg = c.cfg
        kt = knot_times(cfg)
        # top canvas: sea level / observations / inversion
        ax = self.ax_top
        ax.clear()
        td = np.linspace(TIME[0], TIME[-1], 300)
        ax.plot(td, sea_level_truth(td, cfg), 'k-', lw=2, label='truth')
        if c.inv_result is not None:
            ax.plot(kt, c.inv_result.x, 'o-', color='tab:blue', ms=4,
                    lw=1.5, label=f'inversion (J={c.inv_result.fun:.1f})')
        ax.legend(fontsize=8)
        ax.set_xlabel('Time (Ma)')
        ax.set_ylabel('Relative sea level (m)')
        self.fig_top.tight_layout()
        self.canvas_top.draw_idle()
        # bottom canvas: basin fill or L-curve or data fit
        ax = self.ax_bot
        ax.clear()
        if c.lcurve is not None:
            lambdas, misf, roughv, knee = c.lcurve
            ax.loglog(roughv, misf, 'o-', color='tab:blue')
            ax.set_xlabel('roughness')
            ax.set_ylabel('misfit')
            ax.set_title(f'L-curve (knee at $\\lambda$={knee:g})')
        elif c.res_truth is not None:
            res = c.res_truth
            draw_strat_section(ax, res['z_matrix'],
                               res['sand_fraction_matrix'], res['x_value'],
                               plt.cm.YlOrBr, 0, 1, 'sand fraction',
                               sl_last=float(res['params']['sl_series'][-1]))
            ax.set_title('Basin fill (sand fraction)')
        else:
            ax.text(0.5, 0.5, 'No run yet', ha='center', va='center',
                    transform=ax.transAxes)
        self.fig_bot.tight_layout()
        self.canvas_bot.draw_idle()


def launch_gui():
    """Launch the tkinter GUI (requires a display)."""
    os.environ['ZONGHE_GUI'] = '1'
    import tkinter as tk
    root = tk.Tk()
    SedappInversionGUI(root)
    root.mainloop()


def smoke_test():
    """Headless GUI smoke test: tkinter importable, GUI class defined, and the
    full application logic (AppCore) instantiates and runs a micro-workflow
    without a display."""
    import importlib
    tk_spec = importlib.util.find_spec('tkinter')
    assert tk_spec is not None, 'tkinter not available'
    import tkinter  # noqa: F401  (import must succeed)
    assert hasattr(SedappInversionGUI, '__init__'), 'GUI class missing'
    core = AppCore()
    assert core.cfg.n_knots == 12
    e = sea_level_truth(knot_times(core.cfg), core.cfg)
    assert e.shape == (12,) and np.all(np.isfinite(e))
    d, res = core.run_forward()          # one real Sedapp call (~0.5 s)
    assert d.shape == (N_DATA,) and np.all(np.isfinite(d))
    core.generate_observations()
    assert core.d_obs.shape == (N_DATA,)
    core.reset()
    print('SMOKE TEST PASSED: tkinter importable, GUI class present, '
          'AppCore forward+noise workflow runs headlessly.')
    return True


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def main():
    """Command-line entry: --nogui runs the pipeline headless; --figures also
    regenerates all education figures; --smoketest checks the GUI headlessly."""
    ap = argparse.ArgumentParser(
        description='ZongHe: forward–inverse stratigraphic modelling '
                    'teaching tool (Sedapp + variational sea-level inversion).')
    ap.add_argument('--nogui', action='store_true',
                    help='run the full pipeline headless (Agg backend)')
    ap.add_argument('--figures', action='store_true',
                    help='(with --nogui) generate all education figures')
    ap.add_argument('--smoketest', action='store_true',
                    help='headless GUI/logic smoke test and exit')
    args = ap.parse_args()

    if args.smoketest:
        ok = smoke_test()
        sys.exit(0 if ok else 1)

    if args.nogui:
        t0 = time.time()
        R = run_experiments(DEFAULT_CFG)
        if args.figures:
            make_all_figures(R)
        log(f'Headless pipeline finished in {(time.time()-t0)/60:.1f} min; '
            f'forward calls = {FORWARD_STATS["calls"]} '
            f'({FORWARD_STATS["time"]/60:.1f} min)')
        return

    launch_gui()


if __name__ == '__main__':
    main()
