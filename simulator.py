#!/usr/bin/env python3
"""
Room simulator + live visualizer — 20 kHz, 6-mic array, 50 × 10 cm bar.

Signal processing pipeline per frame:
  1. Acoustic simulation of LFM chirp echoes with per-mic Doppler shift
  2. Matched filter (pulse compression) per mic → analytic envelope
  3. Per-mic time of flight (TX → object → mic) from the envelope peak,
     time-gated around the Kalman prediction, sub-sample via parabolic interp.
  4. 2D position via least-squares intersection of the bistatic range ellipses
  5. Constant-velocity Kalman filter (with innovation gating) → position + velocity

Room / clutter / wall mapping:
  * The simulated room has the direct speaker→mic sound, echoes from all four
    walls and second-order (corner / flutter) echoes, a cardioid speaker and
    a fixed noise floor. The transmitted chirp is Tukey-tapered (low range
    sidelobes, so the loud direct sound doesn't mask nearby walls).
  * A static background (median of the first CAL_FRAMES pings, then a slow
    running average that skips the tracked object and pauses while the track
    is lost) is subtracted before tracking — this removes the wall echoes.
  * Wall mapping (wallmap.py) runs on the same background every MAP_EVERY
    frames: walls, room outline and size. The mounting wall is known.
  * --object-multipath (EXPERIMENTAL) also simulates object↔wall bounces and
    tracks with a wall-map-based multipath model. Not yet robust: tracking
    can lose lock when the object is near walls/corners.

Panels:
  Left   — room top-down: true pos (yellow ●) + estimated pos (cyan ◆) + vel arrow
  Top-R  — per-mic envelope history
  Bot-R  — estimated vs true trajectory + speed gauge

Usage:
    python simulator.py                 # live visualizer
    python simulator.py --eval 300      # headless accuracy report over 300 frames
    python simulator.py --eval 300 --seed 1
    python simulator.py --room 4.5 3.2     # different room size (W × D, m)
    python simulator.py --object-multipath # experimental object↔wall bounces

Requires:
    pip install numpy scipy matplotlib   (or ./setupenv.sh)
"""

import argparse

_ap = argparse.ArgumentParser(description="Ultrasonic tracker simulator")
_ap.add_argument("--eval", type=int, metavar="N", default=0,
                 help="run N frames headless and print accuracy statistics")
_ap.add_argument("--seed", type=int, default=None, help="random seed")
_ap.add_argument("--fs", type=int, default=48_000,
                 help="sample rate in Hz (default 48000; e.g. 96000, 192000)")
_ap.add_argument("--band", type=float, nargs=2, metavar=("F0", "F1"),
                 default=(18_000, 22_000), help="chirp start/end frequency in Hz")
_ap.add_argument("--object-multipath", action="store_true",
                 help="EXPERIMENTAL: also simulate object↔wall bounces and "
                      "track with the wall-map multipath model (not yet robust)")
_ap.add_argument("--room", type=float, nargs=2, metavar=("W", "D"),
                 default=(3.0, 2.0), help="room size in metres (default 3 × 2)")
ARGS = _ap.parse_args()

import numpy as np
import scipy.optimize
import matplotlib
if ARGS.eval:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, sosfilt, hilbert as sp_hilbert
from collections import deque

import wallmap

if ARGS.seed is not None:
    np.random.seed(ARGS.seed)

# ─────────────────────────────── constants ────────────────────────────────────
C         = 343.0
FS        = ARGS.fs     # Hz
NOISE_STD = 0.004       # fixed noise floor (mic + ADC), same units as echo amplitude

# ── simulated room acoustics ─────────────────────────────────────────────────
DIRECT_GAIN = 0.05      # speaker → mic leakage, amplitude = DIRECT_GAIN / distance
WALL_GAIN   = 0.30      # wall echo amplitude = WALL_GAIN · WALL_REFL^order / path
WALL_REFL   = 0.7       # pressure reflection coefficient per bounce
OBJ_GAIN    = 0.65      # object echo amplitude = OBJ_GAIN / path²
TX_AXIS     = np.array([1.0, 0.0])   # speaker faces into the room (+x)


def tx_gain(direction):
    """Cardioid speaker directivity: 1 on axis, 0.5 sideways, 0 backwards."""
    d = np.asarray(direction, float)
    return 0.5 * (1.0 + float(d @ TX_AXIS) / max(np.linalg.norm(d), 1e-9))

# ── LFM chirp parameters ──────────────────────────────────────────────────────
CHIRP_F0, CHIRP_F1 = map(float, ARGS.band)   # start / end frequency  Hz
CHIRP_BW  = CHIRP_F1 - CHIRP_F0          # 4 kHz bandwidth by default
if not 0 < CHIRP_F0 < CHIRP_F1 < FS / 2:
    _ap.error(f"chirp band {CHIRP_F0:.0f}-{CHIRP_F1:.0f} Hz must lie below "
              f"Nyquist ({FS/2:.0f} Hz) for --fs {FS}")
CHIRP_T   = 0.005       # 5 ms chirp duration
CHIRP_FC  = (CHIRP_F0 + CHIRP_F1) / 2    # 20 kHz centre (for Doppler display)
# Range resolution = C / (2 * BW) ≈ 43 mm
RANGE_RES = C / (2 * CHIRP_BW)

ROOM_W, ROOM_D = map(float, ARGS.room)

# ── bar / array geometry — mounted on left short wall ────────────────────────
BAR_LEN  = 0.50
BAR_H    = 0.10
ARRAY_X  = 0.05                        # 5 cm from left wall
ARRAY_CY = ROOM_D / 2                  # centred vertically
BAR_BOTTOM = ARRAY_CY - BAR_LEN / 2   # 0.75 m

#   M1(0,0)  M2(10,0)  M3(0,10)  M4(10,10)  M5(0,20)  M6(10,20)  [bar cm]
#   x = depth from wall, y = position along wall
_bx = np.array([0.00, 0.10, 0.00, 0.10, 0.00, 0.10])   # depth into room
_by = np.array([0.00, 0.00, 0.10, 0.10, 0.20, 0.20])   # along wall
MIC_POS = np.column_stack([ARRAY_X + _bx, BAR_BOTTOM + _by])   # (6, 2)
N_MICS  = 6
TX_POS  = np.array([ARRAY_X, ARRAY_CY + 0.20])   # 20 cm above array centre

PAIR_COLORS = ["#e74c3c","#2ecc71","#3498db"]
MIC_COLORS  = ["#e74c3c","#c0392b","#2ecc71","#27ae60","#3498db","#2980b9"]

MOTION_THRESH  = 0.18
BASELINE_ALPHA = 0.06
TRAIL_LEN      = 80
ANIM_INTERVAL  = 100        # ms per frame

# ────────────────────────────── DSP setup ─────────────────────────────────────
from scipy.signal import chirp as scipy_chirp

BURST_LEN  = int(FS * CHIRP_T)                        # 240 samples @ 48 kHz
TX_TAPER   = 0.5        # Tukey taper fraction of the *transmitted* chirp


def tx_window(t):
    """Continuous Tukey window over the chirp (t in seconds). Tapering the
    transmitted chirp (no hard on/off edges) lowers the far range sidelobes
    from ≈ −28 dB to ≈ −57 dB — needed so the loud direct sound does not
    mask nearby wall echoes. Also avoids speaker clicks."""
    a = TX_TAPER * CHIRP_T / 2
    w = np.ones_like(t)
    lo, hi = t < a, t > CHIRP_T - a
    w[lo] = 0.5 * (1 - np.cos(np.pi * t[lo] / a))
    w[hi] = 0.5 * (1 - np.cos(np.pi * (CHIRP_T - t[hi]) / a))
    return w


_t_burst   = np.arange(BURST_LEN) / FS
BURST      = (scipy_chirp(_t_burst, f0=CHIRP_F0, f1=CHIRP_F1,
                          t1=CHIRP_T, method='linear')
              * tx_window(_t_burst)).astype(np.float32)
# Hann-weighted reference for low sidelobes
BURST_WIN  = (BURST * np.hanning(BURST_LEN)).astype(np.float32)

# reception window: long enough for the corner echoes (wall mapping) + burst
SIM_SAMPLES = int(FS * (2*np.hypot(ROOM_W, ROOM_D)/C + CHIRP_T)) + 500

# Pre-compute conjugate spectrum of reference chirp for fast matched filtering
_MF_N      = 1 << (SIM_SAMPLES + BURST_LEN - 1).bit_length()
BURST_CONJ = np.conj(np.fft.rfft(BURST_WIN.astype(np.float64), n=_MF_N))

# Time-of-flight measurement model
TOA_SIGMA_M  = 0.003   # assumed 1σ error of one bistatic path length (m); sim gives ~1-2 mm
# LFM range-Doppler coupling: a Doppler shift moves the matched-filter peak in
# time. Measured path error = DOPPLER_COUPLING_S × d(path)/dt  (≈ 25 mm per m/s),
# so the position fix lands at the *apparent* position  pos + DOPPLER_COUPLING_S·vel.
# The Kalman filter models this exactly (H = [I, k·I]) instead of correcting
# the paths with its own velocity guess, which would form an unstable loop.
DOPPLER_COUPLING_S = CHIRP_FC * CHIRP_T / CHIRP_BW
MIN_GATE_M   = 0.15    # minimum ± search window around predicted path length (m)


# ─────────────────────────── DSP helpers ──────────────────────────────────────
def _parabolic_peak(arr, idx):
    """Sub-sample peak index via parabolic interpolation."""
    if idx <= 0 or idx >= len(arr)-1:
        return float(idx)
    a, b, c = arr[idx-1], arr[idx], arr[idx+1]
    denom = a - 2*b + c
    if abs(denom) < 1e-12:
        return float(idx)
    return float(idx) - 0.5*(c-a)/denom


def matched_filter_real(sig):
    """Matched filter (cross-correlation with the windowed chirp).
    Output index k corresponds to an echo delay of k samples."""
    SIG = np.fft.rfft(sig.astype(np.float64), n=_MF_N)
    mf  = np.fft.irfft(SIG * BURST_CONJ, n=_MF_N)
    return mf[:len(sig)]


def matched_filter_env(sig):
    """Analytic envelope of the matched filter output — one smooth peak per
    echo, free of the 20 kHz carrier ripple."""
    return np.abs(sp_hilbert(matched_filter_real(sig)))


def bistatic_paths(pos):
    """TX → pos → each mic path lengths (m), shape (N_MICS,)."""
    return (np.linalg.norm(pos - TX_POS)
            + np.linalg.norm(MIC_POS - pos, axis=1))


def measure_paths(mf_envs, predicted_pos, gate_m):
    """
    Per-mic time of flight → bistatic path length.
    Searches each envelope within ±gate_m of the path length predicted for
    predicted_pos (rejects wall echoes / clutter), then refines the peak to
    sub-sample precision.
    Returns (paths_m, peak_amplitudes).
    """
    exp_paths = bistatic_paths(predicted_pos)
    g = max(int(gate_m / C * FS), 4)
    paths = np.empty(N_MICS)
    peaks = np.empty(N_MICS)
    for i, env in enumerate(mf_envs):
        c  = int(round(exp_paths[i] / C * FS))
        lo = max(c - g, 1)
        hi = min(c + g, len(env) - 2)
        if lo >= hi:                       # prediction outside the record
            lo, hi = 1, len(env) - 2
        k = lo + int(np.argmax(env[lo:hi + 1]))
        peaks[i] = env[k]
        paths[i] = _parabolic_peak(env, k) / FS * C
    return paths, peaks


def estimate_position(paths, x0):
    """
    Nonlinear least-squares 2D position: the point whose TX→pos→mic path
    lengths best match the measured ones (intersection of 6 ellipses with
    foci TX and mic_i). Warm-started at x0 with a room-centre fallback.
    Returns (pos, rms_residual_m).
    """
    def residuals(pos):
        return bistatic_paths(pos) - paths

    lo_b = np.array([ARRAY_X + 0.02, -20.0])  # object can't be behind the bar
    hi_b = np.array([20.0, 20.0])             # room size is not assumed known
    best = None
    for s in (x0, np.array([ROOM_W/2, ROOM_D/2])):
        s = np.clip(s, lo_b + 0.01, hi_b - 0.01)
        res = scipy.optimize.least_squares(
            residuals, s, bounds=(lo_b, hi_b),
            method="trf", max_nfev=100, ftol=1e-10, xtol=1e-10)
        if best is None or res.cost < best.cost:
            best = res
    return best.x, float(np.sqrt(np.mean(best.fun**2)))


def path_jacobian(pos):
    """d(path_i)/d(pos), shape (N_MICS, 2): sum of unit vectors from TX and
    from mic_i towards pos."""
    u_tx  = (pos - TX_POS) / max(np.linalg.norm(pos - TX_POS), 1e-6)
    d_mic = pos - MIC_POS
    u_mic = d_mic / np.maximum(np.linalg.norm(d_mic, axis=1, keepdims=True), 1e-6)
    return u_tx + u_mic


def position_covariance(pos):
    """Measurement covariance of the position fix from the TOA geometry
    (Gauss-Newton: σ² (JᵀJ)⁻¹). Used as the Kalman R so the filter trusts
    each axis as much as the geometry allows."""
    J = path_jacobian(pos)
    return TOA_SIGMA_M**2 * np.linalg.inv(J.T @ J + 1e-9*np.eye(2))


# ─────────────────────────── acoustic simulation ──────────────────────────────
def _add_chirp(out, ch, delay_s, amp, scale=1.0):
    """Add one echo of the chirp arriving at an exact (fractional) delay.
    The chirp is evaluated analytically at t = n/FS - delay_s, so arrival times
    are not rounded to whole samples — like real sound."""
    n0 = int(np.floor(delay_s * FS))
    n  = np.arange(n0, n0 + BURST_LEN + 2)
    n  = n[(n >= 0) & (n < out.shape[0])]
    t  = n / FS - delay_s
    ok = (t >= 0) & (t < CHIRP_T)
    t, n = t[ok], n[ok]
    phase = 2*np.pi * (CHIRP_F0 * scale * t +
                       0.5 * (CHIRP_BW / CHIRP_T) * scale * t**2)
    out[n, ch] += (np.cos(phase) * tx_window(t) * amp).astype(np.float32)


# true room walls (normals point into the room)
TRUE_WALLS = [
    wallmap.Wall(np.array([ 1.0, 0.0]), 0.0),        # left  (mounting wall)
    wallmap.Wall(np.array([-1.0, 0.0]), -ROOM_W),    # right
    wallmap.Wall(np.array([ 0.0, 1.0]), 0.0),        # bottom
    wallmap.Wall(np.array([ 0.0,-1.0]), -ROOM_D),    # top
]
WALL_NAMES = ["left (mount)", "right", "bottom", "top"]


def _image_sources(src, max_order=2):
    """Image sources of src for reflection sequences up to max_order
    (exact for a rectangular room). Returns [(position, order, L)] where L
    maps an arrival direction (mic − image) back to the emission direction
    at the real source (for the speaker directivity)."""
    found = {}
    frontier = [(src, -1, np.eye(2))]
    for order in range(1, max_order + 1):
        nxt = []
        for p, last, L in frontier:
            for wi, w in enumerate(TRUE_WALLS):
                if wi == last:
                    continue
                q  = wallmap.reflect(p, w)
                Ln = L @ (np.eye(2) - 2 * np.outer(w.normal, w.normal))
                key = tuple(np.round(q, 6))
                if key not in found:
                    found[key] = (q, order, Ln)
                    nxt.append((q, wi, Ln))
        frontier = nxt
    return list(found.values())


def _static_signal():
    """Direct speaker→mic sound + wall/corner echoes (no noise). Identical
    every ping, so it is computed once."""
    out = np.zeros((SIM_SAMPLES, N_MICS), dtype=np.float32)
    for i, mp in enumerate(MIC_POS):
        d = float(np.linalg.norm(mp - TX_POS))
        _add_chirp(out, i, d / C, tx_gain(mp - TX_POS) * DIRECT_GAIN / max(d, 0.02))
    for q, order, L in _image_sources(TX_POS):
        for i, mp in enumerate(MIC_POS):
            d = float(np.linalg.norm(mp - q))
            g = tx_gain(L @ (mp - q))
            _add_chirp(out, i, d / C, g * WALL_GAIN * WALL_REFL**order / d)
    return out


def simulate_received(obj_pos, obj_vel):
    """
    One ping: static room response + object echo (with Doppler) +
    object↔wall multipath + fixed Gaussian noise floor.
    """
    out  = STATIC_SIGNAL.copy()
    d_tx = float(np.linalg.norm(obj_pos - TX_POS))
    n_tx_to_obj = (obj_pos - TX_POS) / max(d_tx, 1e-6)

    for i, mp in enumerate(MIC_POS):
        d_mic   = float(np.linalg.norm(obj_pos - mp))
        d_total = d_tx + d_mic
        amp     = tx_gain(obj_pos - TX_POS) * min(OBJ_GAIN / max(d_total**2, 0.01), 3.0)

        # Doppler: scale instantaneous frequency of chirp by (1 + v_r/c)
        n_obj_to_mic = (mp - obj_pos) / max(d_mic, 1e-6)
        v_r = (float(np.dot(obj_vel, -n_tx_to_obj)) +
               float(np.dot(obj_vel, n_obj_to_mic)))
        _add_chirp(out, i, d_total / C, amp, 1.0 + v_r / C)

    # object ↔ wall multipath: TX→obj→wall→mic and TX→wall→obj→mic
    # (moves with the object, so background subtraction does not remove it)
    # (Doppler from each bounce path's own rate of change)
    eps = 1e-4
    for w in (TRUE_WALLS[1:] if ARGS.object_multipath else []):
        tx_img = wallmap.reflect(TX_POS, w)
        for i, mp in enumerate(MIC_POS):
            mic_img = wallmap.reflect(mp, w)
            for path in (lambda q: np.linalg.norm(q - TX_POS) + np.linalg.norm(q - mic_img),
                         lambda q: np.linalg.norm(q - tx_img) + np.linalg.norm(q - mp)):
                d_total = float(path(obj_pos))
                rate = (float(path(obj_pos + obj_vel * eps)) - d_total) / eps
                _add_chirp(out, i, d_total / C, tx_gain(obj_pos - TX_POS)
                           * WALL_REFL * OBJ_GAIN / max(d_total**2, 0.01),
                           1.0 - rate / C)

    out += (np.random.randn(*out.shape) * NOISE_STD).astype(np.float32)
    return out


# ── multipath-aware refinement (uses the wall map) ────────────────────────────
PULSE          = wallmap.PulseModel(BURST, BURST_WIN)
MP_FIT_HALF_M  = 0.15     # ± fit window around each echo (path metres)
MP_MAX_EXTRA_M = 0.60     # model bounces arriving up to this much after direct


def echo_paths(p, walls):
    """(n_paths, n_mics) path lengths: direct TX→p→mic, plus for every wall
    TX→p→wall→mic and TX→wall→p→mic."""
    d_tx  = np.linalg.norm(p - TX_POS)
    rows  = [d_tx + np.linalg.norm(MIC_POS - p, axis=1)]
    for w in walls:
        mic_img = MIC_POS - 2 * (MIC_POS @ w.normal - w.offset)[:, None] * w.normal
        tx_img  = wallmap.reflect(TX_POS, w)
        rows.append(d_tx + np.linalg.norm(mic_img - p, axis=1))
        rows.append(np.linalg.norm(p - tx_img) + np.linalg.norm(MIC_POS - p, axis=1))
    return np.array(rows)


MP_BETAS = (0.0, 0.5, 0.8)   # candidate wall-bounce / direct amplitude ratios


def explained_energy(z, p, walls, betas=MP_BETAS):
    """Echo energy explained by an object at (apparent) position p.
    Per mic the echo shape is  pulse(direct) + β · Σ pulse(bounce)  for the
    wall bounces arriving close behind the direct echo, with one complex
    amplitude per mic and β shared by all mics (picked from `betas`).
    Free per-bounce amplitudes would let a wrong position near a wall 'fit'
    the merged echo — the shared β keeps the model physical."""
    P = echo_paths(p, walls) / C * FS
    g = int(MP_FIT_HALF_M / C * FS) + 2
    extra = MP_MAX_EXTRA_M / C * FS
    segs = []
    for i in range(N_MICS):
        taus = P[:, i]
        near = taus[1:][np.abs(taus[1:] - taus[0]) < extra]
        lo = max(int(min(taus[0], *near) if len(near) else taus[0]) - g, 0)
        hi = min(int(max(taus[0], *near) if len(near) else taus[0]) + g + 1, z.shape[1])
        if hi - lo < 4:
            return 0.0
        k = np.arange(lo, hi)
        d = PULSE(k - taus[0])
        bnc = sum(PULSE(k - t) for t in near) if len(near) else np.zeros_like(d)
        segs.append((z[i, lo:hi], d, bnc))
    best = 0.0
    for beta in betas:
        e = 0.0
        for y, d, bnc in segs:
            t = d + beta * bnc
            tt = float(np.vdot(t, t).real)
            if tt > 0:
                e += abs(np.vdot(t, y))**2 / tt
        best = max(best, e)
    return best


def bounces_overlap(p, walls):
    """True if any wall bounce arrives within MP_MAX_EXTRA_M of the direct
    echo at some mic — only then is the (slower) multipath fit needed."""
    P = echo_paths(p, walls)
    return bool(np.any(np.abs(P[1:] - P[0]) < MP_MAX_EXTRA_M))


MP_WALL_CLEARANCE = 0.10   # an object (person) can't be closer to a wall


def refine_with_multipath(z, starts, walls):
    """Maximise the explained echo energy over the object position, from
    each start point; returns the best position. Positions closer than
    MP_WALL_CLEARANCE to a wall are excluded — there the direct echo and its
    bounce coincide, and a single echo would 'explain' the merged peak."""
    def cost(p):
        if any(w.signed_distance(p) < MP_WALL_CLEARANCE for w in walls):
            return 0.0
        return -explained_energy(z, p, walls)
    best_p, best_e = None, -1.0
    for p0 in starts:
        p0 = clamp_inside(p0, walls, MP_WALL_CLEARANCE + 0.01)
        res = scipy.optimize.minimize(
            cost, p0, method="Nelder-Mead",
            options=dict(xatol=2e-4, fatol=1e-12,
                         initial_simplex=[p0, p0 + [0.02, 0], p0 + [0, 0.02]]))
        if -res.fun > best_e:
            best_p, best_e = res.x, -res.fun
    return best_p


def process(signals, predicted_pos, pos_uncertainty_m=1.0, walls=None):
    """
    Chirp pipeline: matched filter per channel → per-mic time of flight
    (gated around the predicted position) → least-squares position fix →
    (if a wall map is available) multipath-aware refinement.
    No bandpass needed — the matched filter is inherently selective.
    """
    mf_real = [matched_filter_real(signals[:, ch]) for ch in range(N_MICS)]
    z       = np.array([sp_hilbert(m) for m in mf_real])
    mf_envs = np.abs(z)
    envs    = mf_envs.mean(axis=1)

    # path length changes by up to ~2× the position error → gate accordingly
    gate_m = max(3.0 * pos_uncertainty_m, MIN_GATE_M)
    paths, peaks = measure_paths(mf_envs, predicted_pos, gate_m)

    pos_est, cost = estimate_position(paths, predicted_pos)
    if walls:
        pos_est = clamp_inside(pos_est, walls)
    if walls and (bounces_overlap(pos_est, walls) or bounces_overlap(predicted_pos, walls)):
        pos_est = refine_with_multipath(z, [pos_est, predicted_pos], walls)
        cost = float(np.sqrt(np.mean((bistatic_paths(pos_est) - paths)**2)))
    return mf_envs, envs, paths, pos_est, cost


STATIC_SIGNAL = _static_signal()


# ─────────────────────────── static background ────────────────────────────────
CAL_FRAMES   = 20      # median of the first 20 pings (2 s) = initial background
BG_ALPHA     = 0.02    # slow running-average update afterwards
BG_GUARD_M   = 0.30    # don't update the background within ±this of the object
MAP_EVERY    = 50      # re-run wall mapping every N frames
USE_MULTIPATH = ARGS.object_multipath   # model wall bounces when tracking (experimental)


class Background:
    """Static echo profile per mic (raw signal domain). Starts as the median
    of the first CAL_FRAMES pings (robust to a moving object), then tracks
    slow changes with an EMA that skips samples near the tracked object."""

    def __init__(self):
        self.cal = []
        self.value = None

    @property
    def ready(self):
        return self.value is not None

    def add(self, raw, obj_paths=None):
        if not self.ready:
            self.cal.append(raw.copy())
            if len(self.cal) >= CAL_FRAMES:
                self.value = np.median(np.stack(self.cal), axis=0)
                self.cal = []
            return
        upd = np.ones(raw.shape, bool)
        if obj_paths is not None:           # mask the object's echo per mic
            g = int(BG_GUARD_M / C * FS)
            for i, d in enumerate(obj_paths):
                k = int(d / C * FS)
                upd[max(k - g, 0):k + BURST_LEN + g, i] = False
        self.value[upd] += BG_ALPHA * (raw[upd] - self.value[upd])


BACKGROUND = Background()
MOUNT_WALL = wallmap.Wall(np.array([1.0, 0.0]), 0.0, known=True)
WALL_MAP   = [None]


def map_walls(background=None):
    """Estimate walls + room outline from the static background."""
    bg = BACKGROUND.value if background is None else background
    z = np.array([sp_hilbert(matched_filter_real(bg[:, i])) for i in range(N_MICS)])
    WALL_MAP[0] = wallmap.find_walls(z, MIC_POS, TX_POS, FS, C,
                                     tx_pulse=BURST, ref=BURST_WIN,
                                     known_walls=[MOUNT_WALL])
    return WALL_MAP[0]


# live mode: re-map in a worker thread so the animation doesn't stall
# (eval mode stays synchronous → reproducible)
import concurrent.futures as _cf
_MAP_POOL = None if ARGS.eval else _cf.ThreadPoolExecutor(max_workers=1)
_MAP_JOB  = [None]


def schedule_map():
    if _MAP_POOL is None:
        map_walls()
    elif _MAP_JOB[0] is None or _MAP_JOB[0].done():
        _MAP_JOB[0] = _MAP_POOL.submit(map_walls, BACKGROUND.value.copy())


# ─────────────────────────── object motion ────────────────────────────────────
def obj_pos_at(t):
    # Object moves in depth (x) and laterally (y) relative to wall-mounted array
    x = ROOM_W*0.35 + (ROOM_W*0.55)*(0.5 + 0.5*np.sin(2*np.pi*t/5.0 - np.pi/2))
    y = ROOM_D/2    + (ROOM_D/2-0.25)*np.sin(2*np.pi*t/10.0)
    return np.array([x, y])


def obj_vel_at(t, dt=0.005):
    return (obj_pos_at(t+dt) - obj_pos_at(t-dt)) / (2*dt)


# ─────────────────────────── figure layout ────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(16, 9), facecolor="#111111")
fig.canvas.manager.set_window_title(f"Ultrasonic Motion Detector — Simulator  [LFM Chirp {CHIRP_F0/1e3:.0f}-{CHIRP_F1/1e3:.0f} kHz | {FS/1e3:.0f} kHz SR]")
gs = GridSpec(2, 2, figure=fig,
              left=0.04, right=0.97, top=0.93, bottom=0.07,
              wspace=0.30, hspace=0.40, width_ratios=[1.6, 1])

ax_room  = fig.add_subplot(gs[:, 0])
ax_sigs  = fig.add_subplot(gs[0, 1])
ax_track = fig.add_subplot(gs[1, 1])


# ── room panel ────────────────────────────────────────────────────────────────
ax_room.set_xlim(-0.1, ROOM_W+0.1)
ax_room.set_ylim(-0.22, ROOM_D+0.2)
ax_room.set_aspect("equal")
ax_room.set_facecolor("#1a1a2e")
ax_room.set_title(f"Room View  (3 m × 2 m)  |  LFM chirp {CHIRP_F0/1e3:.0f}–{CHIRP_F1/1e3:.0f} kHz  |  time-of-flight positioning",
                  color="white", fontsize=12, pad=8)
ax_room.set_xlabel("x  (m)", color="#aaaaaa")
ax_room.set_ylabel("y  (m)", color="#aaaaaa")
ax_room.tick_params(colors="#777777")
for s in ax_room.spines.values():
    s.set_edgecolor("#444444")

ax_room.add_patch(mpatches.FancyBboxPatch(
    (0,0), ROOM_W, ROOM_D, boxstyle="square,pad=0",
    linewidth=2, edgecolor="#55aaff", facecolor="none"))
for gx in np.arange(0, ROOM_W+0.01, 0.5):
    ax_room.axvline(gx, color="#ffffff06", linewidth=0.5)
for gy in np.arange(0, ROOM_D+0.01, 0.5):
    ax_room.axhline(gy, color="#ffffff06", linewidth=0.5)

# bar — vertical on left short wall
ax_room.add_patch(mpatches.FancyBboxPatch(
    (0, BAR_BOTTOM), BAR_H, BAR_LEN, boxstyle="square,pad=0",
    linewidth=1.5, edgecolor="#888888", facecolor="#1e1e3a"))
ax_room.text(BAR_H/2, ARRAY_CY,
             "50 cm\nbar", color="#555555", fontsize=6, ha="center", va="center")

# within-pair connectors (depth pairs: connect in x direction)
for idx, (a,b) in enumerate([(0,1),(2,3),(4,5)]):
    ax_room.plot([MIC_POS[a,0],MIC_POS[b,0]],[MIC_POS[a,1],MIC_POS[b,1]],
                 "-", color=PAIR_COLORS[idx], lw=2, alpha=0.5, zorder=3)

# mic dots
for i, (mx,my) in enumerate(MIC_POS):
    ax_room.plot(mx, my, "o", color=MIC_COLORS[i], ms=10, zorder=5)
    off = -0.07 if mx < ARRAY_X + 0.06 else 0.09
    ax_room.text(mx+off, my, f"M{i+1}", color=MIC_COLORS[i],
                 fontsize=7.5, ha="left" if off>0 else "right", va="center")

# tx marker
ax_room.plot(*TX_POS, marker=">", color="#f39c12", ms=14, zorder=5)
ax_room.text(TX_POS[0]+0.08, TX_POS[1], "TX", color="#f39c12", fontsize=8, ha="left", va="center")

# pair labels
for idx,(a,b) in enumerate([(0,1),(2,3),(4,5)]):
    ax_room.text(-0.08, MIC_POS[a,1], f"P{idx+1}",
                 color=PAIR_COLORS[idx], fontsize=7, ha="center", fontweight="bold")

# mutable: true trail/dot
true_trail, = ax_room.plot([], [], "-", color="#ffcc00", alpha=0.35, lw=1.5, zorder=3)
true_dot,   = ax_room.plot([], [], "o", color="#ffcc00", ms=14, zorder=6,
                            markeredgecolor="white", markeredgewidth=1.5)

# mutable: estimated trail/dot (Kalman smoothed)
est_trail,  = ax_room.plot([], [], "-", color="#00ccff", alpha=0.35, lw=1.5, zorder=3)
est_dot,    = ax_room.plot([], [], "D", color="#00ccff", ms=11, zorder=7,
                            markeredgecolor="white", markeredgewidth=1.5)

# mutable: raw (pre-Kalman) position dot + trail
raw_trail,  = ax_room.plot([], [], "-", color="#ff6b35", alpha=0.25, lw=1.0, zorder=2)
raw_dot,    = ax_room.plot([], [], "x", color="#ff6b35", ms=9, zorder=6,
                            markeredgewidth=2.0)

# mutable: velocity arrow at estimated position (quiver)
vel_quiver = ax_room.quiver(0, 0, 0, 0, color="#44ff88", scale=8, scale_units="xy",
                             width=0.005, zorder=8, headwidth=5)

# mutable: estimated room outline (wall map)
wall_line, = ax_room.plot([], [], "--", color="#ff66cc", lw=2.0, zorder=4)
wall_txt = ax_room.text(0.98, 0.03, "", transform=ax_room.transAxes,
                        color="#ff66cc", fontsize=8.5, va="bottom", ha="right",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#00000099",
                                  edgecolor="#444444"))

# text overlay
info_txt = ax_room.text(
    0.02, 0.97, "", transform=ax_room.transAxes,
    color="white", fontsize=8.5, va="top", fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#00000099", edgecolor="#444444"))


# ── envelope panel ─────────────────────────────────────────────────────────────
HISTORY     = 60
env_history = np.zeros((HISTORY, N_MICS))

ax_sigs.set_facecolor("#1a1a2e")
ax_sigs.set_title("Envelope per Mic", color="white", fontsize=11)
ax_sigs.set_xlabel("Frame →", color="#aaaaaa")
ax_sigs.set_ylabel("Norm. amplitude", color="#aaaaaa")
ax_sigs.tick_params(colors="#777777")
for s in ax_sigs.spines.values():
    s.set_edgecolor("#444444")
sig_lines = []
for i in range(N_MICS):
    ln, = ax_sigs.plot([], [], "-", color=MIC_COLORS[i], lw=1.5, label=f"M{i+1}", alpha=0.9)
    sig_lines.append(ln)
ax_sigs.legend(loc="upper right", fontsize=7, framealpha=0.3, ncol=2, labelcolor="white")
ax_sigs.set_xlim(0, HISTORY-1)
ax_sigs.set_ylim(0, 1)


# ── tracking / speed panel ─────────────────────────────────────────────────────
ax_track.set_facecolor("#1a1a2e")
ax_track.set_title("Position tracking  +  speed", color="white", fontsize=11)
ax_track.set_xlim(0, ROOM_W)
ax_track.set_ylim(0, ROOM_D)
ax_track.set_aspect("equal")
ax_track.set_xlabel("x  (m)", color="#aaaaaa")
ax_track.set_ylabel("y  (m)", color="#aaaaaa")
ax_track.tick_params(colors="#777777")
for s in ax_track.spines.values():
    s.set_edgecolor("#444444")

for gx in np.arange(0, ROOM_W+0.01, 0.5):
    ax_track.axvline(gx, color="#ffffff06", linewidth=0.5)
for gy in np.arange(0, ROOM_D+0.01, 0.5):
    ax_track.axhline(gy, color="#ffffff06", linewidth=0.5)

# array position indicator — vertical on left wall
ax_track.plot([ARRAY_X, ARRAY_X], [BAR_BOTTOM, BAR_BOTTOM+BAR_LEN],
              "-", color="#888888", lw=3, solid_capstyle="round")

tr_true_sc  = ax_track.scatter([], [], s=18, c="#ffcc00", alpha=0.4, zorder=3)
tr_est_sc   = ax_track.scatter([], [], s=18, c="#00ccff", alpha=0.4, zorder=3)
tr_raw_sc   = ax_track.scatter([], [], s=12, c="#ff6b35", alpha=0.3, zorder=2)
tr_true_now,= ax_track.plot([], [], "o", color="#ffcc00", ms=10, zorder=6,
                             markeredgecolor="white", markeredgewidth=1.2)
tr_est_now, = ax_track.plot([], [], "D", color="#00ccff", ms=8, zorder=7,
                             markeredgecolor="white", markeredgewidth=1.2)
tr_raw_now, = ax_track.plot([], [], "x", color="#ff6b35", ms=8, zorder=5,
                             markeredgewidth=2.0)
tr_vel_q    = ax_track.quiver(0,0,0,0, color="#44ff88", scale=8, scale_units="xy",
                               width=0.008, zorder=8, headwidth=5)

# speed text inside tracking panel
spd_txt = ax_track.text(0.98, 0.97, "", transform=ax_track.transAxes,
                         color="white", fontsize=8.5, va="top", ha="right",
                         fontfamily="monospace",
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="#00000099", edgecolor="#444444"))

legend_el = [
    mpatches.Patch(color="#ffcc00", label="True position"),
    mpatches.Patch(color="#00ccff", label="Kalman estimate"),
    mpatches.Patch(color="#ff6b35", label="Raw estimate"),
    mpatches.Patch(color="#44ff88", label="Velocity vector"),
]
ax_track.legend(handles=legend_el, loc="lower right", fontsize=7,
                framealpha=0.3, labelcolor="white")


# ─────────────────────────── animation state ──────────────────────────────────
TRUE_HIST  = deque(maxlen=TRAIL_LEN)
EST_HIST   = deque(maxlen=TRAIL_LEN)
RAW_HIST   = deque(maxlen=TRAIL_LEN)
POS_HIST   = deque(maxlen=8)          # for velocity-from-history
baseline_env = [None]
FRAME_DT     = ANIM_INTERVAL / 1000.0

# ── Kalman filter: constant velocity, white-acceleration process noise ───────
# The target can accelerate by up to ~1.3 m/s² on the simulated path.
KF_ACCEL_STD = 1.5                      # m/s², process noise strength
KF_GATE_CHI2 = 13.8                     # 99.9 % for 2 DOF — reject outliers
KF_MAX_MISSES = 5                       # re-acquire after this many rejects
_kf_x   = np.array([ROOM_W/2, ROOM_D/2, 0.0, 0.0])
_kf_P   = np.diag([1.0, 0.5, 0.20, 0.10])
# measurement = apparent position = pos + k·vel  (chirp range-Doppler coupling)
_kf_H   = np.array([[1,0,DOPPLER_COUPLING_S,0],[0,1,0,DOPPLER_COUPLING_S]])
_kf_misses = [0]


def _kf_F(dt):
    return np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)


def _kf_Q(dt):
    q = KF_ACCEL_STD**2
    g = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]]) * q
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = g      # x, vx
    Q[np.ix_([1, 3], [1, 3])] = g      # y, vy
    return Q


def kalman_predict(dt):
    """Predicted (pos, vel, pos_std) for the next frame — used to gate the
    echoes and to correct range-Doppler coupling."""
    xp = _kf_F(dt) @ _kf_x
    Pp = _kf_F(dt) @ _kf_P @ _kf_F(dt).T + _kf_Q(dt)
    return xp[:2].copy(), xp[2:].copy(), float(np.sqrt(max(Pp[0,0], Pp[1,1])))


KF_VMAX = 3.0                           # m/s — cap on the tracked speed
R_MAX_INFLATION = 100.0                 # measurement σ grows to ≤ 10× (≈ 3 cm)
TRACK_STARTED = [False]


def kalman_reset(pos):
    """Re-acquire: restart the track at pos with unknown velocity."""
    global _kf_x, _kf_P
    _kf_x = np.array([pos[0], pos[1], 0.0, 0.0])
    _kf_P = np.diag([0.02, 0.02, 0.5, 0.5])
    _kf_misses[0] = 0


def kalman_update(raw_pos, R, dt=0.1):
    """Predict + update. Measurements whose innovation fails the χ² gate are
    skipped (predict only) unless too many have been rejected in a row.
    Returns (pos, vel, accepted)."""
    global _kf_x, _kf_P
    F  = _kf_F(dt)
    xp = F @ _kf_x
    Pp = F @ _kf_P @ F.T + _kf_Q(dt)
    S  = _kf_H @ Pp @ _kf_H.T + R
    innov = raw_pos - _kf_H @ xp
    d2 = float(innov @ np.linalg.solve(S, innov))
    accepted = d2 <= KF_GATE_CHI2
    if accepted:
        K  = Pp @ _kf_H.T @ np.linalg.inv(S)
        _kf_x = xp + K @ innov
        _kf_P = (np.eye(4) - K @ _kf_H) @ Pp
        _kf_misses[0] = 0
    else:
        _kf_x, _kf_P = xp, Pp
        _kf_misses[0] += 1
    sp = np.linalg.norm(_kf_x[2:])
    if sp > KF_VMAX:
        _kf_x[2:] *= KF_VMAX / sp
    return _kf_x[:2].copy(), _kf_x[2:].copy(), accepted


def global_search(signals, walls, step_m=0.05, clearance=0.10):
    """Re-acquisition: score every grid point inside the mapped room by the
    geometric mean over mics of the echo envelope at its delay (every mic
    must see it); return the best point. Ghosts that only some mics see —
    or that are weak, like a wall-bounce residue — lose to the real echo."""
    env = np.array([matched_filter_env(signals[:, i]) for i in range(N_MICS)])
    poly = WALL_MAP[0].polygon
    xs = np.arange(poly[:, 0].min(), poly[:, 0].max(), step_m)
    ys = np.arange(poly[:, 1].min(), poly[:, 1].max(), step_m)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    inside = np.all([P @ w.normal - w.offset > clearance for w in walls], axis=0)
    P = P[inside]
    d_tx = np.linalg.norm(P - TX_POS, axis=1)
    n = np.arange(env.shape[1])
    floor = 1e-3 * env.max()
    log_s = np.zeros(len(P))
    for i, mp in enumerate(MIC_POS):
        k = (d_tx + np.linalg.norm(P - mp, axis=1)) / C * FS
        log_s += np.log(np.interp(k, n, env[i], right=0.0) + floor)
    return P[int(np.argmax(log_s))]


def clamp_inside(p, walls, margin=0.05):
    """Push p back inside every wall (by `margin`)."""
    p = np.array(p, float)
    for w in walls:
        sd = w.signed_distance(p)
        if sd < margin:
            p += (margin - sd) * w.normal
    return p


def step(frame):
    """Simulate + process one frame. Returns a dict of everything to display."""
    t   = frame * FRAME_DT
    pos = obj_pos_at(t)
    vel = obj_vel_at(t)

    raw = simulate_received(pos, vel)

    # ── static background: calibrate first, then subtract + slowly update ────
    if not BACKGROUND.ready:
        BACKGROUND.add(raw)
        if BACKGROUND.ready:
            map_walls()
        return dict(t=t, pos=pos, vel=vel, calibrating=True,
                    cal_progress=len(BACKGROUND.cal) / CAL_FRAMES)
    clutter_free = raw - BACKGROUND.value

    m = WALL_MAP[0]
    mapped = m is not None and m.closed
    walls = [w for w in m.walls if not w.known] if (mapped and USE_MULTIPATH) else None

    pred_pos, pred_vel, pred_std = kalman_predict(FRAME_DT)
    if mapped:                       # the object can't be beyond a wall
        pred_pos = clamp_inside(pred_pos, m.walls)
    reacquire = _kf_misses[0] >= KF_MAX_MISSES or not TRACK_STARTED[0]
    if reacquire:                    # lost: search the whole room
        start = (global_search(clutter_free, m.walls) if mapped
                 else np.array([ROOM_W/2, ROOM_D/2]))
        pred_pos, pred_vel, pred_std = start, np.zeros(2), (0.1 if mapped else 1.0)
    # gate the echoes around where they will *appear* (incl. Doppler offset)
    pred_app = pred_pos + DOPPLER_COUPLING_S * pred_vel
    mf_envs, envs, paths, pos_app, cost = process(
        clutter_free, pred_app, pos_uncertainty_m=pred_std, walls=walls)
    if frame % MAP_EVERY == 0:
        schedule_map()

    # trust the fix less when the per-mic peaks disagree with it (merged
    # echoes near walls) — otherwise a few cm of bias would be read as a
    # large velocity through the Doppler term of the measurement model
    R = position_covariance(pos_app) * float(np.clip((cost / TOA_SIGMA_M)**2,
                                                     1.0, R_MAX_INFLATION))
    if reacquire or not TRACK_STARTED[0]:
        kalman_reset(pos_app)
        TRACK_STARTED[0] = True
    pos_smooth, vel_smooth, accepted = kalman_update(pos_app, R, dt=FRAME_DT)
    # update the background only while the track is confirmed — otherwise
    # the protective mask is in the wrong place and the real object would be
    # absorbed into the background
    if accepted and not reacquire:
        BACKGROUND.add(raw, obj_paths=bistatic_paths(pos_app))
    # single-frame fix for display/eval, Doppler offset removed with the
    # filtered velocity
    pos_est = pos_app - DOPPLER_COUPLING_S * vel_smooth

    # velocity from finite differences of the filtered track (cross-check)
    POS_HIST.append(pos_smooth.copy())
    n_back = min(len(POS_HIST) - 1, 5)
    vel_hist = ((POS_HIST[-1] - POS_HIST[-1 - n_back]) / (n_back * FRAME_DT)
                if n_back >= 2 else np.zeros(2))

    return dict(t=t, pos=pos, vel=vel, envs=envs, pos_est=pos_est, cost=cost,
                pos_smooth=pos_smooth, vel_smooth=vel_smooth,
                vel_hist=vel_hist, accepted=accepted, calibrating=False)


def wall_report(m):
    """Text summary of a wall map vs the true room."""
    if m is None:
        return "wall map: pending"
    depth, width, area = wallmap.room_size(m.polygon, MOUNT_WALL)
    n = sum(not w.known for w in m.walls)
    lines = [f"── WALL MAP ({n} walls{'' if m.closed else ', open'}) ──",
             f"  room  {depth:.3f} × {width:.3f} m" if m.closed else "  room  not closed",
             f"  true  {ROOM_W:.3f} × {ROOM_D:.3f} m"]
    return "\n".join(lines)


def draw_walls(m):
    if m is None or not m.closed:
        wall_line.set_data([], [])
    else:
        poly = np.vstack([m.polygon, m.polygon[:1]])
        wall_line.set_data(poly[:, 0], poly[:, 1])
    wall_txt.set_text(wall_report(m))


def update(frame):
    s = step(frame)
    draw_walls(WALL_MAP[0])
    if s["calibrating"]:
        info_txt.set_text(f"calibrating static background…\n"
                          f"  {s['cal_progress']*100:.0f} %  (keep the room as is)")
        info_txt.set_color("#cccccc")
        return [info_txt, wall_line, wall_txt]
    pos, vel, envs = s["pos"], s["vel"], s["envs"]
    pos_est, cost  = s["pos_est"], s["cost"]
    pos_smooth     = s["pos_smooth"]
    v_smooth       = s["vel_smooth"]
    speed_hist     = float(np.linalg.norm(s["vel_hist"]))

    # ── truth ─────────────────────────────────────────────────────────────────
    true_speed = float(np.linalg.norm(vel))
    pos_err    = float(np.linalg.norm(pos_smooth - pos))
    vel_err    = float(np.linalg.norm(v_smooth - vel))
    dir_deg    = float(np.degrees(np.arctan2(v_smooth[0], v_smooth[1])))

    # ── motion baseline ───────────────────────────────────────────────────────
    if baseline_env[0] is None:
        baseline_env[0] = envs.copy()
    motion = bool(np.any(np.abs(envs-baseline_env[0])/(baseline_env[0]+1e-9) > MOTION_THRESH))
    if not motion:
        baseline_env[0] = (1-BASELINE_ALPHA)*baseline_env[0] + BASELINE_ALPHA*envs

    # ── histories ─────────────────────────────────────────────────────────────
    TRUE_HIST.append(pos.copy())
    EST_HIST.append(pos_smooth.copy())
    RAW_HIST.append(pos_est.copy())

    tx = [p[0] for p in TRUE_HIST]
    ty = [p[1] for p in TRUE_HIST]
    ex = [p[0] for p in EST_HIST]
    ey = [p[1] for p in EST_HIST]
    rx = [p[0] for p in RAW_HIST]
    ry = [p[1] for p in RAW_HIST]

    # ── room view ─────────────────────────────────────────────────────────────
    true_trail.set_data(tx, ty)
    true_dot.set_data([pos[0]], [pos[1]])
    est_trail.set_data(ex, ey)
    est_dot.set_data([pos_smooth[0]], [pos_smooth[1]])
    raw_trail.set_data(rx, ry)
    raw_dot.set_data([pos_est[0]], [pos_est[1]])

    vel_quiver.set_offsets([pos_smooth])
    vel_quiver.set_UVC(v_smooth[0], v_smooth[1])

    # text overlay
    v_rad    = float(np.linalg.norm(v_smooth))
    lines = [
        "── TRUE ──────────────────",
        f"  pos  ({pos[0]:.2f}, {pos[1]:.2f}) m",
        f"  speed {true_speed:.2f} m/s",
        "── ESTIMATED (Kalman) ────",
        f"  pos  ({pos_smooth[0]:.2f}, {pos_smooth[1]:.2f}) m",
        f"  raw  ({pos_est[0]:.2f}, {pos_est[1]:.2f}) m",
        f"  error  {pos_err*100:.1f} cm",
        f"  speed (Kalman)  {v_rad:.2f} m/s",
        f"  speed (hist)    {speed_hist:.2f} m/s",
        f"  vel error  {vel_err:.2f} m/s",
        f"  direction  {dir_deg:.0f}°",
        "" if s["accepted"] else "  (measurement gated out)",
        "MOTION DETECTED" if motion else "monitoring...",
    ]
    info_txt.set_text("\n".join(lines))
    info_txt.set_color("#ff4444" if motion else "#cccccc")

    # ── envelope history ──────────────────────────────────────────────────────
    env_history[:-1] = env_history[1:]
    env_history[-1]  = envs
    norm = env_history / (env_history.max()+1e-9)
    ax_sigs.set_ylim(0, max(float(norm.max())*1.15, 0.1))
    xs = np.arange(HISTORY)
    for i, ln in enumerate(sig_lines):
        ln.set_data(xs, norm[:,i])

    # ── tracking panel ────────────────────────────────────────────────────────
    if len(tx) > 0:
        tr_true_sc.set_offsets(np.c_[tx, ty])
        tr_est_sc.set_offsets(np.c_[ex, ey])
        tr_raw_sc.set_offsets(np.c_[rx, ry])
    tr_true_now.set_data([pos[0]], [pos[1]])
    tr_est_now.set_data([pos_smooth[0]], [pos_smooth[1]])
    tr_raw_now.set_data([pos_est[0]], [pos_est[1]])
    tr_vel_q.set_offsets([pos_smooth])
    tr_vel_q.set_UVC(v_smooth[0], v_smooth[1])

    spd_txt.set_text(
        f"speed (Kalman)  {v_rad:.2f} m/s\n"
        f"speed (true)    {true_speed:.2f} m/s\n"
        f"direction  {dir_deg:.0f}°\n"
        f"pos error  {pos_err*100:.1f} cm\n"
        f"fit resid  {cost*1000:.1f} mm"
    )

    return ([true_trail, true_dot, est_trail, est_dot,
             raw_trail, raw_dot,
             vel_quiver, info_txt, spd_txt,
             tr_true_now, tr_est_now, tr_raw_now, tr_vel_q,
             tr_true_sc, tr_est_sc, tr_raw_sc, wall_line, wall_txt]
            + sig_lines)


# ─────────────────────────── title + run ──────────────────────────────────────
fig.suptitle(
    f"Ultrasonic Motion Detector  —  LFM Chirp {CHIRP_F0/1e3:.0f}–{CHIRP_F1/1e3:.0f} kHz | {FS/1e3:.0f} kHz SR  |  Matched Filter + TOF  |  2D positioning",
    color="white", fontsize=13, y=0.98
)

room_legend = [
    mpatches.Patch(color="#ffcc00", label="True pos"),
    mpatches.Patch(color="#00ccff", label="Kalman est."),
    mpatches.Patch(color="#ff6b35", label="Raw est."),
    mpatches.Patch(color="#44ff88", label="Velocity"),
    mpatches.Patch(color="#ff66cc", label="Wall map"),
]
ax_room.legend(handles=room_legend, loc="upper right",
               fontsize=7, framealpha=0.3, labelcolor="white", ncol=2)

def run_eval(n_frames, warmup=CAL_FRAMES + 20):
    """Headless accuracy report (no rendering)."""
    pe_raw, pe_kf, ve_kf, ex, ey, rejected = [], [], [], [], [], 0
    for f in range(n_frames):
        s = step(f)
        if f < warmup or s["calibrating"]:
            continue
        e = s["pos_smooth"] - s["pos"]
        pe_kf.append(np.linalg.norm(e)); ex.append(e[0]); ey.append(e[1])
        pe_raw.append(np.linalg.norm(s["pos_est"] - s["pos"]))
        ve_kf.append(np.linalg.norm(s["vel_smooth"] - s["vel"]))
        rejected += not s["accepted"]
    pe_raw, pe_kf, ve_kf = map(np.array, (pe_raw, pe_kf, ve_kf))
    cm = lambda a: a * 100
    print(f"frames evaluated: {len(pe_kf)}  (after {warmup} warm-up)")
    print(f"raw position     mean {cm(pe_raw.mean()):5.1f} cm  "
          f"p95 {cm(np.percentile(pe_raw, 95)):5.1f} cm  max {cm(pe_raw.max()):5.1f} cm")
    print(f"Kalman position  mean {cm(pe_kf.mean()):5.1f} cm  "
          f"p95 {cm(np.percentile(pe_kf, 95)):5.1f} cm  max {cm(pe_kf.max()):5.1f} cm")
    print(f"  RMS x (depth) {cm(np.sqrt(np.mean(np.square(ex)))):.1f} cm   "
          f"RMS y (lateral) {cm(np.sqrt(np.mean(np.square(ey)))):.1f} cm")
    print(f"Kalman velocity  mean {ve_kf.mean():.3f} m/s  "
          f"p95 {np.percentile(ve_kf, 95):.3f} m/s")
    print(f"gated-out measurements: {rejected}")

    m = WALL_MAP[0]
    print()
    print(wall_report(m).replace("──", "").strip())
    if m is not None:
        for name, tw in zip(WALL_NAMES, TRUE_WALLS):
            if name.startswith("left"):
                continue                       # known (mounting wall)
            near = [w for w in m.walls if not w.known
                    and np.degrees(np.arccos(np.clip(w.normal @ tw.normal, -1, 1))) < 15]
            if not near:
                print(f"  {name:7s} NOT FOUND")
                continue
            w = min(near, key=lambda w: abs(w.offset - tw.offset))
            ang = np.degrees(np.arccos(np.clip(w.normal @ tw.normal, -1, 1)))
            print(f"  {name:7s} distance error {(w.offset - tw.offset)*1000:+6.1f} mm   "
                  f"angle error {ang:.2f}° (raw fit {abs((w.raw_angle_deg - tw.angle_deg + 180) % 360 - 180):.2f}°)")
        extra = sum(not w.known for w in m.walls) - 3
        if extra > 0:
            print(f"  + {extra} spurious wall(s)")


if __name__ == "__main__" and ARGS.eval:
    run_eval(ARGS.eval)
elif __name__ == "__main__":
    ani = animation.FuncAnimation(
        fig, update, interval=ANIM_INTERVAL, blit=False, cache_frame_data=False
    )
    plt.show()
