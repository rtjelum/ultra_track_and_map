"""
Ultrasonic tracker — the signal-processing pipeline, independent of the simulator.

Feed it one ping at a time: the raw multichannel recording (n_samples, n_mics),
starting at the sample the chirp was sent. It returns whether something is in
the room that wasn't there during calibration, its distance, position and
velocity, and (every so often) a map of the walls.

The geometry is just data — mic and speaker positions in any 2D frame (room
coordinates, or the array's own frame on real hardware) plus, optionally, the
wall the array is mounted on. Nothing assumes a particular layout or where the
array sits in the room.

Per ping:
  0. Sync: find the direct speaker→mic sound (its delay is known from the
     geometry), measure the playback/recording latency, and shift the
     recording so sample 0 is the moment the chirp left the speaker; then
     lock it to the background with the carrier phase of the early echoes.
  1. Matched filter per mic → analytic envelope. Subtract the static
     background (median of the first CAL_FRAMES pings, then a slow update).
  2. Detection: the strongest residual echo per mic vs that mic's noise
     level. A new track needs most mics above DETECT_SNR; an existing track
     only needs ≥ 3 mics above MEAS_SNR at its predicted position.
  3. Acquisition (no track yet / track lost): grid search over the room for
     the point whose echo delays line up across all mics.
  4. Per-mic time of flight around the predicted position; mics whose echo
     is too weak are dropped.
  5. Robust least-squares fix (soft-L1 loss: one mic with a wrong peak
     doesn't drag the fix), kept in front of the mounting wall.
  6. Constant-velocity Kalman filter with χ² gating; models the chirp's
     range–Doppler coupling.
"""

import concurrent.futures as cf
import functools
import multiprocessing
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.optimize
from scipy.signal import chirp as sp_chirp

import wallmap

C = 343.0


# ───────────────────────────────── chirp ──────────────────────────────────────
@dataclass(frozen=True)
class Chirp:
    """Linear FM chirp, Tukey-tapered on transmit; Hann-weighted reference."""
    fs: float = 48_000
    f0: float = 18_000
    f1: float = 22_000
    duration: float = 0.005
    taper: float = 0.5           # Tukey taper fraction of the transmitted chirp

    def __post_init__(self):
        if not 0 < self.f0 < self.f1 < self.fs / 2:
            raise ValueError(f"chirp band {self.f0:.0f}-{self.f1:.0f} Hz must lie "
                             f"below Nyquist ({self.fs/2:.0f} Hz)")

    @property
    def n(self):
        return int(self.fs * self.duration)

    @property
    def bw(self):
        return self.f1 - self.f0

    @property
    def fc(self):
        return (self.f0 + self.f1) / 2

    @property
    def doppler_coupling_s(self):
        """LFM range-Doppler coupling: the echo peak shifts by this × the path
        rate (≈ 25 mm per m/s by default), so a fix lands at pos + k·vel."""
        return self.fc * self.duration / self.bw

    def window(self, t):
        """Continuous Tukey window (t in s). Tapering the transmitted chirp
        lowers the far range sidelobes from ≈ −28 dB to ≈ −57 dB, so the loud
        direct sound doesn't mask nearby echoes. Also avoids speaker clicks."""
        a = self.taper * self.duration / 2
        w = np.ones_like(t)
        lo, hi = t < a, t > self.duration - a
        w[lo] = 0.5 * (1 - np.cos(np.pi * t[lo] / a))
        w[hi] = 0.5 * (1 - np.cos(np.pi * (self.duration - t[hi]) / a))
        return w

    @functools.cached_property
    def burst(self):
        """Samples to transmit."""
        t = np.arange(self.n) / self.fs
        return (sp_chirp(t, f0=self.f0, f1=self.f1, t1=self.duration, method="linear")
                * self.window(t)).astype(np.float32)

    @functools.cached_property
    def ref(self):
        """Matched-filter reference (Hann-weighted for low sidelobes)."""
        return (self.burst * np.hanning(self.n)).astype(np.float32)


# ─────────────────────────────── helpers ──────────────────────────────────────
def _parabolic_peak(arr, idx):
    """Sub-sample peak index via parabolic interpolation."""
    if idx <= 0 or idx >= len(arr) - 1:
        return float(idx)
    a, b, c = arr[idx - 1], arr[idx], arr[idx + 1]
    denom = a - 2 * b + c
    return float(idx) if abs(denom) < 1e-12 else float(idx) - 0.5 * (c - a) / denom


def clamp_inside(p, walls, margin=0.05):
    """Push p back inside every wall (by `margin`)."""
    p = np.array(p, float)
    for w in walls:
        sd = w.signed_distance(p)
        if sd < margin:
            p += (margin - sd) * w.normal
    return p


# ─────────────────────────────── Kalman ───────────────────────────────────────
class KalmanCV:
    """Constant velocity, white-acceleration process noise. The measurement
    is the *apparent* position pos + k·vel (chirp range-Doppler coupling),
    modelled exactly (H = [I, k·I]) instead of correcting the paths with the
    filter's own velocity guess, which would form an unstable loop."""

    def __init__(self, coupling_s, accel_std=1.5, gate_chi2=13.8, max_misses=5, vmax=3.0):
        self.H = np.array([[1, 0, coupling_s, 0], [0, 1, 0, coupling_s]], float)
        self.accel_std, self.gate_chi2 = accel_std, gate_chi2
        self.max_misses, self.vmax = max_misses, vmax
        self.x, self.P = np.zeros(4), np.eye(4)
        self.misses, self.started = 0, False

    @property
    def lost(self):
        return not self.started or self.misses >= self.max_misses

    @staticmethod
    def F(dt):
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)

    def Q(self, dt):
        g = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * self.accel_std**2
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = g
        Q[np.ix_([1, 3], [1, 3])] = g
        return Q

    def _predicted(self, dt):
        F = self.F(dt)
        return F @ self.x, F @ self.P @ F.T + self.Q(dt)

    def predict(self, dt):
        """(pos, vel, pos_std) expected next ping — used to gate the echoes."""
        xp, Pp = self._predicted(dt)
        return xp[:2].copy(), xp[2:].copy(), float(np.sqrt(max(Pp[0, 0], Pp[1, 1])))

    def reset(self, pos):
        self.x = np.array([pos[0], pos[1], 0.0, 0.0])
        self.P = np.diag([0.02, 0.02, 0.5, 0.5])
        self.misses, self.started = 0, True

    def coast(self, dt):
        """No usable measurement: predict only."""
        self.x, self.P = self._predicted(dt)
        self.misses += 1

    def update(self, z, R, dt):
        """Predict + update; measurements failing the χ² gate are skipped."""
        xp, Pp = self._predicted(dt)
        S = self.H @ Pp @ self.H.T + R
        innov = z - self.H @ xp
        accepted = float(innov @ np.linalg.solve(S, innov)) <= self.gate_chi2
        if accepted:
            K = Pp @ self.H.T @ np.linalg.inv(S)
            self.x, self.P = xp + K @ innov, (np.eye(4) - K @ self.H) @ Pp
            self.misses = 0
        else:
            self.x, self.P = xp, Pp
            self.misses += 1
        sp = np.linalg.norm(self.x[2:])
        if sp > self.vmax:
            self.x[2:] *= self.vmax / sp
        return accepted


# ─────────────────────────────── tracker ──────────────────────────────────────
class Tracker:
    CAL_FRAMES   = 20      # pings in the initial background (median)
    BG_ALPHA     = 0.02    # slow background update afterwards
    BG_GUARD_M   = 0.30    # don't update the background within ± this of the object
    MAP_EVERY    = 50      # re-run wall mapping every N pings
    TOA_SIGMA_M  = 0.003   # assumed 1σ error of one path length
    MIN_GATE_M   = 0.15    # min ± search window around the predicted path
    DETECT_SNR   = 12.0    # residual echo peak / median envelope → "detected"
    MEAS_SNR     = 6.0     # a mic is used in the fix only above this
    R_MAX_INFLATION = 100.0
    WALL_CLEARANCE  = 0.10  # an object (person) can't be closer to a wall
    SYNC_MIN_MATCH  = 0.9   # early window vs background correlation to trust a sync

    def __init__(self, mic_pos, tx_pos, chirp, n_samples, *, c=C, frame_dt=0.1,
                 mount_wall=None, multipath=False, async_map=False,
                 sync=True, max_latency_s=0.05):
        """
        mic_pos (n, 2), tx_pos (2,): metres, any frame.
        n_samples:  samples per ping after alignment (sets the maximum range).
        sync:       measure the playback→recording latency on every ping from
                    the direct speaker→mic sound and align the recording to the
                    moment the sound left the speaker. Then the recording only
                    has to start *before* the chirp is heard — it may start up
                    to max_latency_s earlier, by an unknown, varying amount —
                    and must be n_samples + max_latency_s·fs long. With
                    sync=False, sample 0 must be the instant of transmission.
        mount_wall: wallmap.Wall the array is mounted on (normal pointing into
                    the room), or None if free-standing. It tells front from
                    back and its echo 'twins' are modelled in wall mapping.
        multipath:  EXPERIMENTAL — model object↔wall bounces using the wall map.
        """
        self.mic_pos = np.asarray(mic_pos, float)
        self.tx_pos = np.asarray(tx_pos, float)
        self.n_mics = len(self.mic_pos)
        self.chirp, self.fs, self.c, self.dt = chirp, chirp.fs, c, frame_dt
        self.n_samples = n_samples
        self.centre = self.mic_pos.mean(axis=0)
        if mount_wall is not None:
            mount_wall.known = True
        self.known_walls = [mount_wall] if mount_wall is not None else []
        self.multipath = multipath
        self.pulse = wallmap.PulseModel(chirp.burst, chirp.ref) if multipath else None

        self._ref_spec = {}                      # FFT size → conj(ref spectrum)
        direct_m = float(np.max(np.linalg.norm(self.mic_pos - self.tx_pos, axis=1)))
        self.sync = sync
        self.max_latency = int(round(max_latency_s * self.fs)) if sync else 0
        self.latency = 0.0                       # last measured latency, samples
        self._direct = np.linalg.norm(self.mic_pos - self.tx_pos, axis=1) / c * self.fs
        # early arrivals with known delays: direct sound + (if mounted) its
        # reflection off the mounting wall right behind the array, which
        # overlaps it and would otherwise pull the peak late
        self._early = [self._direct] + [
            np.linalg.norm(self.mic_pos - wallmap.reflect(self.tx_pos, w), axis=1) / c * self.fs
            for w in self.known_walls]
        self._sync_pulse = wallmap.PulseModel(chirp.burst, chirp.ref) if sync else None
        self._bg_early = None                    # bg's direct-sound window (analytic)
        self._direct_level = None                # bg's direct sound, summed envelope
        self._last_good, self.sync_ok = None, True
        self.blank_m = direct_m + 0.3           # direct sound + pulse main lobe
        self.map_blank_m = max(0.9, direct_m + 0.6)
        self.max_range_m = min(n_samples / self.fs * c / 2, 6.0)

        self.bg, self._cal = None, []
        self.kf = KalmanCV(chirp.doppler_coupling_s)
        self.wall_map, self.frame = None, 0
        self.async_map, self._pool, self._job = async_map, None, None

        pts = np.vstack([self.mic_pos, self.tx_pos])
        s = np.linalg.svd(pts - pts.mean(0), compute_uv=False)
        self.collinear = bool(s[-1] < 0.02 * s[0])
        if self.collinear and not self.known_walls:
            warnings.warn("mics and speaker are (nearly) on one line and no mounting "
                          "wall is given: front and back are indistinguishable "
                          "until the wall map closes")

    # ── geometry ──────────────────────────────────────────────────────────────
    def paths(self, pos, mics=None):
        """TX → pos → mic path lengths (m)."""
        m = self.mic_pos if mics is None else mics
        return np.linalg.norm(pos - self.tx_pos) + np.linalg.norm(m - pos, axis=1)

    def _jacobian(self, pos, mics):
        u_tx = (pos - self.tx_pos) / max(np.linalg.norm(pos - self.tx_pos), 1e-6)
        d = pos - mics
        return u_tx + d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)

    def position_covariance(self, pos, used=None):
        """σ² (JᵀJ)⁻¹ of the fix — the filter trusts each axis as much as the
        array geometry allows (large aperture → good angle, etc.)."""
        mics = self.mic_pos if used is None else self.mic_pos[used]
        J = self._jacobian(pos, mics)
        return self.TOA_SIGMA_M**2 * np.linalg.inv(J.T @ J + 1e-9 * np.eye(2))

    @property
    def constraints(self):
        """Half-planes the object must be inside: the mounting wall. The
        measured wall map is not used (a wrong map would clamp a good fix or
        hide the object from acquisition) — except in multipath mode, which
        relies on the map anyway and needs it to rule out the wall-bounce
        ghosts outside the room."""
        return self.known_walls + (self.mapped_walls if self.multipath else [])

    @property
    def mapped_walls(self):
        m = self.wall_map
        return [w for w in m.walls if not w.known] if (m is not None and m.closed) else []

    # ── DSP ───────────────────────────────────────────────────────────────────
    def analytic(self, sig):
        """(n_samples, n_mics) signal → (n_mics, n_samples) complex analytic
        matched-filter output; index k = echo delay of k samples.
        Done in the frequency domain on the zero-padded record (positive
        frequencies only). Not scipy's hilbert() on the truncated output: that
        is circular and wraps the loud direct sound at the start of the record
        onto its end, where it looks like distant echoes."""
        # zero padding leaves room for the negative lags, so they don't wrap
        n = 1 << (sig.shape[0] + 2 * self.chirp.n).bit_length()
        if n not in self._ref_spec:
            self._ref_spec[n] = np.conj(np.fft.rfft(self.chirp.ref.astype(np.float64), n=n))
        S = np.fft.rfft(np.asarray(sig, np.float64).T, n=n, axis=-1) * self._ref_spec[n]
        S[:, 1:] *= 2.0
        S[:, -1] *= 0.5                           # Nyquist bin counted once
        full = np.zeros((S.shape[0], n), complex)
        full[:, :S.shape[1]] = S
        return np.fft.ifft(full, axis=-1)[:, :sig.shape[0]]

    # ── speaker/mic sync ──────────────────────────────────────────────────────
    def _advance(self, sig, shift):
        """sig advanced by `shift` (fractional) samples, cropped/padded to
        n_samples:  out[k] = sig[k + shift]."""
        n = 1 << (sig.shape[0] + self.chirp.n).bit_length()
        X = np.fft.rfft(sig, n=n, axis=0)
        X *= np.exp(2j * np.pi * np.fft.rfftfreq(n) * shift)[:, None]
        return np.fft.irfft(X, n=n, axis=0)[:self.n_samples]

    def _coarse_latency(self, sig):
        """Latency from the direct sound: the first strong arrival that lines
        up across all mics with the known speaker→mic distances, then refined
        by a pulse-shape fit. Returns samples (fractional)."""
        hi = min(sig.shape[0], self.max_latency + int(self._direct.max()) + 4 * self.chirp.n)
        env = np.abs(self.analytic(sig[:hi]))
        n = np.arange(env.shape[1])
        k = np.arange(self.max_latency + 1)
        S = sum(np.interp(k + d, n, e, right=0.0) for e, d in zip(env, self._direct))
        lobe = max(int(2 * self.fs / self.chirp.bw), 2)
        # earliest strong arrival: the direct sound comes first, but a nearby
        # object's echo right after it can be much louder. Once calibrated,
        # the threshold is set from the direct sound's own level (the
        # background's); before that, a low fraction of the peak, kept well
        # above the noise before the chirp.
        thr = (0.3 * self._direct_level if self._direct_level else
               max(0.1 * S.max(), 8.0 * float(np.median(S))))
        first = int(np.argmax(S >= thr))
        k0 = first + int(np.argmax(S[first:first + lobe]))
        return self._fit_latency(self.analytic(sig[:hi]), float(k0), lobe)

    def _fit_latency(self, z, lat0, lobe):
        """Refine the latency by fitting the exact pulse shape at the known
        early-arrival delays (free complex amplitude per mic and arrival, so
        the carrier phase doesn't matter) and maximising the explained energy."""
        hw = self._sync_pulse.half_width // 2
        def explained(lat):
            e = 0.0
            for i in range(self.n_mics):
                taus = [lat + a[i] for a in self._early]
                lo, hi = max(int(min(taus)) - hw, 0), min(int(max(taus)) + hw, z.shape[1])
                if hi - lo < 4:
                    continue
                k = np.arange(lo, hi)
                B = np.stack([self._sync_pulse(k - t) for t in taus], axis=1)
                y = z[i, lo:hi]
                coef = np.linalg.lstsq(B, y, rcond=None)[0]
                e += float(np.real(np.vdot(B @ coef, y)))
            return e
        r = scipy.optimize.minimize_scalar(lambda x: -explained(x), method="bounded",
                                           bounds=(lat0 - lobe / 2, lat0 + lobe / 2),
                                           options=dict(xatol=1e-3))
        return float(r.x)

    def _fine_latency(self, aligned):
        """Residual misalignment vs the background, from the carrier phase of
        the static early echoes (direct sound, mounting wall): exact to a
        small fraction of a sample, so the background subtracts coherently.
        Valid for |error| < half a carrier period. Also returns how well the
        early window matches the background (normalised correlation)."""
        w = self._bg_early.shape[1]
        zc = self.analytic(aligned)[:, :w]
        x = np.sum(zc * np.conj(self._bg_early))
        rho = abs(x) / np.sqrt(np.sum(np.abs(zc)**2) * np.sum(np.abs(self._bg_early)**2) + 1e-30)
        return -np.angle(x) / (2 * np.pi * self.chirp.fc / self.fs), float(rho)

    def align(self, raw):
        """Raw recording → recording starting at the transmission instant
        (n_samples long). Sets self.latency (samples)."""
        if not self.sync:
            out = np.zeros((self.n_samples, raw.shape[1]))
            m = min(self.n_samples, raw.shape[0])
            out[:m] = raw[:m]
            self.latency = 0.0
            return out
        lat = self._coarse_latency(raw)
        aligned = self._advance(raw, lat)
        self.sync_ok = True
        if self._bg_early is None:                # calibrating: coarse only
            self.latency = lat
            return aligned
        # Fine-lock to the background. If the early window doesn't look like
        # the background's (an object right at the array overlaps the direct
        # sound), the fresh estimate may be wrong: also try the last good
        # latency (exact if the latency is constant, e.g. one full-duplex
        # stream) and keep whichever matches the background better.
        cands = [lat] + ([self._last_good] if self._last_good is not None else [])
        best = None
        for i, l0 in enumerate(cands):
            al = aligned if i == 0 else self._advance(raw, l0)
            d, rho = self._fine_latency(al)
            l1 = l0 + d if abs(d) < 0.4 * self.fs / self.chirp.fc else l0
            if best is None or rho > best[1]:
                best = (l1, rho)
            if rho >= self.SYNC_MIN_MATCH:
                break
        lat, rho = best
        self.sync_ok = rho >= self.SYNC_MIN_MATCH
        if self.sync_ok:
            self._last_good = lat
        aligned = self._advance(raw, lat)
        self.latency = lat
        return aligned

    def _set_reference(self):
        """Remember the background's early (static, object-free) window for
        fine alignment."""
        if self.sync:
            # just the direct sound (+ mounting-wall reflection): main lobes
            w = int(max(a.max() for a in self._early) + 2 * self.fs / self.chirp.bw) + 1
            self._bg_early = self.analytic(self.bg)[:, :w]
            env, n = np.abs(self._bg_early), np.arange(w)
            self._direct_level = float(sum(np.interp(d, n, e)
                                           for e, d in zip(env, self._direct)))

    def measure_paths(self, env, predicted_pos, gate_m):
        """Per-mic echo peak within ± gate_m of the predicted path, refined to
        sub-sample. Returns (paths_m, peak_amplitudes)."""
        k_per_m = self.fs / self.c
        g = max(int(gate_m * k_per_m), 4)
        paths, peaks = np.empty(self.n_mics), np.empty(self.n_mics)
        for i, (e, p) in enumerate(zip(env, self.paths(predicted_pos))):
            ctr = int(round(p * k_per_m))
            lo, hi = max(ctr - g, 1), min(ctr + g, len(e) - 2)
            if lo >= hi:
                lo, hi = 1, len(e) - 2
            k = lo + int(np.argmax(e[lo:hi + 1]))
            peaks[i], paths[i] = e[k], _parabolic_peak(e, k) / k_per_m
        return paths, peaks

    def solve(self, paths, used, x0):
        """Robust LS position from the used mics' path lengths. Kept in front
        of the mounting wall (for a line array the mirror image behind it fits
        equally well)."""
        mics, p = self.mic_pos[used], paths[used]
        f = lambda x: self.paths(x, mics) - p

        def ls(s):
            return scipy.optimize.least_squares(
                f, s, loss="soft_l1", f_scale=3 * self.TOA_SIGMA_M,
                max_nfev=100, ftol=1e-10, xtol=1e-10)

        res = ls(x0)
        for w in self.known_walls:
            if w.signed_distance(res.x) < 0.02:
                alt = ls(clamp_inside(wallmap.reflect(res.x, w), [w], 0.05))
                if w.signed_distance(alt.x) >= 0.02 or alt.cost < res.cost:
                    res = alt
        pos = clamp_inside(res.x, self.constraints, 0.02)
        return pos, float(np.sqrt(np.mean(f(pos)**2)))

    def acquire(self, env, step=0.05):
        """Grid search for the point whose echo delays line up across all
        mics (geometric mean of the envelopes). Ghosts only some mics see,
        or weak ones, lose to the real echo."""
        lo, hi = self.centre - self.max_range_m, self.centre + self.max_range_m
        if self.multipath and self.mapped_walls:
            lo, hi = self.wall_map.polygon.min(0), self.wall_map.polygon.max(0)
        X, Y = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step),
                           indexing="ij")
        P = np.stack([X.ravel(), Y.ravel()], axis=1)
        ok = np.ones(len(P), bool)
        for w in self.constraints:
            ok &= P @ w.normal - w.offset > self.WALL_CLEARANCE
        d_tx = np.linalg.norm(P - self.tx_pos, axis=1)
        ok &= d_tx > self.blank_m / 2
        P, d_tx = P[ok], d_tx[ok]
        n = np.arange(env.shape[1])
        floor = 1e-3 * env.max()
        score = np.zeros(len(P))
        for e, mp in zip(env, self.mic_pos):
            k = (d_tx + np.linalg.norm(P - mp, axis=1)) * self.fs / self.c
            score += np.log(np.interp(k, n, e, right=0.0) + floor)
        return P[int(np.argmax(score))]

    # ── background + wall map ─────────────────────────────────────────────────
    def _update_background(self, raw, obj_paths=None):
        upd = np.ones(raw.shape, bool)
        if obj_paths is not None:               # don't absorb the object
            g = int(self.BG_GUARD_M / self.c * self.fs)
            for i, d in enumerate(obj_paths):
                k = int(d / self.c * self.fs)
                upd[max(k - g, 0):k + self.chirp.n + g, i] = False
        self.bg[upd] += self.BG_ALPHA * (raw[upd] - self.bg[upd])

    def _map_args(self, bg):
        return ((self.analytic(bg), self.mic_pos, self.tx_pos, self.fs, self.c),
                dict(tx_pulse=self.chirp.burst, ref=self.chirp.ref,
                     known_walls=self.known_walls, blank_m=self.map_blank_m))

    def map_walls(self, background=None):
        args, kw = self._map_args(self.bg if background is None else background)
        self.wall_map = wallmap.find_walls(*args, **kw)
        return self.wall_map

    def _schedule_map(self):
        """Re-map the walls. With async_map, in a separate process: the wall
        search is ~1 s of Python-level optimisation, which in a thread would
        hold the GIL and stall whatever runs the pings (e.g. an animation)."""
        if not self.async_map:
            self.map_walls()
            return
        if self._job is not None and not self._job.done():
            return
        args, kw = self._map_args(self.bg)
        if self._pool is None:
            self._pool = cf.ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        # A spawned worker re-runs the main script unless it can't find it —
        # and for a script like the simulator that would rebuild everything.
        # The worker only needs wallmap, so hide the script while it starts.
        main = sys.modules["__main__"]
        saved = {k: main.__dict__[k] for k in ("__file__", "__spec__") if k in main.__dict__}
        main.__dict__.pop("__file__", None)
        main.__spec__ = None
        try:
            self._job = self._pool.submit(wallmap.find_walls, *args, **kw)
        finally:
            main.__dict__.update(saved)
        self._job.add_done_callback(self._map_done)

    def _map_done(self, job):
        try:
            self.wall_map = job.result()
        except Exception as e:                    # keep tracking without a map
            warnings.warn(f"wall mapping failed: {e!r}")

    # ── one ping ──────────────────────────────────────────────────────────────
    def process(self, raw):
        """raw: (samples, n_mics) recording of one ping (see `sync`). Returns a
        dict: latency_ms, sync_ok (False: reused the last latency), calibrating, cal_progress | detected, snr (per mic peak/noise),
        range_m (from the array centre), pos_app (single-ping fix, apparent),
        pos_est (single-ping fix, Doppler offset removed), pos, vel (Kalman),
        accepted, cost (fit RMS, m), used (mics in the fix), envs."""
        raw = self.align(np.asarray(raw, np.float64))
        lat = dict(latency_ms=self.latency / self.fs * 1e3, sync_ok=self.sync_ok)
        if self.bg is None:                       # ── calibration
            self._cal.append(raw.copy())
            if len(self._cal) < self.CAL_FRAMES:
                return dict(lat, calibrating=True,
                            cal_progress=len(self._cal) / self.CAL_FRAMES)
            self.bg, self._cal = np.median(np.stack(self._cal), axis=0), []
            self._set_reference()
            self.map_walls()
            return dict(lat, calibrating=True, cal_progress=1.0)

        self.frame += 1
        if self.frame % self.MAP_EVERY == 0:
            self._schedule_map()
            self._set_reference()
        z = self.analytic(raw - self.bg)
        env = np.abs(z)
        b = int(self.blank_m / self.c * self.fs)
        noise = np.median(env[:, b:], axis=1) + 1e-12
        snr = env[:, b:].max(axis=1) / noise
        # Detection has two levels: a new track needs a strong echo on most
        # mics (DETECT_SNR, anywhere in the record); an existing track only
        # needs ≥ 3 mics above MEAS_SNR within its gate — so a weak, distant
        # object stays tracked without false alarms in an empty room.
        need = max(3, (self.n_mics + 1) // 2)
        strong = int(np.sum(snr >= self.DETECT_SNR)) >= need
        lost = self.kf.lost
        out = dict(lat, calibrating=False, detected=False, snr=snr,
                   envs=env.mean(axis=1), accepted=False, cost=float("nan"))

        if not strong and lost:                   # empty room: coast, learn
            self.kf.coast(self.dt)
            if self.sync_ok:                      # never learn a misaligned ping
                self._update_background(raw)
            return dict(out, pos=self.kf.x[:2].copy(), vel=self.kf.x[2:].copy(),
                        range_m=float("nan"), pos_app=None, pos_est=None,
                        used=np.zeros(self.n_mics, bool))

        if lost:                                  # search the whole room
            pred_pos, pred_vel, pred_std = self.acquire(env), np.zeros(2), 0.1
        else:
            pred_pos, pred_vel, pred_std = self.kf.predict(self.dt)
            pred_pos = clamp_inside(pred_pos, self.constraints)
        # gate around where the echoes will *appear* (incl. Doppler offset)
        pred_app = pred_pos + self.chirp.doppler_coupling_s * pred_vel
        gate_m = max(3.0 * pred_std, self.MIN_GATE_M)
        paths, peaks = self.measure_paths(env, pred_app, gate_m)
        used = peaks / noise >= self.MEAS_SNR
        if used.sum() < 3:
            self.kf.coast(self.dt)
            return dict(out, pos=self.kf.x[:2].copy(), vel=self.kf.x[2:].copy(),
                        range_m=float("nan"), pos_app=None, pos_est=None, used=used)

        out["detected"] = True
        pos_app, cost = self.solve(paths, used, pred_app)
        walls = self.mapped_walls if self.multipath else []
        if walls and (self._bounces_overlap(pos_app, walls)
                      or self._bounces_overlap(pred_app, walls)):
            pos_app = self._refine_multipath(z, [pos_app, pred_app], walls)
            cost = float(np.sqrt(np.mean((self.paths(pos_app)[used] - paths[used])**2)))

        # trust the fix less when the per-mic peaks disagree with it (merged
        # echoes near walls) — a few cm of bias would otherwise be read as a
        # large velocity through the Doppler term of the measurement model
        R = self.position_covariance(pos_app, used) * float(
            np.clip((cost / self.TOA_SIGMA_M)**2, 1.0, self.R_MAX_INFLATION))
        if lost:
            self.kf.reset(pos_app)
        accepted = self.kf.update(pos_app, R, self.dt)
        # update the background only while the track is confirmed — otherwise
        # the mask is in the wrong place and the object gets absorbed
        if accepted and not lost and self.sync_ok:
            self._update_background(raw, obj_paths=self.paths(pos_app))
        vel = self.kf.x[2:].copy()
        pos_est = pos_app - self.chirp.doppler_coupling_s * vel
        return dict(out, pos=self.kf.x[:2].copy(), vel=vel, pos_app=pos_app,
                    pos_est=pos_est, cost=cost, accepted=accepted, used=used,
                    range_m=float(np.linalg.norm(self.kf.x[:2] - self.centre)))

    # ── EXPERIMENTAL: object ↔ wall multipath, using the wall map ─────────────
    MP_FIT_HALF_M, MP_MAX_EXTRA_M, MP_BETAS = 0.15, 0.60, (0.0, 0.5, 0.8)

    def _echo_paths(self, p, walls):
        """(n_paths, n_mics): direct TX→p→mic, plus per wall TX→p→wall→mic
        and TX→wall→p→mic."""
        d_tx = np.linalg.norm(p - self.tx_pos)
        rows = [d_tx + np.linalg.norm(self.mic_pos - p, axis=1)]
        for w in walls:
            mic_img = self.mic_pos - 2 * (self.mic_pos @ w.normal - w.offset)[:, None] * w.normal
            rows.append(d_tx + np.linalg.norm(mic_img - p, axis=1))
            rows.append(np.linalg.norm(p - wallmap.reflect(self.tx_pos, w))
                        + np.linalg.norm(self.mic_pos - p, axis=1))
        return np.array(rows)

    def _bounces_overlap(self, p, walls):
        P = self._echo_paths(p, walls)
        return bool(np.any(np.abs(P[1:] - P[0]) < self.MP_MAX_EXTRA_M))

    def _explained_energy(self, z, p, walls):
        """Echo energy explained at p: per mic pulse(direct) + β·Σ pulse(bounce)
        with one complex amplitude per mic and β shared by all mics."""
        P = self._echo_paths(p, walls) / self.c * self.fs
        g = int(self.MP_FIT_HALF_M / self.c * self.fs) + 2
        extra = self.MP_MAX_EXTRA_M / self.c * self.fs
        segs = []
        for i in range(self.n_mics):
            taus = P[:, i]
            near = taus[1:][np.abs(taus[1:] - taus[0]) < extra]
            span = np.concatenate([[taus[0]], near])
            lo = max(int(span.min()) - g, 0)
            hi = min(int(span.max()) + g + 1, z.shape[1])
            if hi - lo < 4:
                return 0.0
            k = np.arange(lo, hi)
            d = self.pulse(k - taus[0])
            bnc = sum((self.pulse(k - t) for t in near), np.zeros_like(d))
            segs.append((z[i, lo:hi], d, bnc))
        best = 0.0
        for beta in self.MP_BETAS:
            e = 0.0
            for y, d, bnc in segs:
                t = d + beta * bnc
                tt = float(np.vdot(t, t).real)
                if tt > 0:
                    e += abs(np.vdot(t, y))**2 / tt
            best = max(best, e)
        return best

    def _refine_multipath(self, z, starts, walls):
        def cost(p):
            if any(w.signed_distance(p) < self.WALL_CLEARANCE for w in walls):
                return 0.0
            return -self._explained_energy(z, p, walls)
        best_p, best_e = None, -1.0
        for p0 in starts:
            p0 = clamp_inside(p0, walls, self.WALL_CLEARANCE + 0.01)
            res = scipy.optimize.minimize(
                cost, p0, method="Nelder-Mead",
                options=dict(xatol=2e-4, fatol=1e-12,
                             initial_simplex=[p0, p0 + [0.02, 0], p0 + [0, 0.02]]))
            if -res.fun > best_e:
                best_p, best_e = res.x, -res.fun
        return best_p
