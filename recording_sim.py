"""
Simulate a recorded Audiodevice session, so the tracker can be tested on
synthetic audio of the same scene.

    python recording_sim.py recordings/walk1                # → recordings/walk1_sim.{flac,npz,png}
    python recording_sim.py recordings/walk1 --static physical
    python recording_sim.py recordings/walk1 --no-drift --no-speckle   # ablations

The scene comes from the recording's LiDAR data, the device and acoustic
parameters from its audio:

* Room: walls (line segments) fitted to the empty-room LiDAR scans give
  first- and second-order specular echoes (image sources, with a Fresnel-zone
  taper at segment ends); every LiDAR point also scatters weakly (diffuse).
  `--static hybrid` (default) adds the measured static echoes the 2D model
  misses (ceiling, floor, things below/above the LiDAR plane) as fitted taps;
  `--static physical` keeps only the model plus the device's internal coupling.
* Device: speaker→mic coupling (the "direct sound", which on this device is
  not airborne-geometric), per-mic gains and noise levels, all fitted to the
  empty-room part of the recording. Speaker: cardioid facing +depth.
* Static echoes drift: each echo's phase wanders slowly (AR(1), ~5 s), with a
  spread proportional to its path length (air movement, temperature), mostly
  shared by all mics. Smooth over delay, so the taps of one echo move together.
* Person: the LiDAR walker track (the scanned body surface). The body is a
  set of scatterers on the front arc of a cylinder (speckle), with Doppler; it
  shadows static echoes whose path it crosses (walk1: the back-wall echo drops
  to ~30% when someone stands in its path). Its echo strength is fitted to the
  recording as A / path^n, with a fade at close range (out of the speaker's
  vertical beam) and slow random fades (body aspect, ~1 s).

The fitted values (person echo, drift, fades) are printed and stored in the
output's metadata; `--set PARAM VALUE` sets a starting value (or the value,
with `--no-calibrate`).

Writes STEM_sim.flac/.npz (same format, same chunk timing, the recording's
LiDAR scans and track as ground truth), then runs the mapper on both
recordings and prints/plots the comparison. The fit uses the same recording
the comparison runs on: agreement of the fitted signal statistics is by
construction, agreement of the tracking results is the actual check.
"""

import argparse
import json
import time

import numpy as np
import scipy.ndimage

import audiodevice_recording as rec_io
import audiodevice_tracking as geometry
import tracker as T

FS = rec_io.RATE
C = T.C
PERIOD = geometry.PERIOD
NBLOCK = 8192                # samples synthesised per ping
MAX_DELAY = 4400             # samples; static echoes beyond this are dropped
FRONT = [0, 2, 4]
LIDAR_DELAY = 0.1            # s, as in audiodevice_recording.evaluate
DEVICE_BOX = ((-0.10, 0.06), (-0.32, 0.32))   # (depth, right) ranges of the device itself
DRIFT_KNOT = 8               # samples between drift-field knots
DRIFT_SMOOTH = 3.0           # knots: the drift is smooth over delay (~0.17 m of path), so
                             # the overlapping taps of one echo drift together


# ───────────────────────────── matched filter ─────────────────────────────────
class MF:
    """Analytic matched-filter output in echo-amplitude units: a chirp of
    amplitude a at delay d gives a peak of magnitude a at index d."""

    def __init__(self, chirp):
        self.chirp = chirp
        self._spec = {}
        n = chirp.n
        x = np.zeros(4 * n)
        x[n:2 * n] = chirp.burst
        z = self._raw(x[:, None])[0]
        self.gain = float(np.abs(z).max())
        self.pulse = z[1:3 * n] / self.gain          # lag -(n-1)..(n-1)+n
        self.pulse_lag0 = n - 1

    def _raw(self, sig):
        N = 1 << (sig.shape[0] + 2 * self.chirp.n).bit_length()
        if N not in self._spec:
            self._spec[N] = np.conj(np.fft.rfft(self.chirp.ref.astype(np.float64), n=N))
        S = np.fft.rfft(np.asarray(sig, np.float64).T, n=N, axis=-1) * self._spec[N]
        S[:, 1:] *= 2.0
        S[:, -1] *= 0.5
        full = np.zeros((S.shape[0], N), complex)
        full[:, :S.shape[1]] = S
        return np.fft.ifft(full, axis=-1)[:, :sig.shape[0]]

    def __call__(self, sig):
        """(samples, mics) → (mics, samples) complex."""
        return self._raw(sig) / self.gain


def clean(resid, mf, lo, hi, thresh, loop_gain=0.5, max_iter=6000):
    """CLEAN deconvolution of a complex MF trace into integer-delay complex taps."""
    r = resid.copy()
    taps = {}
    p, z0, K = mf.pulse, mf.pulse_lag0, len(resid)
    for _ in range(max_iter):
        k = lo + int(np.argmax(np.abs(r[lo:hi])))
        if abs(r[k]) < thresh:
            break
        a = loop_gain * r[k]
        a0, a1 = max(0, k - z0), min(K, k - z0 + len(p))
        r[a0:a1] -= a * p[a0 - (k - z0):a1 - (k - z0)]
        taps[k] = taps.get(k, 0) + a
    return taps, r


# ───────────────────────────── scene from LiDAR ───────────────────────────────
def lidar_xy(points):
    """[angle_deg (clockwise), range_m, intensity] → (depth, right) in metres."""
    p = np.asarray(points, float).reshape(-1, 3)
    a = np.radians(p[:, 0])
    return np.column_stack([p[:, 1] * np.cos(a), p[:, 1] * np.sin(a)])


def track_span(rec):
    tr = rec.track
    ok = ~np.isnan(tr[:, 1])
    return tr[ok, 0].min(), tr[ok, 0].max()


def empty_scans(rec, margin=(0.5, 1.0)):
    t0 = rec.chunk_times[0]
    a, b = track_span(rec)
    return [s for t, s in rec.scans if t - t0 < a - margin[0] or t - t0 > b + margin[1]]


def static_map(rec, bin_deg=0.5, min_frac=0.3):
    """Median range per angle bin over the empty-room scans → (depth, right) points."""
    scans = empty_scans(rec)
    nb = int(round(360 / bin_deg))
    ranges = [[] for _ in range(nb)]
    for s in scans:
        p = np.asarray(s["points"], float).reshape(-1, 3)
        p = p[p[:, 1] > 0.05]
        for b, r in zip((p[:, 0] / bin_deg).astype(int) % nb, p[:, 1]):
            ranges[b].append(r)
    pts = [[b * bin_deg + bin_deg / 2, np.median(r), 0]
           for b, r in enumerate(ranges) if len(r) >= min_frac * len(scans)]
    xy = lidar_xy(pts)
    # the LiDAR sees the device's own bar just behind it: not part of the room
    (d0, d1), (r0, r1) = DEVICE_BOX
    inside = (xy[:, 0] > d0) & (xy[:, 0] < d1) & (xy[:, 1] > r0) & (xy[:, 1] < r1)
    return xy[~inside]


def _dp_split(pts, tol):
    if len(pts) < 3:
        return [pts]
    a, b = pts[0], pts[-1]
    u = (b - a) / (np.linalg.norm(b - a) + 1e-12)
    d = np.abs((pts - a) @ np.array([-u[1], u[0]]))
    i = int(np.argmax(d))
    if d[i] <= tol:
        return [pts]
    return _dp_split(pts[:i + 1], tol) + _dp_split(pts[i:], tol)


def extract_segments(xy, gap=0.12, tol=0.03, min_len=0.12, min_pts=4):
    """Angle-ordered points → wall segments [(p0, p1), ...]."""
    if len(xy) < 2:
        return []
    jumps = np.linalg.norm(np.diff(xy, axis=0), axis=1) > gap
    runs = np.split(xy, np.where(jumps)[0] + 1)
    if len(runs) > 1 and np.linalg.norm(xy[0] - xy[-1]) <= gap:   # wrap at 0°/360°
        runs = [np.vstack([runs[-1], runs[0]])] + runs[1:-1]
    segs = []
    for run in runs:
        for part in _dp_split(run, tol):
            if len(part) < min_pts:
                continue
            m = part.mean(0)
            u = np.linalg.svd(part - m)[2][0]
            s = (part - m) @ u
            p0, p1 = m + s.min() * u, m + s.max() * u
            if np.linalg.norm(p1 - p0) >= min_len:
                segs.append((p0, p1))
    return segs


# ───────────────────────────── static echoes ──────────────────────────────────
def _mirror(p, a, n):
    return p - 2 * ((p - a) @ n) * n


def _seg_frame(seg):
    a, b = seg
    L = np.linalg.norm(b - a)
    u = (b - a) / L
    return a, u, np.array([-u[1], u[0]]), L


def _hit(p_from, p_img, seg, lam):
    """Reflection point on seg of the ray p_from → p_img (image source), with a
    Fresnel-zone edge weight; None if it misses."""
    a, u, n, L = seg
    den = (p_img - p_from) @ n
    if abs(den) < 1e-9:
        return None
    t = ((a - p_from) @ n) / den
    if not 0 < t < 1:
        return None
    P = p_from + t * (p_img - p_from)
    s = (P - a) @ u
    d1, d2 = np.linalg.norm(p_img - P), np.linalg.norm(p_from - P)
    wf = np.sqrt(lam * d1 * d2 / (d1 + d2))
    w = np.clip(0.5 + min(s, L - s) / wf, 0.0, 1.0)
    return (P, w) if w > 0 else None


def specular_echoes(segments, tx, mics, lam, order=2, min_len2=0.4):
    """List of (mic, path_m, weight, n_bounces, legs (k+1, 2))."""
    frames = [_seg_frame(s) for s in segments]
    out = []
    for i, fa in enumerate(frames):
        a, u, n, L = fa
        if np.sign((tx - a) @ n) == 0:
            continue
        img = _mirror(tx, a, n)
        for m, mic in enumerate(mics):
            if np.sign((tx - a) @ n) != np.sign((mic - a) @ n):
                continue
            h = _hit(mic, img, fa, lam)
            if h:
                P, w = h
                out.append((m, np.linalg.norm(img - mic), w, 1, np.array([tx, P, mic])))
        if order < 2 or L < min_len2:
            continue
        for j, fb in enumerate(frames):
            b, ub, nb, Lb = fb
            if j == i or Lb < min_len2:
                continue
            img2 = _mirror(img, b, nb)
            for m, mic in enumerate(mics):
                if np.sign((img - b) @ nb) != np.sign((mic - b) @ nb):
                    continue
                hb = _hit(mic, img2, fb, lam)
                if not hb:
                    continue
                PB, wb = hb
                if np.sign((tx - a) @ n) != np.sign((PB - a) @ n):
                    continue
                ha = _hit(PB, img, fa, lam)
                if not ha:
                    continue
                PA, wa = ha
                out.append((m, np.linalg.norm(img2 - mic), wa * wb, 2,
                            np.array([tx, PA, PB, mic])))
    return out


def tx_gain(direction, beta):
    """Speaker directivity, facing +depth: (1-β) + β·cos θ."""
    d = np.atleast_2d(direction)
    cos = d[:, 0] / (np.linalg.norm(d, axis=1) + 1e-12)
    return np.clip((1 - beta) + beta * cos, 0.0, None)


# ───────────────────────────── the simulator ─────────────────────────────────
DEFAULTS = dict(
    refl=0.7,            # wall pressure reflection coefficient
    tx_beta=0.5,         # speaker directivity (0 omni, 0.5 cardioid)
    drift_per_m=0.055,   # rad of static-echo phase spread per metre of path
    drift_rho=0.982,     # ping-to-ping correlation of the drift
    drift_amp=0.05,      # relative amplitude jitter of static echoes
    body_radius=0.15,    # m, cylinder the scatterers sit on
    body_points=8,
    body_arc_deg=70.0,   # ± arc facing the speaker
    body_offset=0.0,     # m, scattering centre behind the LiDAR-seen surface
    speckle_rho=0.6,     # ping-to-ping correlation of the scatterer weights
    person_a=0.4,        # person echo amplitude at 1 m path (× mic gain) …
    person_n=1.5,        # … falling as path^-n
    near_fade_m=1.1,     # range (m) where the echo is halved by leaving the speaker's
                         # vertical beam: × 1 / (1 + (near_fade_m / range)^6)
    drift_shared=0.8,    # fraction of the drift common to all mics (shared air path)
    fade_std=0.75,       # slow random fade of the person echo: log-amplitude std …
    fade_tau=0.9,        # … and correlation time (s). Stands in for body aspect/pose,
                         # which one 2D cylinder can't reproduce (walk1: the echo stays
                         # weak for seconds, then jumps ×3 at turning points)
    shadow=0.85,         # attenuation of a static echo whose path the body crosses …
    shadow_r=0.35,       # … falling off over this distance (m) from the path (measured
                         # on walk1: back-wall echo ×0.3 within 0.2 m, ×0.5 at 0.2–0.4 m)
)


class Scene:
    def __init__(self, rec, static="hybrid", seed=0, **params):
        self.rec = rec
        self.p = dict(DEFAULTS, **params)
        self.seed = seed
        self.static_mode = static
        low, high = rec.meta.get("band_hz", (18_000, 22_000))
        self.chirp = T.Chirp(fs=FS, f0=low, f1=high)
        self.mf = MF(self.chirp)
        self.lam = C / self.chirp.fc
        self.mics = geometry.MIC_POSITIONS
        self.tx = geometry.TX_POSITION
        self.n_mics = len(self.mics)
        # transmissions: phase taken from the mapper's lock on the real stream
        self.ping_idx = self._ping_indices()
        self.t_ping = rec.sample_time(self.ping_idx)
        self.map_xy = static_map(rec)
        self.segments = extract_segments(self.map_xy)
        self._person_track()
        # in-band bins for the frequency-domain synthesis
        B = np.fft.rfft(self.chirp.burst.astype(np.float64), NBLOCK)
        self.bins = np.where(np.abs(B) > 1e-3 * np.abs(B).max())[0]
        self.B = B[self.bins]
        self.f = self.bins * FS / NBLOCK

    # ── timing and truth ──
    def _ping_indices(self):
        import audiodevice_mapper
        m = audiodevice_mapper.AudiodeviceMapper(*self.rec.meta.get("band_hz", (18_000, 22_000)))
        for _, c in self.rec.chunks():
            for fr in m.push(c):
                if fr.sample_index >= 0:
                    first = fr.sample_index
                    return np.arange(first % PERIOD, len(self.rec.samples), PERIOD)
        raise RuntimeError("the mapper never locked on the recording")

    def _person_track(self, max_gap=0.35):
        tr = self.rec.track
        ok = ~np.isnan(tr[:, 1])
        t, xy = tr[ok, 0], tr[ok, 1:3]
        q = self.t_ping + LIDAR_DELAY
        surf = np.column_stack([np.interp(q, t, xy[:, k], left=np.nan, right=np.nan)
                                for k in (0, 1)])
        # absent if the nearest track sample is far (gaps in the LiDAR track)
        j = np.clip(np.searchsorted(t, q), 1, len(t) - 1)
        near = np.minimum(np.abs(t[j] - q), np.abs(t[j - 1] - q))
        surf[near > max_gap] = np.nan
        self.surface = surf
        self.present = ~np.isnan(surf[:, 0])

    def body_centre(self, surf):
        r = np.linalg.norm(surf, axis=-1, keepdims=True)
        return surf * (r + self.p["body_offset"] + self.p["body_radius"]) / r

    def empty_mask(self):
        a, b = track_span(self.rec)
        return (self.t_ping < a - 0.5) | (self.t_ping > b + 1.0)

    # ── measurement on the real recording ──
    def real_pings(self, idx=None):
        x = self.rec.samples
        idx = self.ping_idx if idx is None else idx
        out = np.zeros((len(idx), PERIOD, self.n_mics))
        for i, s in enumerate(idx):
            seg = x[s:s + PERIOD]
            out[i, :len(seg)] = seg
        return out

    # ── fit ──
    def fit(self, verbose=True):
        """Device and static-room parameters from the empty-room pings."""
        empty = self.empty_mask()
        X = self.real_pings(self.ping_idx[empty])
        Zr = np.stack([self.mf(x) for x in X])
        self.Zb_real = np.median(Zr.real, 0) + 1j * np.median(Zr.imag, 0)
        self.Zr_empty = Zr
        # noise per mic (white): MF residual level at late delays, empty room
        late = slice(3000, 4600)
        res_late = np.median(np.abs(Zr[:, :, late] - self.Zb_real[None, :, late]), axis=(0, 2))
        unit = np.median(np.abs(self.mf(np.random.default_rng(1).standard_normal((PERIOD, 1)))[0, late]))
        self.noise_std = res_late / unit
        # geometric static echoes, unit gains → per-mic gain and diffuse gain
        self._build_geometric()
        Zs = self._static_mf(self.geo, self.geo["amp0"])
        Zd = self._static_mf(self.dif, self.dif["amp0"])
        self.mic_gain = np.ones(self.n_mics)
        for m in range(self.n_mics):
            spec_m = np.abs(Zs[m])
            cand = np.argsort(spec_m[150:MAX_DELAY])[::-1][:400] + 150
            peaks = [k for k in cand if spec_m[k] == spec_m[max(0, k - 20):k + 21].max()][:3]
            if peaks:
                ratio = [np.abs(self.Zb_real[m, k - 15:k + 16]).max() / spec_m[k] for k in peaks]
                self.mic_gain[m] = float(np.median(ratio))
        k_room = int(min(MAX_DELAY, max(self.geo["delay"].max(), self.dif["delay"].max()) * FS + 60))
        quiet = np.zeros(MAX_DELAY, bool)
        quiet[150:k_room] = True
        for m in range(self.n_mics):
            ks = np.where(np.abs(Zs[m, :MAX_DELAY]) * self.mic_gain[m] > 2e-6)[0]
            for k in ks:
                quiet[max(0, k - 20):k + 21] = False
        bgq = np.sqrt(np.maximum(
            np.median(np.abs(self.Zb_real[FRONT, :MAX_DELAY][:, quiet]) ** 2, axis=1)
            - (res_late[FRONT] / np.sqrt(len(Zr))) ** 2, 0))
        dq = np.median(np.abs(Zd[FRONT, :MAX_DELAY][:, quiet] * self.mic_gain[FRONT, None]), axis=1)
        self.diffuse_gain = float(np.median(bgq / (dq + 1e-15)))
        self.geo["amp"] = self.geo["amp0"] * self.mic_gain[self.geo["mic"]]
        self.dif["amp"] = self.dif["amp0"] * self.mic_gain[self.dif["mic"]] * self.diffuse_gain
        # coupling (+ hybrid fill) taps: CLEAN of what the model doesn't explain
        Zmodel = self._static_mf(self.geo, self.geo["amp"]) + self._static_mf(self.dif, self.dif["amp"])
        hi = MAX_DELAY if self.static_mode == "hybrid" else 130
        thresh = 3 * res_late / np.sqrt(len(Zr)) + 3e-7
        taps = {"mic": [], "delay": [], "amp": []}
        for m in range(self.n_mics):
            tp, _ = clean(self.Zb_real[m] - Zmodel[m], self.mf, 0, hi, thresh[m])
            for k, a in tp.items():
                taps["mic"].append(m), taps["delay"].append(k / FS), taps["amp"].append(a)
        self.taps = {k: np.array(v) for k, v in taps.items()}
        self.taps["legs"] = [None] * len(self.taps["mic"])
        self._assemble_static()
        if verbose:
            print(f"scene: {len(self.segments)} wall segments, {len(self.map_xy)} LiDAR points, "
                  f"{len(self.geo['mic'])} specular + {len(self.dif['mic'])} diffuse + "
                  f"{len(self.taps['mic'])} fitted taps ({self.static_mode})")
            print("noise std per mic", np.array2string(self.noise_std, precision=7))
            print("mic gains", np.array2string(self.mic_gain, precision=3),
                  f" diffuse gain {self.diffuse_gain:.3g}")

    def _build_geometric(self):
        p = self.p
        spec = specular_echoes(self.segments, self.tx, self.mics, self.lam)
        g = {"mic": [], "delay": [], "amp0": [], "legs": []}
        for m, path, w, nb, legs in spec:
            if path / C * FS >= MAX_DELAY:
                continue
            amp = w * p["refl"] ** nb / path * tx_gain(legs[1] - legs[0], p["tx_beta"])[0]
            g["mic"].append(m), g["delay"].append(path / C), g["amp0"].append(amp), g["legs"].append(legs)
        self.geo = {k: (np.array(v) if k != "legs" else v) for k, v in g.items()}
        # diffuse: each map point, random fixed complex weight, ∝ √(arc length)
        rng = np.random.default_rng(self.seed + 7)
        q = self.map_xy
        ds = np.linalg.norm(np.gradient(q, axis=0), axis=1).clip(0.005, 0.1)
        w = (rng.standard_normal(len(q)) + 1j * rng.standard_normal(len(q))) / np.sqrt(2)
        d = {"mic": [], "delay": [], "amp0": [], "legs": []}
        r1 = np.linalg.norm(q - self.tx, axis=1).clip(0.1)
        gtx = tx_gain(q - self.tx, p["tx_beta"])
        for m, mic in enumerate(self.mics):
            r2 = np.linalg.norm(q - mic, axis=1).clip(0.1)
            for i in range(len(q)):
                if (r1[i] + r2[i]) / C * FS >= MAX_DELAY:
                    continue
                d["mic"].append(m), d["delay"].append((r1[i] + r2[i]) / C)
                d["amp0"].append(w[i] * np.sqrt(ds[i]) * gtx[i] / (r1[i] * r2[i]))
                d["legs"].append(np.array([self.tx, q[i], mic]))
        self.dif = {k: (np.array(v) if k != "legs" else v) for k, v in d.items()}

    def _static_mf(self, echoes, amps):
        """MF response (mics, PERIOD) of static echoes without drift."""
        z = np.zeros((self.n_mics, PERIOD + 2 * self.chirp.n), complex)
        p, z0 = self.mf.pulse, self.mf.pulse_lag0
        for m, d, a in zip(echoes["mic"], echoes["delay"], amps):
            k = d * FS
            k0 = int(np.floor(k))
            frac = k - k0
            # linear interpolation between integer shifts of the pulse
            for kk, wt in ((k0, 1 - frac), (k0 + 1, frac)):
                a0, a1 = max(0, kk - z0), min(z.shape[1], kk - z0 + len(p))
                if a1 > a0:
                    z[m, a0:a1] += wt * a * p[a0 - (kk - z0):a1 - (kk - z0)] * \
                        np.exp(-2j * np.pi * self.chirp.fc * (k - kk) / FS)
        return z[:, :PERIOD]

    def _assemble_static(self):
        parts = [self.geo, self.dif, self.taps]
        self.s_mic = np.concatenate([np.asarray(e["mic"], int) for e in parts])
        self.s_delay = np.concatenate([np.asarray(e["delay"], float) for e in parts])
        self.s_amp = np.concatenate([np.asarray(e["amp"], complex) for e in parts])
        legs = sum((list(e["legs"]) for e in parts), [])
        A, Bp, owner = [], [], []
        for i, L in enumerate(legs):
            if L is None:
                continue
            for a, b in zip(L[:-1], L[1:]):
                A.append(a), Bp.append(b), owner.append(i)
        self.leg_a, self.leg_b = np.array(A).reshape(-1, 2), np.array(Bp).reshape(-1, 2)
        self.leg_owner = np.array(owner, int)
        # drift-field sampling: each echo reads the field of its mic at its delay
        self.n_knots = int(PERIOD / DRIFT_KNOT) + 2
        kf = np.clip(self.s_delay * FS / DRIFT_KNOT, 0, self.n_knots - 1.001)
        self.knot_i, self.knot_w = np.floor(kf).astype(int), kf - np.floor(kf)
        self.E = []
        for m in range(self.n_mics):
            sel = np.where(self.s_mic == m)[0]
            self.E.append((sel, (self.B[:, None] * np.exp(
                -2j * np.pi * self.f[:, None] * self.s_delay[None, sel])).astype(np.complex64)))

    # ── synthesis ──
    def _drift_innovation(self, rng):
        # index n_mics = the component shared by all mics
        w = scipy.ndimage.gaussian_filter1d(
            rng.standard_normal((2, self.n_mics + 1, self.n_knots)), DRIFT_SMOOTH, axis=-1, mode="wrap")
        return w / w.std(axis=-1, keepdims=True)

    def synthesize(self, drift=True, speckle=True, shadow=True, noise=True, person=True,
                   static=True, fade=True, verbose=True):
        p = self.p
        rng = np.random.default_rng(self.seed)
        n_s, N = len(self.s_mic), len(self.rec.samples)
        out = np.zeros((N + NBLOCK, self.n_mics))
        # drift field per static echo: AR(1) in pings, std ∝ path
        sig_phi = p["drift_per_m"] * self.s_delay * C if drift else np.zeros(n_s)
        sig_amp = p["drift_amp"] if drift else 0.0
        rho = p["drift_rho"]
        field = self._drift_innovation(rng)          # (phase/amp, mics, knots), AR(1) in pings
        mi, ki, kw = self.s_mic, self.knot_i, self.knot_w
        # body scatterers
        K = int(p["body_points"])
        offs = np.radians(rng.uniform(-p["body_arc_deg"], p["body_arc_deg"], K))
        w_b = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2)
        k_chirp = (self.chirp.f1 - self.chirp.f0) / self.chirp.duration
        n_ch = self.chirp.n
        prev_pts = None
        fade_rho = np.exp(-PERIOD / FS / p["fade_tau"])
        fade_u = rng.standard_normal()
        t0 = time.perf_counter()
        for i, s in enumerate(self.ping_idx):
            field = rho * field + np.sqrt(1 - rho ** 2) * self._drift_innovation(rng)
            own = field[:, mi, ki] * (1 - kw) + field[:, mi, ki + 1] * kw
            shared = field[:, -1, ki] * (1 - kw) + field[:, -1, ki + 1] * kw
            fs_ = p["drift_shared"]
            u_phi, u_amp = np.sqrt(fs_) * shared + np.sqrt(1 - fs_) * own
            if speckle:
                w_b = p["speckle_rho"] * w_b + np.sqrt(1 - p["speckle_rho"] ** 2) * \
                    (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2)
            fade_u = fade_rho * fade_u + np.sqrt(1 - fade_rho ** 2) * rng.standard_normal()
            wts = self.s_amp * np.exp(1j * sig_phi * u_phi) * (1 + sig_amp * u_amp)
            here = person and self.present[i]
            if here:
                cen = self.body_centre(self.surface[i])
                if shadow and len(self.leg_a):
                    d = _seg_dist(cen, self.leg_a, self.leg_b)
                    att = 1 - p["shadow"] * np.exp(-(d / p["shadow_r"]) ** 2)
                    f = np.ones(n_s)
                    np.minimum.at(f, self.leg_owner, att)
                    wts = wts * f
            block = np.zeros((NBLOCK, self.n_mics))
            for m, (sel, E) in enumerate(self.E if static else []):
                X = np.zeros(NBLOCK // 2 + 1, complex)
                X[self.bins] = E @ wts[sel].astype(np.complex64)
                block[:, m] = np.fft.irfft(X, NBLOCK)
            if here:
                face = np.arctan2(*(self.tx - cen)[::-1])
                ang = face + offs
                pts = cen + p["body_radius"] * np.column_stack([np.cos(ang), np.sin(ang)])
                vel = np.zeros_like(pts) if prev_pts is None else (pts - prev_pts) / (PERIOD / FS)
                r1 = np.linalg.norm(pts - self.tx, axis=1)
                g = tx_gain(pts - self.tx, p["tx_beta"])
                ww = (w_b if speckle else np.ones(K)) * (np.exp(p["fade_std"] * fade_u) if fade else 1.0)
                for m, mic in enumerate(self.mics):
                    r2 = np.linalg.norm(pts - mic, axis=1)
                    L = r1 + r2
                    rate = (((pts - self.tx) / r1[:, None] + (pts - mic) / r2[:, None]) * vel).sum(1)
                    amp = (p["person_a"] * ww * g * L ** -p["person_n"] / np.sqrt(K) * self.mic_gain[m]
                           * near_fade(np.linalg.norm(cen), p["near_fade_m"]))
                    for j in range(K):
                        k0 = int(L[j] / C * FS)
                        n = np.arange(k0, min(k0 + n_ch + 2, NBLOCK))
                        tt = (n / FS - L[j] / C) * (1 - rate[j] / C)
                        ok = (tt >= 0) & (tt < self.chirp.duration)
                        tt = tt[ok]
                        ph = 2 * np.pi * (self.chirp.f0 * tt + 0.5 * k_chirp * tt ** 2)
                        block[n[ok], m] += np.real(amp[j] * np.exp(1j * ph)) * self.chirp.window(tt)
                prev_pts = pts
            else:
                prev_pts = None
            out[s:s + NBLOCK] += block
        out = out[:N]
        if noise:
            out += rng.standard_normal(out.shape) * self.noise_std[None, :]
        if verbose:
            print(f"synthesised {N / FS:.1f} s, {len(self.ping_idx)} pings "
                  f"in {time.perf_counter() - t0:.1f} s")
        return out.astype(np.float32)

    # ── statistics used for fitting and comparing ──
    def echo_stats(self, samples):
        """Per-ping peak of the residual (vs the empty-room median) near the
        person's predicted path, and the empty residual there; front mics."""
        idx = self.ping_idx
        X = np.zeros((len(idx), PERIOD, self.n_mics))
        for i, s in enumerate(idx):
            seg = samples[s:s + PERIOD]
            X[i, :len(seg)] = seg
        Z = np.stack([self.mf(x) for x in X])
        empty = self.empty_mask()
        Zb = np.median(Z[empty].real, 0) + 1j * np.median(Z[empty].imag, 0)
        R = Z - Zb
        rows = []
        w = int(0.25 / C * FS)
        ref = np.where(empty)[0][::max(1, empty.sum() // 40)]
        for i in np.where(self.present)[0]:
            cen = self.surface[i] * (1 + 0.12 / np.linalg.norm(self.surface[i]))
            for m in FRONT:
                L = np.linalg.norm(cen - self.tx) + np.linalg.norm(cen - self.mics[m])
                k = int(L / C * FS)
                seg = np.abs(R[i, m, k - w:k + w])
                floor = np.median(np.abs(R[ref, m, k - w:k + w]).max(axis=1))
                rows.append((np.linalg.norm(self.surface[i]), L, seg.max(),
                             (np.argmax(seg) - w) / FS * C, m, i, floor))
        rows = np.array(rows)
        # static drift: empty-room residual / static level at strong static bins
        b = np.abs(Zb)
        res = np.median(np.abs(R[empty]), axis=0) / np.sqrt(np.log(2))   # robust rms
        drift = []
        for lo, hi in ((0.8, 3.0), (3.0, 7.0), (7.0, 12.0)):
            k0, k1 = int(lo / C * FS), int(hi / C * FS)
            vals = []
            for m in FRONT:
                fl = np.median(res[m, 3000:4600])
                ks = np.arange(k0, k1)[b[m, k0:k1] > 6 * fl]
                vals += list(np.sqrt(np.maximum(res[m, ks] ** 2 - fl ** 2, 0)) / b[m, ks])
            drift.append(np.median(vals) if vals else np.nan)
        return dict(rows=rows, Zb=Zb, res_rms=res, drift=np.array(drift))


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / ((ab * ab).sum(1) + 1e-12), 0, 1)
    return np.linalg.norm(a + t[:, None] * ab - p, axis=1)


def range_bins(rows, edges=np.arange(1.0, 3.51, 0.5)):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        out.append(np.median(rows[s, 2]) if s.sum() >= 10 else np.nan)
    return np.array(out), (edges[:-1] + edges[1:]) / 2


def slow_fade(rows, scene):
    """Std of the 1 s-averaged log echo SNR (front-mic mean), range trend removed."""
    pi = rows[:, 5].astype(int)
    snr = np.log(rows[:, 2] / rows[:, 6])
    u = np.unique(pi)
    v = np.array([snr[pi == k].mean() for k in u])
    rng_ = np.linalg.norm(scene.surface[u], axis=1)
    d = v - np.polyval(np.polyfit(rng_, v, 2), rng_)
    sm = np.convolve(d, np.ones(10) / 10, "valid")
    return float(np.std(sm))


def near_fade(r, r0):
    return 1.0 / (1.0 + (r0 / np.maximum(r, 1e-3)) ** 6) if r0 > 0 else 1.0


def echo_only(rows):
    """Person echo with the noise/drift floor removed (power subtraction)."""
    return np.sqrt(np.maximum(rows[:, 2] ** 2 - rows[:, 6] ** 2, (0.3 * rows[:, 2]) ** 2))


def calibrate(scene, real_stats, verbose=True):
    """Fit drift_per_m (empty-room residual of static echoes), then person_a
    and person_n (echo strength vs path, noise-free synthesis vs the real
    echo with its floor removed)."""
    for it in range(2):
        sim = scene.echo_stats(scene.synthesize(person=False, verbose=False))
        ok = ~np.isnan(real_stats["drift"]) & ~np.isnan(sim["drift"]) & (sim["drift"] > 0)
        if ok.any():
            scene.p["drift_per_m"] = float(scene.p["drift_per_m"] * np.clip(
                np.median(real_stats["drift"][ok] / sim["drift"][ok]), 0.2, 5.0))
        if verbose:
            print(f"  drift fit {it + 1}: sim/real {np.round(sim['drift'] / real_stats['drift'], 2)}"
                  f" → {scene.p['drift_per_m']:.3f} rad/m")
    real = real_stats["rows"]
    sim = scene.echo_stats(scene.synthesize(static=False, noise=False, drift=False, fade=False,
                                            verbose=False))["rows"]
    edges = np.arange(0.75, 3.51, 0.25)
    lr, lp, rr = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        r = (real[:, 0] >= lo) & (real[:, 0] < hi)
        q = (sim[:, 0] >= lo) & (sim[:, 0] < hi)
        if r.sum() >= 10 and q.sum() >= 10:
            lr.append(np.log(np.median(echo_only(real[r])) / np.median(sim[q, 2])))
            lp.append(np.log(np.median(real[r, 1])))
            rr.append(np.median(real[r, 0]))
    lr, lp, rr = map(np.array, (lr, lp, rr))
    far = rr >= 1.5                     # strength vs path where the near fade doesn't act
    slope, icpt = np.polyfit(lp[far], lr[far], 1)
    resid = lr - (icpt + slope * lp)
    fades = np.arange(0.0, 1.41, 0.05)
    cost = [np.sum((resid - np.log([near_fade(r, f) for r in rr])) ** 2) for f in fades]
    scene.p["near_fade_m"] = float(fades[int(np.argmin(cost))])
    scene.p["person_n"] = float(scene.p["person_n"] - slope)
    scene.p["person_a"] = float(scene.p["person_a"] * np.exp(icpt))
    # where the echo comes from relative to the LiDAR-seen surface (path → range: ½)
    shift = 0.5 * (np.median(real[:, 3]) - np.median(sim[:, 3]))
    scene.p["body_offset"] = float(scene.p["body_offset"] + shift)
    # slow fades: match the spread of the 1 s-averaged echo SNR
    target = slow_fade(real_stats["rows"], scene)
    base = slow_fade(scene.echo_stats(scene.synthesize(fade=False, verbose=False))["rows"], scene)
    for it in range(2):
        got = slow_fade(scene.echo_stats(scene.synthesize(verbose=False))["rows"], scene)
        want = np.sqrt(max(target ** 2 - base ** 2, 1e-4))
        have = np.sqrt(max(got ** 2 - base ** 2, 1e-4))
        scene.p["fade_std"] = float(np.clip(scene.p["fade_std"] * want / have, 0.0, 2.0))
        if verbose:
            print(f"  fade fit {it + 1}: 1 s SNR spread sim {got:.2f} vs real {target:.2f} "
                  f"(no fade {base:.2f}) → fade_std {scene.p['fade_std']:.2f}")
    if verbose:
        print(f"  person echo fit: A {scene.p['person_a']:.3g} at 1 m path, "
              f"falling as path^-{scene.p['person_n']:.2f}, halved at {scene.p['near_fade_m']:.2f} m "
              f"range; scattering centre "
              f"{scene.p['body_offset'] + scene.p['body_radius']:+.2f} m behind the LiDAR surface")


# ───────────────────────────── output and comparison ──────────────────────────
def save_like(rec, samples, stem, meta):
    starts = np.concatenate([[0], np.cumsum(rec.chunk_lengths)])
    chunks = [(t, samples[a:b]) for t, a, b in zip(rec.chunk_times, starts[:-1], starts[1:])]
    rec_io.save(stem, chunks, rec.scans, track=rec.track, meta=meta)


def summary(name, m):
    re, le = m.get("range_err", np.array([])), m.get("lateral_err", np.array([]))
    q = lambda a, p: np.percentile(a, p) if len(a) else np.nan
    return (f"{name:10s} tracked {100 * m.get('tracked', np.nan):4.0f}%  "
            f"range err med {q(re, 50):.2f} p90 {q(re, 90):.2f} m  "
            f"lateral err med {q(le, 50):.2f} p90 {q(le, 90):.2f} m  "
            f"false {100 * m.get('false_tracks', np.nan):4.1f}%  "
            f"facing wall {m['facing_wall_m'] if m['facing_wall_m'] is None else round(m['facing_wall_m'], 2)} m")


def plot(scene, real_stats, sim_stats, m_real, m_sim, fn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    a = ax[0, 0]
    a.scatter(scene.map_xy[:, 1], scene.map_xy[:, 0], s=2, c="#999", label="LiDAR (empty)")
    for p0, p1 in scene.segments:
        a.plot([p0[1], p1[1]], [p0[0], p1[0]], "k-", lw=1.5)
    s = scene.surface[scene.present]
    a.plot(s[:, 1], s[:, 0], "-", c="#2a2", lw=1, label="LiDAR walker")
    for m, col, lab in ((m_real, "#e70", "tracked: real"), (m_sim, "#08c", "tracked: sim")):
        r = m["rows"]
        a.plot(r[:, 2], r[:, 1], ".", c=col, ms=3, label=lab)
    a.plot(scene.mics[:, 1], scene.mics[:, 0], "r^", ms=5)
    a.plot(scene.tx[1], scene.tx[0], "bs", ms=5)
    a.set_aspect("equal"); a.grid(alpha=0.3); a.legend(fontsize=8, loc="lower right")
    a.set_xlabel("right (m)"); a.set_ylabel("depth (m)"); a.set_title("scene and tracks")
    path = np.arange(PERIOD) / FS * C
    for j, m in enumerate((2, 4)):
        a = ax[0, 1 + j]
        a.semilogy(path, np.abs(real_stats["Zb"][m]), c="#e70", lw=0.8, label="static: real")
        a.semilogy(path, np.abs(sim_stats["Zb"][m]), c="#08c", lw=0.8, alpha=0.8, label="static: sim")
        a.semilogy(path, real_stats["res_rms"][m], c="#a40", lw=0.8, ls=":", label="empty residual: real")
        a.semilogy(path, sim_stats["res_rms"][m], c="#046", lw=0.8, ls=":", label="empty residual: sim")
        a.set_xlim(0, 14); a.set_ylim(1e-6, 1e-3); a.grid(alpha=0.3)
        a.set_xlabel("echo path (m)"); a.set_title(f"mic {m}: matched-filter envelope")
        a.legend(fontsize=8)
    a = ax[1, 0]
    for st, col, lab in ((real_stats, "#e70", "real"), (sim_stats, "#08c", "sim")):
        amp, rc = range_bins(st["rows"], np.arange(0.5, 3.51, 0.25))
        a.semilogy(rc, amp, "o-", c=col, label=lab)
    a.set_xlabel("person range (m)"); a.set_title("person echo peak (front mics, median)")
    a.grid(alpha=0.3); a.legend()
    a = ax[1, 1]
    for m, col, lab in ((m_real, "#e70", "real"), (m_sim, "#08c", "sim")):
        for k, ls in (("lateral_err", "-"), ("range_err", "--")):
            e = np.sort(m.get(k, []))
            if len(e):
                a.plot(e, np.arange(1, len(e) + 1) / len(e), ls, c=col, label=f"{lab}: {k.split('_')[0]}")
    a.set_xlim(0, 1.5); a.set_xlabel("error (m)"); a.set_title("tracking error CDF")
    a.grid(alpha=0.3); a.legend(fontsize=8)
    a = ax[1, 2]
    tt = scene.t_ping
    a.plot(tt, np.linalg.norm(scene.surface, axis=1), c="#2a2", label="LiDAR")
    for m, col, lab in ((m_real, "#e70", "real"), (m_sim, "#08c", "sim")):
        r = m["rows"]
        a.plot(r[:, 0], np.hypot(r[:, 1], r[:, 2]), ".", c=col, ms=3, label=lab)
    a.set_xlabel("time (s)"); a.set_ylabel("range (m)"); a.set_title("range over time")
    a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fn, dpi=80)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="recording to simulate, e.g. recordings/walk1")
    ap.add_argument("--out", help="output stem (default STEM_sim)")
    ap.add_argument("--static", choices=("hybrid", "physical"), default="hybrid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip fitting person echo strength and drift (use defaults)")
    for k in ("drift", "speckle", "shadow", "noise", "fade"):
        ap.add_argument(f"--no-{k}", action="store_true", help=f"ablation: disable {k}")
    ap.add_argument("--set", nargs=2, action="append", default=[], metavar=("PARAM", "VALUE"),
                    help=f"override a model parameter ({', '.join(DEFAULTS)})")
    a = ap.parse_args()
    out = a.out or f"{a.stem}_sim"

    rec = rec_io.load(a.stem)
    scene = Scene(rec, static=a.static, seed=a.seed, **{k: float(v) for k, v in a.set})
    scene.fit()
    real = rec.samples.astype(np.float64)
    real_stats = scene.echo_stats(real)
    if not a.no_calibrate:
        calibrate(scene, real_stats)
    flags = dict(drift=not a.no_drift, speckle=not a.no_speckle,
                 shadow=not a.no_shadow, noise=not a.no_noise, fade=not a.no_fade)
    sim = scene.synthesize(**flags)
    sim_stats = scene.echo_stats(sim)
    meta = dict(rec.meta, simulated_from=a.stem, static=a.static, seed=a.seed,
                params=scene.p, flags=flags,
                note=f"Simulated from {a.stem} by recording_sim.py. " + rec.meta.get("note", ""))
    save_like(rec, sim, out, meta)
    print(f"wrote {out}.flac / {out}.npz")

    print("running the mapper on both …")
    m_real = rec_io.evaluate(rec)
    m_sim = rec_io.evaluate(rec_io.load(out))
    print(summary("real", m_real))
    print(summary("simulated", m_sim))
    amp_r, rc = range_bins(real_stats["rows"])
    amp_s, _ = range_bins(sim_stats["rows"])
    print("person echo peak by range  " + "  ".join(f"{r:.2f} m" for r in rc))
    print("  real                     " + "  ".join(f"{v:.1e}" for v in amp_r))
    print("  sim                      " + "  ".join(f"{v:.1e}" for v in amp_s))
    print("static drift (res/echo) by path 0.8–3 / 3–7 / 7–12 m:  real "
          f"{np.round(real_stats['drift'], 2)}  sim {np.round(sim_stats['drift'], 2)}")
    plot(scene, real_stats, sim_stats, m_real, m_sim, f"{out}.png")
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
