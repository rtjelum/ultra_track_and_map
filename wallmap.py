"""
Wall mapping from static echoes — image-source method + CLEAN.

A flat wall reflects the transmitted chirp as if it came from a mirror image
of the transmitter behind the wall. So each wall shows up in the static
(background) echo profile as a "virtual speaker" whose distance to each mic is
the echo path length TX → wall → mic. A wall is described by the angle φ of
its inward normal and its distance d from TX; its image source is
TX − 2·d·n(φ).

Because the array is small (≈ 22 cm), range is measured far better than
angle, and echoes from different walls can overlap in time. So walls are
extracted one at a time, strongest first (CLEAN, as in radar/radio astronomy):

  1. Detect: grid search over (φ, d) for the geometric mean across mics of
     the matched-filter envelope at the predicted echo delays (every mic must
     see the echo where the wall predicts it).
  2. Fit: maximise the echo energy explained by the wall — per mic, a
     least-squares fit of the analytic pulse shape at the predicted delays of
     the wall echo and its 'twin' via each known wall (TX → wall → mounting
     wall → mic, ≈ 2·5 cm later or merged). A free complex amplitude per mic
     and echo absorbs carrier phase, so there is no cycle ambiguity.
  3. Subtract the fitted echoes from the signals, and repeat until nothing
     significant is left. A weak echo can no longer borrow evidence from a
     stronger, overlapping one.
  4. Classify: an image that coincides with the image of an image across two
     accepted walls is a higher-order (corner / flutter) echo; a wall lying
     outside the outline of stronger walls is spurious (it couldn't be heard).
  5. Room outline = intersection of the half-planes "inside each wall".

Independent of the simulator: mic geometry, sample rate, speed of sound and
the transmitted pulse are passed in, so it can run on recorded hardware data.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.optimize


@dataclass
class Wall:
    normal: np.ndarray          # unit normal pointing into the room
    offset: float               # wall line:  normal · x = offset
    image: np.ndarray = None    # image source of TX that produced it
    strength: float = 0.0       # RMS fitted echo amplitude (range-compensated)
    known: bool = False         # supplied, not measured (e.g. mounting wall)
    raw_angle_deg: float = None # fitted angle before rectilinear snapping

    def signed_distance(self, p):
        """> 0 inside the room."""
        return float(self.normal @ p - self.offset)

    @property
    def angle_deg(self):
        """Direction of the inward normal, degrees (0 = +x)."""
        return float(np.degrees(np.arctan2(self.normal[1], self.normal[0])))


@dataclass
class WallMap:
    walls: list                                   # accepted walls (incl. known)
    rejected: list = field(default_factory=list)  # (image, reason)
    polygon: np.ndarray = None                    # room outline vertices (k, 2)
    closed: bool = False                          # outline bounded by walls only


def reflect(p, wall):
    """Mirror point p across wall."""
    return p - 2.0 * wall.signed_distance(p) * wall.normal


def wall_from_image(tx, q):
    """Wall = perpendicular bisector of TX and its image q."""
    d = tx - q
    n = d / np.linalg.norm(d)
    return Wall(normal=n, offset=float(n @ (tx + q) / 2.0), image=q.copy())


# ───────────────────────────── pulse model ────────────────────────────────────
class PulseModel:
    """Analytic (complex) matched-filter response of one echo, tabulated
    finely so it can be evaluated at any fractional delay.
    mf[k] = Σ a · R(k − τ)   with   R = xcorr(tx_pulse, ref)."""

    def __init__(self, tx_pulse, ref, oversample=32):
        n = 1 << (len(tx_pulse) + len(ref)).bit_length()
        spec = np.fft.rfft(tx_pulse, n) * np.conj(np.fft.rfft(ref, n))
        full = np.zeros(n * oversample, complex)
        full[:len(spec)] = 2 * spec                  # analytic: positive freqs
        full[0] = spec[0]
        self.table = np.fft.ifft(full) * oversample  # circular, lag 0 at index 0
        self.n, self.os = n, oversample
        self.half_width = (len(tx_pulse) + len(ref)) // 2 + 2   # samples

    def __call__(self, lag):
        """R at (fractional) lag in samples, zero outside the pulse."""
        lag = np.asarray(lag, float)
        x = (lag * self.os) % (self.n * self.os)
        i = np.floor(x).astype(int)
        f = x - i
        t = self.table
        v = t[i] * (1 - f) + t[(i + 1) % len(t)] * f
        return np.where(np.abs(lag) < self.half_width, v, 0.0)


# ─────────────────────────────── geometry ─────────────────────────────────────
def _images(tx, phi, dist, known_walls):
    """Image sources a wall (normal angle phi, distance dist from TX) produces:
    first-order image + its twin via each known wall."""
    n = np.array([np.cos(phi), np.sin(phi)])
    q = tx - 2.0 * dist * n
    return [q] + [reflect(q, k) for k in known_walls]


def _in_front(tx, phi, dist, known_walls, margin=0.02):
    """The wall's closest point to TX must lie inside the room w.r.t. every
    known wall. Needed because a wall echo + its twin via the mounting wall
    look identical to a mirror 'wall' behind the mounting wall."""
    phi, dist = np.broadcast_arrays(np.asarray(phi, float), np.asarray(dist, float))
    fx = tx[0] - dist * np.cos(phi)
    fy = tx[1] - dist * np.sin(phi)
    ok = np.ones(phi.shape, bool)
    for k in known_walls:
        ok &= (k.normal[0] * fx + k.normal[1] * fy - k.offset) > margin
    return ok


def _score_grid(env, mic_pos, fs, c, tx, PHI, D, known_walls, floor):
    """Geometric mean over mics of (envelope at wall echo + twins)."""
    nx, ny = np.cos(PHI), np.sin(PHI)
    qx, qy = tx[0] - 2 * D * nx, tx[1] - 2 * D * ny
    imgs = [(qx, qy)]
    for k in known_walls:
        sd = k.normal[0] * qx + k.normal[1] * qy - k.offset
        imgs.append((qx - 2 * sd * k.normal[0], qy - 2 * sd * k.normal[1]))
    idx = np.arange(env.shape[1])
    log_sum = np.zeros(PHI.shape)
    for e, (mx, my) in zip(env, mic_pos):
        s = sum(np.interp(np.hypot(ix - mx, iy - my) / c * fs, idx, e, right=0.0)
                for ix, iy in imgs)
        log_sum += np.log(s + floor)
    return np.exp(log_sum / len(mic_pos)) * _in_front(tx, PHI, D, known_walls)


TWIN_RATIOS = (0.0, 0.35, 0.7, 1.0)


def _fit_echoes(z, mic_pos, fs, c, pulse, imgs, margin, twin_ratios=TWIN_RATIOS):
    """Fit the wall's echo at every mic: shape = pulse at the first-order
    delay + β · pulse at each twin delay, one complex amplitude per mic.
    β (twin/primary ratio, physically ≈ the known wall's reflection
    coefficient) is shared by all mics and picked from twin_ratios — free
    per-mic twin amplitudes would let a wrong angle 'fit' a merged echo.
    Returns (explained_energy, model (n_mics, n) complex, amps)."""
    segs = []
    for mp in mic_pos:
        taus = [np.linalg.norm(q - mp) / c * fs for q in imgs]
        lo = max(int(min(taus)) - margin, 0)
        hi = min(int(max(taus)) + margin + 1, z.shape[1])
        if hi - lo < 4:
            return 0.0, np.zeros_like(z), None
        k = np.arange(lo, hi)
        segs.append((lo, hi, pulse(k - taus[0]),
                     sum(pulse(k - t) for t in taus[1:]) if len(taus) > 1 else 0))
    best = None
    for beta in (twin_ratios if len(imgs) > 1 else (0.0,)):
        explained, parts = 0.0, []
        for i, (lo, hi, p0, pt) in enumerate(segs):
            t = p0 + beta * pt
            y = z[i, lo:hi]
            tt = float(np.vdot(t, t).real)
            a = np.vdot(t, y) / tt if tt > 0 else 0.0
            explained += abs(np.vdot(t, y))**2 / tt if tt > 0 else 0.0
            parts.append((lo, hi, a * t, a))
        if best is None or explained > best[0]:
            best = (explained, parts)
    explained, parts = best
    model = np.zeros_like(z)
    for i, (lo, hi, fit, _) in enumerate(parts):
        model[i, lo:hi] = fit
    return explained, model, [np.atleast_1d(a) for *_, a in parts]


# ──────────────────────────────── mapping ─────────────────────────────────────
def find_walls(z, mic_pos, tx_pos, fs, c, tx_pulse, ref, known_walls=(),
               blank_m=0.9, d_min=0.45, d_max=None,
               dd=0.01, dphi_deg=1.0, max_walls=10,
               rel_threshold=0.25, noise_factor=8.0,
               higher_order_tol_m=0.15, rectilinear=True, snap_deg=8.0,
               fit_halfwidth_m=0.15, debug=False):
    """
    Detect walls from the static echo profile.

    z:           (n_mics, n_samples) complex analytic matched-filter output of
                 the background (sample index = echo delay).
    tx_pulse:    transmitted pulse samples; ref: matched-filter reference.
    known_walls: walls known in advance (the mounting wall: its own echo
                 overlaps the direct speaker→mic sound, but it creates a twin
                 of every other wall echo, which is modelled).
    blank_m:     ignore echo paths shorter than this (direct sound region).
    rel_threshold: stop when the best remaining detection, compensated for
                 spherical spreading (× path length), drops below this
                 fraction of the first (strongest) one.
    rectilinear: snap angles within snap_deg of a multiple of 90° to the
                 first known wall (rectangular-room prior) and re-fit d.
    fit_halfwidth_m: ± window (path metres) around each predicted echo used
                 for the amplitude fit and the subtraction.
    """
    known_walls = list(known_walls)
    z = np.array(z, complex)
    z[:, :int(blank_m / c * fs)] = 0.0
    pulse = PulseModel(tx_pulse, ref)
    # fit only over the pulse main lobe: a wider window lets the fit use the
    # pulse sidelobes to soak up neighbouring echoes from other walls
    margin = int(fit_halfwidth_m / c * fs) + 2
    if d_max is None:
        d_max = (z.shape[1] / fs * c) / 2 - 0.05
    centre = mic_pos.mean(axis=0)
    ref_ang = (np.arctan2(known_walls[0].normal[1], known_walls[0].normal[0])
               if known_walls else 0.0)

    d   = np.arange(d_min, d_max, dd)
    phi = np.radians(np.arange(-180.0, 180.0, dphi_deg))
    D, PHI = np.meshgrid(d, phi, indexing="ij")
    noise = float(np.median(np.abs(z[:, int(blank_m / c * fs):])))
    floor = 1e-3 * float(np.abs(z).max() + 1e-12)

    def explained(ph, dist, zr):
        if not _in_front(tx_pos, ph, dist, known_walls) or dist < d_min * 0.5:
            return 0.0
        return _fit_echoes(zr, mic_pos, fs, c, pulse,
                           _images(tx_pos, ph, dist, known_walls), margin)[0]

    found, first_level = [], None
    for _ in range(max_walls):
        env = np.abs(z)
        score = _score_grid(env, mic_pos, fs, c, tx_pos, PHI, D, known_walls, floor)
        level = score * (2 * D)                      # undo 1/path spreading
        j = np.unravel_index(np.argmax(level), level.shape)
        if first_level is None:
            first_level = level[j]
        if level[j] < rel_threshold * first_level or score[j] < noise_factor * noise:
            break

        # refine (φ, d) by maximising the explained echo energy
        x0 = np.array([PHI[j], D[j]])
        res = scipy.optimize.minimize(
            lambda x: -explained(x[0], x[1], z), x0, method="Nelder-Mead",
            options=dict(xatol=1e-5, fatol=1e-9 * np.vdot(z, z).real,
                         initial_simplex=[x0, x0 + [np.radians(2.0), 0],
                                          x0 + [0, 0.01]]))
        ph, dist = res.x
        raw_deg = float(np.degrees(ph))
        if debug:
            print(f"  [clean] grid φ={np.degrees(x0[0]):7.1f}° d={x0[1]:.3f} "
                  f"level={level[j]:.2f} score={score[j]:.3f} → NM φ={raw_deg:7.2f}° "
                  f"d={dist:.3f} explained {-res.fun:.3g} (start {explained(*x0, z):.3g})")
        if rectilinear:
            rel = np.degrees(ph - ref_ang)
            snapped = 90.0 * np.round(rel / 90.0)
            if abs((rel - snapped + 180) % 360 - 180) <= snap_deg:
                ph = ref_ang + np.radians(snapped)
                r1 = scipy.optimize.minimize_scalar(
                    lambda x: -explained(ph, x, z),
                    bounds=(dist - 0.05, dist + 0.05), method="bounded",
                    options=dict(xatol=1e-5))
                dist = float(r1.x)

        imgs = _images(tx_pos, ph, dist, known_walls)
        e, model, amps = _fit_echoes(z, mic_pos, fs, c, pulse, imgs, margin)
        if amps is None:
            break
        z = z - model                                # CLEAN: explain away
        w = wall_from_image(tx_pos, imgs[0])
        w.raw_angle_deg = raw_deg
        w.strength = float(np.sqrt(np.mean([abs(a[0])**2 for a in amps]))) * 2 * dist
        found.append(w)

    # classify, in order of path length (first-order before higher-order)
    found.sort(key=lambda w: np.linalg.norm(w.image - tx_pos))
    accepted, rejected = list(known_walls), []
    for w in found:
        ho = [reflect(reflect(tx_pos, a), b)
              for a in accepted for b in accepted if a is not b]
        if any(np.linalg.norm(w.image - h) < higher_order_tol_m for h in ho):
            rejected.append((w.image, "higher-order echo"))
            continue
        accepted.append(w)

    # a wall lying entirely outside the outline formed by the other walls
    # cannot be heard (it is behind another wall) — it is an echo the fit
    # explained with a wrong angle. Remove such walls, weakest first.
    final = list(accepted)
    for w in sorted((w for w in accepted if not w.known), key=lambda w: w.strength):
        others = [o for o in final if o is not w]
        poly, closed = room_polygon(others, inside=centre)
        if closed and all(w.signed_distance(v) >= -0.02 for v in poly):
            rejected.append((w.image, "outside room"))
            final = others

    poly, closed = room_polygon(final, inside=centre)
    return WallMap(walls=final, rejected=rejected, polygon=poly, closed=closed)


# ─────────────────────────────── room shape ───────────────────────────────────
def room_polygon(walls, inside, box=50.0):
    """Intersect the half-planes {x : n·x ≥ offset} of all walls, starting
    from a big square around `inside`. Returns (vertices, closed) where
    closed is False if the outline still touches the bounding square
    (some side has no detected wall)."""
    poly = [inside + np.array(v) * box for v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for w in walls:
        out = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            da, db = w.signed_distance(a), w.signed_distance(b)
            if da >= 0:
                out.append(a)
            if (da >= 0) != (db >= 0):
                out.append(a + (b - a) * da / (da - db))
        poly = out
        if not poly:
            return np.zeros((0, 2)), False
    poly = np.array(poly)
    closed = bool(np.all(np.abs(poly - inside).max(axis=1) < box * 0.99))
    return poly, closed


def room_size(polygon, ref_wall):
    """Extent of the outline in the frame of ref_wall (normally the mounting
    wall): (depth away from that wall, width along it, area)."""
    if len(polygon) < 3:
        return float("nan"), float("nan"), float("nan")
    u = ref_wall.normal
    v = np.array([-u[1], u[0]])
    depth = float(np.max(polygon @ u - ref_wall.offset))
    along = polygon @ v
    x, y = polygon[:, 0], polygon[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return depth, float(along.max() - along.min()), area
