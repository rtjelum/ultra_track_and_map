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
  2. Fit: for the strongest grid peak in each 30° sector of wall angle,
     maximise the echo energy explained by the wall; keep the best — per mic, a
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


def _fit_echoes(z, mic_pos, fs, c, pulse, imgs, margin, twin_ratios=TWIN_RATIOS,
                want_model=True):
    """Fit the wall's echo at every mic: shape = pulse at the first-order
    delay + β · pulse at each twin delay, one complex amplitude per mic.
    β (twin/primary ratio, physically ≈ the known wall's reflection
    coefficient) is shared by all mics and picked from twin_ratios — free
    per-mic twin amplitudes would let a wrong angle 'fit' a merged echo.
    Vectorised over mics (it is the inner loop of the wall search).
    Returns (explained_energy, model (n_mics, n) complex or None, amps)."""
    imgs = np.asarray(imgs, float)                              # (n_img, 2)
    taus = np.linalg.norm(imgs[:, None, :] - mic_pos[None], axis=2) / c * fs   # (n_img, n_mics)
    lo = np.maximum(taus.min(0).astype(int) - margin, 0)
    hi = np.minimum(taus.max(0).astype(int) + margin + 1, z.shape[1])
    if np.any(hi - lo < 4):
        return 0.0, (np.zeros_like(z) if want_model else None), None
    K = lo[:, None] + np.arange(int((hi - lo).max()))[None]     # (n_mics, L)
    valid = K < hi[:, None]
    Kc = np.minimum(K, z.shape[1] - 1)
    P0 = pulse(K - taus[0][:, None]) * valid
    PT = sum((pulse(K - t[:, None]) for t in taus[1:]), np.zeros_like(P0)) * valid
    Y = np.take_along_axis(z, Kc, axis=1) * valid
    best = None
    for beta in (twin_ratios if len(imgs) > 1 else (0.0,)):
        T = P0 + beta * PT
        tt = np.sum(np.abs(T)**2, axis=1)
        ty = np.sum(np.conj(T) * Y, axis=1)
        ok = tt > 0
        e = float(np.sum(np.abs(ty[ok])**2 / tt[ok]))
        if best is None or e > best[0]:
            best = (e, T, np.where(ok, ty / np.where(ok, tt, 1.0), 0.0))
    explained, T, a = best
    model = None
    if want_model:
        model = np.zeros_like(z)
        for i in range(len(mic_pos)):
            v = valid[i]
            model[i, K[i, v]] = a[i] * T[i, v]
    return explained, model, [np.atleast_1d(x) for x in a]


# ──────────────────────────────── mapping ─────────────────────────────────────
def find_walls(z, mic_pos, tx_pos, fs, c, tx_pulse, ref, known_walls=(),
               blank_m=0.9, d_min=0.45, d_max=None,
               dd=0.01, dphi_deg=1.0, max_walls=10,
               rel_threshold=0.25, noise_factor=8.0,
               higher_order_tol_m=0.15, rectilinear=True, snap_deg=30.0, snap_keep=0.8,
               fit_halfwidth_m=0.15, n_candidates=8, cand_rel=0.6, debug=False):
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
    rectilinear: rectangular-room prior — a wall within snap_deg of a right
                 angle to the first known wall (or, with none, to the first
                 detected wall) is snapped to it if the snapped wall still
                 explains ≥ snap_keep of the echo energy.
    fit_halfwidth_m: ± window (path metres) around each predicted echo used
                 for the amplitude fit and the subtraction.
    n_candidates, cand_rel: per step, up to n_candidates grid peaks (one
                 per 30° of wall angle, ≥ cand_rel × the strongest) are
                 refined; the one explaining the most echo energy is the
                 next wall.
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
    # rectangular-room prior: right angles relative to the first known wall,
    # or (free-standing array) to the strongest detected wall
    ref_ang = (np.arctan2(known_walls[0].normal[1], known_walls[0].normal[0])
               if known_walls else None)

    d   = np.arange(d_min, d_max, dd)
    phi = np.radians(np.arange(-180.0, 180.0, dphi_deg))
    D, PHI = np.meshgrid(d, phi, indexing="ij")
    noise = float(np.median(np.abs(z[:, int(blank_m / c * fs):])))
    floor = 1e-3 * float(np.abs(z).max() + 1e-12)

    def explained(ph, dist, zr):
        if not _in_front(tx_pos, ph, dist, known_walls) or dist < d_min * 0.5:
            return 0.0
        return _fit_echoes(zr, mic_pos, fs, c, pulse,
                           _images(tx_pos, ph, dist, known_walls), margin,
                           want_model=False)[0]

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

        # The grid score only looks at envelopes, and with a small array the
        # echo delays barely depend on the wall angle — the same echo can be
        # 'explained' by a wrong wall (e.g. the wall ahead as a side wall
        # whose twin lines up). So take the best grid peak in each 30° sector
        # of wall angle, refine the strong ones by maximising the explained
        # echo energy (exact pulse shape, all mics coherently) and keep the
        # best explanation.
        sector = (np.degrees(phi) // 30).astype(int)
        cands = []
        for sct in np.unique(sector):
            sub = np.where(sector == sct, level, -1.0)
            k = np.unravel_index(np.argmax(sub), sub.shape)
            if sub[k] >= cand_rel * level[j] and score[k] >= noise_factor * noise:
                cands.append(k)
        cands = sorted(cands, key=lambda k: -level[k])[:n_candidates]
        fatol = 1e-9 * np.vdot(z, z).real
        best = None                                  # (energy, φ, d, raw φ)
        for k in cands:
            x0 = np.array([PHI[k], D[k]])
            res = scipy.optimize.minimize(
                lambda x: -explained(x[0], x[1], z), x0, method="Nelder-Mead",
                options=dict(xatol=1e-5, fatol=fatol,
                             initial_simplex=[x0, x0 + [np.radians(2.0), 0],
                                              x0 + [0, 0.01]]))
            ph, dist, e = res.x[0], res.x[1], -res.fun
            cand = (e, ph, dist, ph)
            # rectangular-room prior: take the nearest right angle if it
            # explains nearly as much — how much a wrong angle costs depends
            # on the array, so a small array (flat in angle) snaps readily
            # while a large one can still report a slanted wall
            if rectilinear and ref_ang is not None:
                rel = np.degrees(ph - ref_ang)
                snapped = 90.0 * np.round(rel / 90.0)
                if abs((rel - snapped + 180) % 360 - 180) <= snap_deg:
                    phs = ref_ang + np.radians(snapped)
                    r1 = scipy.optimize.minimize_scalar(
                        lambda x: -explained(phs, x, z),
                        bounds=(dist - 0.08, dist + 0.08), method="bounded",
                        options=dict(xatol=1e-5))
                    if -r1.fun >= snap_keep * e:
                        cand = (-r1.fun, phs, float(r1.x), ph)
            if debug:
                print(f"  [clean]   cand φ={np.degrees(x0[0]):7.1f}° d={x0[1]:.3f} "
                      f"level={level[k]:.2f} → φ={np.degrees(ph):7.2f}° d={dist:.3f} "
                      f"explained {e:.3g}"
                      + (f" | snapped φ={np.degrees(cand[1]):7.2f}° {cand[0]:.3g}"
                         if cand[1] != ph else ""))
            if best is None or cand[0] > best[0]:
                best = cand
        _, ph, dist, raw = best
        raw_deg = float(np.degrees(raw))
        if rectilinear and ref_ang is None:
            ref_ang = ph                             # first wall sets the axes
        if debug:
            print(f"  [clean] → wall φ={np.degrees(ph):7.2f}° d={dist:.3f}")

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
