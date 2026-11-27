"""Person tracking and room mapping on a live Audiodevice stream, built on tracker.py.

`tracker.Tracker` supplies the matched filter, the background and its learned
per-delay noise level (`noise="per_bin"`), and the Kalman filter.
This module adds what the Audiodevice hardware needs, as measured against an
LD-19 LiDAR:

* Timing: playback and capture are sample-locked (pings repeat every PERIOD
  samples with < 0.2 samples of jitter), so per-ping latency sync is not
  needed. The phase is locked once from a fold of settled pings and checked
  on every ping.
* Channels: only the front mics (0, 2, 4) are used; the rear ones are 3-7x
  noisier. Front mics and speaker are collinear, so a `front` half-plane
  tells front from back.
* Detection: a person's echo is weak per ping (SNR 1-3 past ~2 m) while
  static echoes fluctuate, so detection is track-before-detect in range: the
  normalised echo energy, averaged across mics, is integrated over the last
  INTEGRATE pings along constant-velocity hypotheses.
* Angle: echo peaks jitter ~8 cm independently per mic (speckle of an
  extended body, ~2 kHz usable bandwidth), so with a 32 cm aperture the
  angle is coarse. Range is accurate to ~5 cm. The Kalman measurement
  covariance is polar to match: tight in range, wide in angle.
* False tracks: the noise floor drifts, so each ping's range profile is
  normalised by its median (CFAR over range), and a new track is reported
  only after CONFIRM consecutive hits.
* Walls: the same angle limit applies to wall echoes. `wallmap.find_walls`
  locks onto the facing wall's distance but not its angle, and the side walls
  (at grazing incidence) don't echo back. So the map reports the facing wall
  only: the strongest range-compensated static echo, taken as parallel to
  the mounting plane.
"""

import dataclasses
from collections import deque
from dataclasses import dataclass

import numpy as np
import scipy.ndimage

import audiodevice_capture as capture
import audiodevice_tracking as geometry
import tracker as T


PERIOD = geometry.PERIOD
FRONT_CHANNELS = (0, 2, 4)
SETTLE_PINGS = 8        # the first pings after playback starts are incoherent
LOCK_PINGS = 10         # pings folded to find the chirp phase
# Fold peak (summed envelope of all mics) → transmission instant, in samples.
# Calibrated against LiDAR: makes echo path lengths match its ranges.
LOCK_OFFSET = 57
WINDOW = 2400           # samples per ping given to the tracker (~8.6 m range)
LOCK_MAX_SLIP = 3       # samples: direct-sound lag vs the background to stay locked
LOCK_LOST_PINGS = 3


@dataclass(frozen=True)
class MapperFrame:
    state: str                       # locking | calibrating | searching | tracking
    lock_quality: float = 0.0        # locking: fold peak/median; then 1/(1+|slip|)
    cal_progress: float = 0.0
    position: tuple | None = None    # Kalman (depth, lateral), m
    velocity: tuple | None = None
    range_m: float | None = None
    score: float = 0.0               # integrated detection score at the peak
    threshold: float = 0.0
    profile: np.ndarray | None = None   # integrated score per range bin
    envelopes: np.ndarray | None = None  # (samples, mics) residual / noise
    heat: np.ndarray | None = None       # (depths, laterals) single-ping score
    facing_wall_m: float | None = None   # distance to the wall facing the device
    static_absorbed: bool = False
    sample_index: int = -1           # stream sample of this ping's transmission


class AudiodeviceMapper:
    """Push interleaved 6-channel samples; get one MapperFrame per ping."""

    INTEGRATE = 4            # pings in the track-before-detect window
    DETECT = 2.7             # integrated score to start a track (empty room p99 ~2.6)
    MEASURE = 2.1            # ... to keep one, within the gate
    CONFIRM = 2              # hits before a tentative track is reported
    MAX_SPEED = 1.5          # m/s radial, velocity hypotheses of the integration
    RANGE_SIGMA = 0.05       # m, measurement 1σ
    ANGLE_SIGMA = 0.35       # rad, measurement 1σ
    MIN_GATE = 0.30          # m, ± range gate around the prediction
    STATIC_S = 8.0           # a track that doesn't move this long is absorbed
    STATIC_TOL = 0.12        # m, range spread that counts as not moving
    RANGE_STEP = 0.02
    ANGLE_WINDOW = 0.2       # m: ± range over which the mics' envelopes are aligned
    ANGLE_METHOD = "pairs"   # delay-and-sum of unsmoothed envelopes, integrated over
                             # ANGLE_INTEGRATE pings; "legacy": per-ping, max-filtered
    ANGLE_INTEGRATE = 5      # pings whose angle evidence is summed (one ping is too noisy)
    # "pairs": constant correction added to the measured angle, calibrated against
    # the LiDAR on walk1 (bias −12.6° on its first half, −3.6° on its second; ±5°).
    # Cause not found: not the LiDAR yaw (it sees the device bar square, ±1°).
    ANGLE_OFFSET_DEG = 8.0

    def __init__(self, low_hz=18_000, high_hz=22_000, *, channels=FRONT_CHANNELS,
                 max_range_m=4.0, lock_offset=LOCK_OFFSET, angle_offset_deg=None,
                 heat_depths=None, heat_laterals=None):
        self.channels = list(channels)
        self.chirp = T.Chirp(fs=capture.RATE, f0=low_hz, f1=high_hz)
        self.mic_pos = geometry.MIC_POSITIONS[self.channels]
        self.tx_pos = geometry.TX_POSITION
        self.max_range_m = max_range_m
        self.lock_offset = lock_offset
        if angle_offset_deg is not None:      # e.g. 0 for simulated audio, which lacks the bias
            self.ANGLE_OFFSET_DEG = angle_offset_deg
        self.dt = PERIOD / capture.RATE
        self.k_per_m = capture.RATE / T.C
        self._lock_tracker = T.Tracker(geometry.MIC_POSITIONS, self.tx_pos, self.chirp,
                                       WINDOW, sync=False, front=self._front())
        centre = self.mic_pos.mean(axis=0)
        self.ranges = np.arange(0.5, max_range_m + 1e-9, self.RANGE_STEP)
        broadside = centre + np.outer(self.ranges, [1.0, 0.0])
        self._profile_idx = np.stack([
            self._path_index(broadside, m) for m in self.mic_pos])  # (mics, ranges)
        self._angles = np.radians(np.arange(-75, 76, 1.0))
        self.heat_depths = np.arange(0.3, 4.001, 0.05) if heat_depths is None else heat_depths
        self.heat_laterals = (np.arange(-2.0, 2.001, 0.05) if heat_laterals is None
                              else heat_laterals)
        d, l = np.meshgrid(self.heat_depths, self.heat_laterals, indexing="ij")
        grid = np.stack([d, l], axis=-1).reshape(-1, 2)
        self._heat_idx = np.stack([self._path_index(grid, m) for m in self.mic_pos])
        self._heat_shape = d.shape
        self.buffer = np.empty((0, 6), np.float32)
        self.consumed = 0            # stream samples dropped from the buffer
        self.reset_calibration(relock=True)

    # ── setup ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _front():
        return T.wallmap.Wall(np.array([1.0, 0.0]), 0.0)

    def _path_index(self, points, mic):
        paths = np.linalg.norm(points - self.tx_pos, axis=-1) + np.linalg.norm(points - mic, axis=-1)
        return np.clip(np.rint(paths * self.k_per_m).astype(int), 0, WINDOW - 1)

    def _new_tracker(self):
        return T.Tracker(self.mic_pos, self.tx_pos, self.chirp, WINDOW, sync=False,
                         front=self._front(), noise="per_bin")

    def reset_calibration(self, relock=False):
        """Learn the background again (keep the room still and empty)."""
        self.tracker = self._new_tracker()
        self._cal = []
        self.facing_wall = None
        self.history = deque(maxlen=self.INTEGRATE)
        self.angle_history = deque(maxlen=self.ANGLE_INTEGRATE)
        self.recent_raw = deque(maxlen=20)
        self.static_ranges = deque(maxlen=int(round(self.STATIC_S / self.dt)))
        self.heat = None
        self.lost_pings = 0
        self.hits = 0
        self._bg_direct = None
        if relock:
            self.next_start = None   # absolute stream index of the next ping window

    # ── stream ────────────────────────────────────────────────────────────────
    def push(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != 6 or not np.isfinite(samples).all():
            raise ValueError("Expected finite audio samples with six channels")
        self.buffer = np.concatenate((self.buffer, samples), axis=0)
        frames = []
        if self.next_start is None:
            frame = self._try_lock()
            if frame is not None:
                frames.append(frame)
        while self.next_start is not None:
            lo = self.next_start - self.consumed
            if lo + WINDOW > len(self.buffer):
                break
            raw = self.buffer[lo:lo + WINDOW].astype(np.float64)
            index = self.next_start
            self.next_start += PERIOD
            frame = self._process(raw)
            frames.append(dataclasses.replace(frame, sample_index=index))
        self._trim()
        return frames

    def _trim(self):
        keep_from = (len(self.buffer) - (SETTLE_PINGS + LOCK_PINGS + 2) * PERIOD
                     if self.next_start is None else self.next_start - self.consumed)
        keep_from = max(0, min(keep_from, len(self.buffer)))
        if keep_from:
            self.buffer = self.buffer[keep_from:]
            self.consumed += keep_from

    def _try_lock(self):
        span = (LOCK_PINGS + 1) * PERIOD
        lo = max(0, SETTLE_PINGS * PERIOD - self.consumed)
        if len(self.buffer) - lo < span:
            return MapperFrame("locking")
        seg = self.buffer[lo:lo + span].astype(np.float64)
        env = np.abs(self._lock_tracker.analytic(seg))[:, :LOCK_PINGS * PERIOD]
        fold = env.reshape(6, LOCK_PINGS, PERIOD).mean(axis=1).sum(axis=0)
        peak = int(np.argmax(fold))
        quality = float(fold[peak] / (np.median(fold) + 1e-12))
        if quality < 4.0:
            return MapperFrame("locking", lock_quality=quality)
        start = self.consumed + lo + peak - self.lock_offset
        while start < self.consumed + lo:
            start += PERIOD
        self.next_start = start
        return None

    # ── one ping ──────────────────────────────────────────────────────────────
    def _lock_slip(self, raw):
        """Shift (samples) of the direct-sound peak vs the background's —
        insensitive to its level changing (a hand near the device)."""
        n = int(self.tracker.blank_m * self.k_per_m) + 60
        a = np.abs(self.tracker.analytic(raw[:n])).sum(axis=0)
        if self._bg_direct is None:
            bg = np.abs(self.tracker.analytic(self.tracker.bg[:n])).sum(axis=0)
            self._bg_direct = int(np.argmax(bg))
        return int(np.argmax(a)) - self._bg_direct

    def _process(self, raw6):
        tr = self.tracker
        raw = raw6[:, self.channels]
        if tr.bg is None:
            return self._calibrate(raw)
        slip = self._lock_slip(raw)
        match = 1.0 / (1.0 + abs(slip))
        if abs(slip) > LOCK_MAX_SLIP:
            self.lost_pings += 1
            if self.lost_pings >= LOCK_LOST_PINGS:   # stream glitch: start over
                self.reset_calibration(relock=True)
                return MapperFrame("locking", lock_quality=match)
            return self._coast_frame(match)
        self.lost_pings = 0
        self.recent_raw.append(raw)

        tr.frame += 1
        if tr.frame % tr.MAP_EVERY == 0:
            self.facing_wall = self._facing_wall()
        env = np.abs(tr.analytic(raw - tr.bg))
        b = int(tr.blank_m * self.k_per_m)
        nb = scipy.ndimage.maximum_filter1d(tr.noise, tr.NOISE_SPREAD, axis=1)
        nb = np.maximum(nb, np.median(nb[:, b:], axis=1, keepdims=True)) + 1e-12
        en = env / nb
        en[:, :b] = 0.0
        spread = scipy.ndimage.maximum_filter1d(en, tr.NOISE_SPREAD, axis=1)
        mics = np.arange(len(self.channels))[:, None]
        prof = spread[mics, self._profile_idx].mean(axis=0)
        self.history.append(prof / (np.median(prof) + 1e-12))   # floor drifts: CFAR over range
        profile, speed = self._integrate()
        heat = np.exp(np.log(spread[mics, self._heat_idx] + 0.1).mean(axis=0)).reshape(self._heat_shape)
        self.heat = heat if self.heat is None else 0.5 * self.heat + 0.5 * heat

        kf = tr.kf
        lost = kf.lost
        if lost:
            i = int(np.argmax(profile))
            found = profile[i] >= self.DETECT and len(self.history) == self.INTEGRATE
            threshold = self.DETECT
        else:
            pred, pred_vel, pred_std = kf.predict(self.dt)
            r_pred = float(np.linalg.norm(pred - tr.centre))
            gate = max(3 * pred_std, self.MIN_GATE)
            sel = np.flatnonzero(np.abs(self.ranges - r_pred) <= gate)
            i = int(sel[np.argmax(profile[sel])]) if len(sel) else int(np.argmax(profile))
            found = len(sel) > 0 and profile[i] >= self.MEASURE
            threshold = self.MEASURE

        accepted = False
        if found:
            rng = float(self.ranges[i])
            prior = None if lost else pred
            z = self._angle_fix(spread, rng, prior, en)
            R = self._polar_covariance(z)
            if lost:
                kf.reset(z)
                kf.x[2:] = speed[i] * (z - tr.centre) / max(rng, 1e-6)
                self.hits = 1
                accepted = True
            else:
                accepted = kf.update(z, R, self.dt)
                self.hits += accepted
        else:
            kf.coast(self.dt)
        if not accepted and not lost and self.hits < self.CONFIRM:
            kf.started = False                   # tentative track: one miss drops it

        if accepted and not lost:
            tr._update_background(raw, obj_paths=tr.paths(kf.x[:2]), residual_env=env)
        elif not found:
            tr._update_background(raw, residual_env=env)

        absorbed = self._absorb_static(accepted)
        tracking = not kf.lost and not absorbed and self.hits >= self.CONFIRM
        rng = float(np.linalg.norm(kf.x[:2] - tr.centre)) if tracking else None
        return MapperFrame(
            "tracking" if tracking else "searching", lock_quality=match, cal_progress=1.0,
            position=tuple(map(float, kf.x[:2])) if tracking else None,
            velocity=tuple(map(float, kf.x[2:])) if tracking else None,
            range_m=rng, score=float(profile[i]), threshold=threshold, profile=profile,
            envelopes=en.T, heat=self.heat, facing_wall_m=self.facing_wall, static_absorbed=absorbed)

    def _calibrate(self, raw):
        """Background (median) and per-delay noise from CAL_FRAMES pings."""
        tr = self.tracker
        self._cal.append(raw.copy())
        if len(self._cal) < tr.CAL_FRAMES:
            return MapperFrame("calibrating", lock_quality=1.0,
                               cal_progress=len(self._cal) / tr.CAL_FRAMES)
        cal, self._cal = np.stack(self._cal), []
        tr.bg = np.median(cal, axis=0)
        tr.noise = np.mean([np.abs(tr.analytic(f - tr.bg)) for f in cal], axis=0)
        self.facing_wall = self._facing_wall()
        return MapperFrame("calibrating", lock_quality=1.0, cal_progress=1.0,
                           facing_wall_m=self.facing_wall)

    def _facing_wall(self):
        """Distance of a wall parallel to the mounting plane that explains the
        strongest static echo, range-compensated (× path) as in find_walls."""
        tr = self.tracker
        level = np.abs(tr.analytic(tr.bg)).sum(axis=0) * np.arange(WINDOW)
        level[:int(tr.map_blank_m * self.k_per_m)] = 0.0
        k = int(np.argmax(level))
        if 0 < k < WINDOW - 1:                  # sub-sample peak
            y0, y1, y2 = level[k - 1:k + 2]
            k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - 1e-12)
        path = k / self.k_per_m
        # image of the speaker at depth 2D: path² = (2D)² + (mic − tx lateral)²
        off = self.mic_pos[:, 1] - self.tx_pos[1]
        return float(np.mean(np.sqrt(np.maximum(path**2 - off**2, 0.0))) / 2)

    def _coast_frame(self, match):
        self.tracker.kf.coast(self.dt)
        return MapperFrame("searching", lock_quality=match, cal_progress=1.0,
                           facing_wall_m=self.facing_wall)

    def _integrate(self):
        """Mean over the last pings along each constant-velocity hypothesis;
        returns (best score per range bin, its radial speed)."""
        h = list(self.history)[::-1]             # h[k]: k pings ago
        best = h[0].copy()
        speed = np.zeros_like(best)
        n = len(h)
        if n == 1:
            return best, speed
        steps = int(round(self.MAX_SPEED * self.dt * (n - 1) / self.RANGE_STEP))
        for s in range(-steps, steps + 1):
            v = s * self.RANGE_STEP / (self.dt * (n - 1))
            acc = np.zeros_like(best)
            for k, prof in enumerate(h):
                shift = int(round(v * k * self.dt / self.RANGE_STEP))
                acc += np.roll(prof, shift) if shift else prof
                if shift > 0:
                    acc[:shift] -= prof[-shift:]
                elif shift < 0:
                    acc[shift:] -= prof[:-shift]
            acc /= n
            better = acc > best
            best[better], speed[better] = acc[better], v
        return best, speed

    def _angle_fix(self, spread, rng, prior, en=None):
        """Point on the arc at range rng (from the array centre) where the
        mics' echoes line up best, nudged towards the prediction.

        "pairs": the per-mic envelopes (noise-normalised, not max-filtered:
        the angle is in delay differences of only a few samples across the
        aperture) are delay-and-summed along each direction over ±ANGLE_WINDOW
        of range, so the whole extended echo votes, not one speckle peak.
        Only cross-products of mic pairs count, so one mic's noise burst
        can't pull the angle towards itself."""
        if self.ANGLE_METHOD == "pairs" and en is not None:
            return self._angle_fix_pairs(en, rng, prior)
        centre = self.tracker.centre
        pts = centre + rng * np.stack([np.cos(self._angles), np.sin(self._angles)], axis=1)
        mics = np.arange(len(self.channels))[:, None]
        idx = np.stack([self._path_index(pts, m) for m in self.mic_pos])
        score = np.log(spread[mics, idx] + 0.1).sum(axis=0)
        if prior is not None:
            a0 = np.arctan2(prior[1] - centre[1], prior[0] - centre[0])
            score -= 0.5 * ((self._angles - a0) / (2 * self.ANGLE_SIGMA))**2
        return pts[int(np.argmax(score))]

    def _angle_fix_pairs(self, en, rng, prior):
        centre = self.tracker.centre
        dr = np.arange(-self.ANGLE_WINDOW, self.ANGLE_WINDOW + 1e-9, self.RANGE_STEP)
        u = np.stack([np.cos(self._angles), np.sin(self._angles)], axis=1)
        pts = centre + (rng + dr[:, None, None]) * u[None]              # (ranges, angles, 2)
        d_tx = np.linalg.norm(pts - self.tx_pos, axis=-1)
        e = []
        for j, mic in enumerate(self.mic_pos):
            k = (d_tx + np.linalg.norm(pts - mic, axis=-1)) * self.k_per_m
            k0 = np.clip(np.floor(k).astype(int), 0, en.shape[1] - 2)
            f = k - k0
            e.append(en[j, k0] * (1 - f) + en[j, k0 + 1] * f)
        n = len(e)
        s = sum(e[a] * e[b] for a in range(n) for b in range(a + 1, n)).sum(axis=0)
        if prior is None:                       # new track: old evidence is someone else's
            self.angle_history.clear()
        self.angle_history.append((s - s.mean()) / (s.std() + 1e-12))
        s = np.sum(self.angle_history, axis=0)
        off = np.radians(self.ANGLE_OFFSET_DEG)
        if prior is not None:
            a0 = np.arctan2(prior[1] - centre[1], prior[0] - centre[0]) - off
            s = s - 0.5 * ((self._angles - a0) / (2 * self.ANGLE_SIGMA))**2
        a = self._angles[int(np.argmax(s))] + off
        return centre + rng * np.array([np.cos(a), np.sin(a)])

    def _polar_covariance(self, z):
        d = z - self.tracker.centre
        r = max(np.linalg.norm(d), 1e-6)
        u = d / r
        J = np.column_stack([u, [-u[1], u[0]]])
        return J @ np.diag([self.RANGE_SIGMA**2, (r * self.ANGLE_SIGMA)**2]) @ J.T

    def _absorb_static(self, accepted):
        """A track that stays at one range is a moved object, not a person
        walking: make it part of the background."""
        kf = self.tracker.kf
        if not accepted or kf.lost:
            self.static_ranges.clear()
            return False
        self.static_ranges.append(float(np.linalg.norm(kf.x[:2] - self.tracker.centre)))
        h = np.array(self.static_ranges)
        if len(h) < self.static_ranges.maxlen or np.ptp(h) > self.STATIC_TOL:
            return False
        self.tracker.bg = np.median(np.stack(self.recent_raw), axis=0)
        kf.started = False
        self.static_ranges.clear()
        self.history.clear()
        return True


class LiveMapper:
    """AudiodeviceMapper behind the live GUIs' interface.

    ``push`` returns ``audiodevice_tracking.TrackingFrame`` objects, so
    audiodevice_live.py and audiodevice_server.py show the mapper's tracking:
    ``envelopes`` are the front mics' echoes (change from the background in
    noise units; rear-mic columns are zero) indexed by echo path from the
    speaker, ``scores`` the mapper's change-evidence grid over
    ``depths`` × ``laterals``."""

    def __init__(self, low_hz=18_000, high_hz=22_000, *, notch_hz=(), **mapper_args):
        self.mapper = AudiodeviceMapper(low_hz, high_hz, **mapper_args)
        self.notch_filter = geometry.NotchFilter(notch_hz)
        self.depths = self.mapper.heat_depths
        self.laterals = self.mapper.heat_laterals
        self.channels = list(self.mapper.channels)
        self._last = (None, None)          # envelopes, heat of the last processed ping

    def reset_calibration(self):
        self.mapper.reset_calibration()
        self._last = (None, None)

    def push(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != 6 or not np.isfinite(samples).all():
            raise ValueError("Expected finite audio samples with six channels")
        return [self._frame(f) for f in self.mapper.push(self.notch_filter.push(samples))]

    def _frame(self, f):
        TF = geometry.TrackingFrame
        if f.state == "locking":
            return TF("syncing", f.lock_quality, 0)
        calibration = int(round(f.cal_progress * geometry.CALIBRATION_UPDATES))
        if f.state == "calibrating":
            return TF("calibrating", f.lock_quality, calibration, facing_wall_m=f.facing_wall_m)
        if f.envelopes is not None:
            env = np.zeros((len(f.envelopes), 6))
            env[:, self.channels] = f.envelopes
            self._last = (env, f.heat)
        else:                               # ping skipped (sync check): keep the display
            env = self._last[0]
        heat = f.heat if f.heat is not None else self._last[1]
        tracking = f.state == "tracking"
        return TF("tracking" if tracking else "no_target", f.lock_quality, calibration, env,
                  heat, f.threshold, f.position if tracking else None,
                  len(self.channels) if tracking else 0, f.facing_wall_m)


def live_model(low_hz, high_hz, *, notch_hz=(), raw=False):
    if raw:
        return geometry.AudiodeviceTracker(low_hz, high_hz, notch_hz=notch_hz, raw=True)
    return LiveMapper(low_hz, high_hz, notch_hz=notch_hz)
