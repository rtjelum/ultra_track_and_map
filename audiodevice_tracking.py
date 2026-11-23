"""Experimental chirp synchronization and 2D tracking for Audiodevice."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, hilbert

import audiodevice_capture as capture
import tracking


SPEED_OF_SOUND = 343.0
PERIOD = capture.RATE // 10
PINGS_PER_UPDATE = 5
# Each update folds the latest PINGS_PER_UPDATE pings but advances one ping.
HOP = PERIOD
CALIBRATION_UPDATES = 4
REFERENCE_CHANNEL = 1

# Coordinates are (depth from the speaker, lateral offset from device centre),
# lateral positive to the RIGHT seen from behind the device looking into the
# room (the LiDAR's frame). The speaker is at the right end; along the bar from
# it: mic 2 (11 cm), mic 4 (27 cm), mic 0 (43 cm). This is the supplied table
# mirrored (it was given as seen from the front) with channels 0 and 4 swapped,
# found from recordings/walk1 (LiDAR walker and direct-sound arrival order) and
# confirmed on the hardware. Rear mics (1, 3, 5) are assumed to sit behind
# 0, 2, 4 as in the table; that pairing is not verified.
TX_POSITION = np.array([0.0, 0.19])
MIC_POSITIONS = np.array([
    [0.0, 0.19 - 0.43], [-0.035, 0.19 - 0.43],
    [0.0, 0.19 - 0.11], [-0.035, 0.19 - 0.11],
    [0.0, 0.19 - 0.27], [-0.035, 0.19 - 0.27],
])


@dataclass(frozen=True)
class TrackingFrame:
    state: str
    sync_ratio: float
    calibration: int
    envelopes: np.ndarray | None = None
    scores: np.ndarray | None = None
    threshold: float | None = None
    position: tuple[float, float] | None = None
    support: int = 0


class AudiodeviceTracker:
    """Track changes from a stationary background, not calibrated ground truth."""

    def __init__(self, low_hz=18_000, high_hz=22_000):
        chirp = capture.make_chirp(low_hz, high_hz)
        self.reference = chirp * np.hanning(len(chirp))
        self.buffer = np.empty((0, 6), dtype=np.float32)
        self.anchor = None
        self.pending_anchor = None
        self.pending_count = 0
        self.depths = np.arange(0.30, 2.501, 0.05)
        self.laterals = np.arange(-1.50, 1.501, 0.05)
        depth, lateral = np.meshgrid(self.depths, self.laterals, indexing="ij")
        positions = np.stack((depth, lateral), axis=-1)
        reference_path = np.linalg.norm(MIC_POSITIONS[REFERENCE_CHANNEL] - TX_POSITION)
        paths = tracking.bistatic_paths(positions, MIC_POSITIONS, TX_POSITION)
        self.delay_samples = (paths - reference_path) * capture.RATE / SPEED_OF_SOUND
        self.reset_calibration()

    def reset_calibration(self):
        self.calibration_frames = []
        self.background = None
        self.noise = None
        self.threshold = None
        self.calibration_phase = 0
        # Candidates from the last PINGS_PER_UPDATE updates; only the oldest
        # one shares no pings with the current update.
        self.recent_candidates = deque(maxlen=PINGS_PER_UPDATE)

    def _lose_sync(self):
        self.anchor = None
        self.pending_anchor = None
        self.pending_count = 0
        self.reset_calibration()

    def _score(self, envelopes):
        indices = np.arange(PERIOD)
        ratios = np.empty(self.delay_samples.shape)
        for ch in range(6):
            delay = self.delay_samples[..., ch]
            signal = np.interp(delay, indices, envelopes[:, ch], left=0, right=0)
            noise = np.interp(delay, indices, self.noise[:, ch], left=np.inf, right=np.inf)
            ratios[..., ch] = signal / noise
        scores = np.mean(np.log1p(ratios), axis=-1)
        return scores, ratios

    def _calibrate(self):
        frames = np.stack(self.calibration_frames)
        self.background = (np.median(frames.real, axis=0)
                           + 1j * np.median(frames.imag, axis=0))
        differences = np.abs(frames - self.background)
        noise = np.median(differences, axis=0)
        floor = np.maximum(np.median(noise[50:1200], axis=0), 1e-12)
        self.noise = np.maximum(noise, floor[None, :] * 0.5)
        background_max = max(float(self._score(frame)[0].max()) for frame in differences)
        self.threshold = max(1.5, background_max + 0.25)
        self.calibration_frames.clear()

    def _process_group(self, samples):
        correlated = fftconvolve(samples, self.reference[::-1, None], mode="valid", axes=0)
        folded = correlated.reshape(PINGS_PER_UPDATE, PERIOD, 6).mean(axis=0)
        analytic = hilbert(folded, axis=0)
        reference_env = np.abs(analytic[:, REFERENCE_CHANNEL])
        median = float(np.median(reference_env))
        if median <= 1e-12:
            self._lose_sync()
            return TrackingFrame("syncing", 0.0, 0)

        if self.anchor is None:
            index = int(np.argmax(reference_env))
            quality = float(reference_env[index] / median)
            if quality < 4.5:
                self.pending_anchor = None
                self.pending_count = 0
                return TrackingFrame("syncing", quality, 0)
            if self.pending_anchor is not None:
                delta = (index - self.pending_anchor + PERIOD // 2) % PERIOD - PERIOD // 2
                self.pending_count = self.pending_count + 1 if abs(delta) <= 10 else 1
            else:
                self.pending_count = 1
            self.pending_anchor = index
            # Lock only once the reference is stable across non-overlapping pings.
            if self.pending_count <= PINGS_PER_UPDATE:
                return TrackingFrame("syncing", quality, 0)
            self.anchor = index
        else:
            candidates = (self.anchor + np.arange(-10, 11)) % PERIOD
            index = int(candidates[np.argmax(reference_env[candidates])])
            quality = float(reference_env[index] / median)
            if quality < 2.5:
                self._lose_sync()
                return TrackingFrame("syncing", quality, 0)
            self.anchor = index

        neighbours = reference_env[(index + np.arange(-1, 2)) % PERIOD]
        fraction = float(np.clip(tracking.parabolic_peak(neighbours, 1) - 1, -0.5, 0.5))
        freqs = np.fft.fftfreq(PERIOD)
        shift = np.exp(2j * np.pi * freqs * (index + fraction))
        aligned = np.fft.ifft(np.fft.fft(analytic, axis=0) * shift[:, None], axis=0)
        direct = aligned[0, REFERENCE_CHANNEL]
        if abs(direct) <= 1e-12:
            self._lose_sync()
            return TrackingFrame("syncing", quality, 0)
        if (self.background is not None
                and abs(direct) < 0.25 * abs(self.background[0, REFERENCE_CHANNEL])):
            self._lose_sync()
            return TrackingFrame("syncing", quality, 0)
        aligned *= np.conj(direct) / abs(direct)

        if self.background is None:
            # Calibrate from non-overlapping updates so the noise estimate is independent.
            if self.calibration_phase == 0:
                self.calibration_frames.append(aligned)
            self.calibration_phase = (self.calibration_phase + 1) % PINGS_PER_UPDATE
            progress = len(self.calibration_frames)
            if progress == CALIBRATION_UPDATES:
                self._calibrate()
            return TrackingFrame("calibrating", quality, progress, np.abs(aligned))

        envelopes = np.abs(aligned - self.background)
        scores, ratios = self._score(envelopes)
        peak = np.unravel_index(int(np.argmax(scores)), scores.shape)
        strength = float(scores[peak])
        support = int(np.count_nonzero(ratios[peak] >= 3.0))
        if strength <= self.threshold or support < 3:
            self.recent_candidates.append(None)
            return TrackingFrame("no_target", quality, CALIBRATION_UPDATES,
                                 envelopes, scores, self.threshold, support=support)

        candidate = np.array([self.depths[peak[0]], self.laterals[peak[1]]])
        previous = (self.recent_candidates[0]
                    if len(self.recent_candidates) == PINGS_PER_UPDATE else None)
        confirmed = previous is not None and np.linalg.norm(candidate - previous) <= 0.35
        self.recent_candidates.append(candidate)
        position = (float(candidate[0]), float(candidate[1]))
        return TrackingFrame("tracking" if confirmed else "candidate", quality,
                             CALIBRATION_UPDATES, envelopes, scores, self.threshold,
                             position, support)

    def push(self, samples):
        """Process complete groups from arbitrary chunks of interleaved audio."""
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != 6 or not np.isfinite(samples).all():
            raise ValueError("Expected finite audio samples with six channels")
        if len(samples) == 0:
            return []
        self.buffer = np.concatenate((self.buffer, samples), axis=0)
        stride = PINGS_PER_UPDATE * PERIOD
        required = stride + len(self.reference) - 1
        result = []
        while len(self.buffer) >= required:
            result.append(self._process_group(self.buffer[:required]))
            self.buffer = self.buffer[HOP:]
        return result
