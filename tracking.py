"""Shared geometry and peak helpers for the chirp trackers."""

import numpy as np


def bistatic_paths(position, mic_positions, tx_position):
    """Return transmitter-to-position-to-microphone paths in metres."""
    position = np.asarray(position, dtype=float)
    return (np.linalg.norm(position - tx_position, axis=-1)[..., None]
            + np.linalg.norm(position[..., None, :] - mic_positions, axis=-1))


def parabolic_peak(values, index):
    """Estimate a peak's fractional sample position from its neighbours."""
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, centre, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2 * centre + right
    if abs(denominator) < 1e-12:
        return float(index)
    return float(index) - 0.5 * (right - left) / denominator
