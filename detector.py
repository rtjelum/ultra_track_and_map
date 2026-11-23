#!/usr/bin/env python3
"""
6-mic linear array ultrasonic motion detector.

Captures all channels simultaneously from a USB multi-channel audio interface,
applies a bandpass filter around the transmit frequency, extracts signal
envelopes, detects motion per mic, and computes GCC-PHAT TDOA for each mic
pair to estimate angle of arrival.

Requires:
    pip install sounddevice numpy scipy
Hardware:
    - 6-channel USB audio interface supporting 48 kHz sample rate
    - 6x 20 kHz receivers wired to interface inputs
    - transmitter.py running on GPIO18
"""

import sounddevice as sd
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.signal import hilbert
import argparse
import sys

# ── config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 96_000        # Hz — must be >= 2 × TX_FREQ
CHUNK_FRAMES    = 4096
N_MICS          = 6
TX_FREQ         = 20_000
BANDPASS_BW     = 1_500         # Hz either side of TX_FREQ
MOTION_THRESH   = 0.15          # normalised envelope Δ to trigger event
BASELINE_ALPHA  = 0.05          # EMA weight for baseline drift correction
SPEED_OF_SOUND  = 343.0         # m/s
# Diagonal baseline: 10 cm in x + 10 cm in y → Euclidean = sqrt(0.1²+0.1²)
MIC_SPACING_M   = 0.14142       # metres (diagonal pair separation)

# pairs: (channel_index_a, channel_index_b)  — zero-based
PAIRS = [(0, 1), (2, 3), (4, 5)]


# ── DSP helpers ───────────────────────────────────────────────────────────────
def make_bandpass(low_hz, high_hz, fs, order=6):
    nyq = fs / 2
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")
    return sos


def envelope(signal_1d):
    return np.abs(hilbert(signal_1d))


def gcc_phat(sig_a, sig_b, fs, mic_sep_m):
    n = len(sig_a) + len(sig_b) - 1
    n_fft = 1 << (n - 1).bit_length()
    A = np.fft.rfft(sig_a, n=n_fft)
    B = np.fft.rfft(sig_b, n=n_fft)
    cross = A * np.conj(B)
    gcc = np.fft.irfft(cross / (np.abs(cross) + 1e-10), n=n_fft)
    max_lag = int(fs * mic_sep_m / SPEED_OF_SOUND) + 2
    search = np.concatenate([gcc[-max_lag:], gcc[:max_lag + 1]])
    lag = np.argmax(search) - max_lag
    return lag / fs     # seconds


def tdoa_to_angle(tdoa_s, mic_sep_m):
    arg = np.clip(tdoa_s * SPEED_OF_SOUND / mic_sep_m, -1.0, 1.0)
    return np.degrees(np.arcsin(arg))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="6-mic ultrasonic motion detector")
    parser.add_argument("--device",  type=int,   default=None,
                        help="sounddevice input device index (omit to list devices)")
    parser.add_argument("--rate",    type=int,   default=SAMPLE_RATE)
    parser.add_argument("--freq",    type=int,   default=TX_FREQ)
    parser.add_argument("--thresh",  type=float, default=MOTION_THRESH)
    parser.add_argument("--spacing", type=float, default=MIC_SPACING_M,
                        help="Intra-pair mic spacing in metres")
    args = parser.parse_args()

    if args.device is None:
        print("Available audio devices:\n")
        print(sd.query_devices())
        print("\nRe-run with  --device <index>")
        sys.exit(0)

    sos      = make_bandpass(args.freq - BANDPASS_BW, args.freq + BANDPASS_BW, args.rate)
    baseline = None

    print(f"Device {args.device} | {N_MICS} ch | {args.rate} Hz")
    print(f"Bandpass {args.freq - BANDPASS_BW}–{args.freq + BANDPASS_BW} Hz")
    print(f"Motion threshold {args.thresh:.0%} | mic spacing {args.spacing*100:.1f} cm\n")

    with sd.InputStream(device=args.device,
                        channels=N_MICS,
                        samplerate=args.rate,
                        blocksize=CHUNK_FRAMES,
                        dtype="float32") as stream:
        while True:
            chunk, _ = stream.read(CHUNK_FRAMES)    # (frames, N_MICS)

            # filter + envelope per channel
            envs     = np.zeros(N_MICS)
            filtered = np.zeros_like(chunk)
            for ch in range(N_MICS):
                sig = sosfilt(sos, chunk[:, ch])
                filtered[:, ch] = sig
                envs[ch] = float(np.mean(envelope(sig)))

            # baseline init
            if baseline is None:
                baseline = envs.copy()
                print("Baseline captured — monitoring...\n")
                continue

            # motion detection
            motion_flags = np.zeros(N_MICS, dtype=bool)
            for ch in range(N_MICS):
                if baseline[ch] > 1e-9:
                    change = abs(envs[ch] - baseline[ch]) / baseline[ch]
                    if change > args.thresh:
                        motion_flags[ch] = True
                        print(f"  Motion  mic{ch+1}  Δ={change:.0%}")
                if not motion_flags[ch]:
                    baseline[ch] = ((1 - BASELINE_ALPHA) * baseline[ch]
                                    + BASELINE_ALPHA * envs[ch])

            # TDOA / direction per pair
            if motion_flags.any():
                for a, b in PAIRS:
                    tdoa  = gcc_phat(filtered[:, a], filtered[:, b],
                                     args.rate, args.spacing)
                    angle = tdoa_to_angle(tdoa, args.spacing)
                    side  = "→ right" if tdoa > 0 else "← left "
                    print(f"  Pair (mic{a+1},mic{b+1})  "
                          f"TDOA={tdoa*1e6:+.1f} µs  "
                          f"angle≈{angle:+.1f}°  {side}")
                print()


if __name__ == "__main__":
    main()
