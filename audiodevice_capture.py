#!/usr/bin/env python3
"""Capture raw Audiodevice microphones while playing a short chirp train."""

import argparse
import json
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import tukey


RATE = 48_000


def _find_mic_config():
    config_root = Path(__file__).resolve().parent.parent / "config/platforms"
    for path in sorted(config_root.glob("*/pcm/mic_in.json")):
        try:
            config = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (config.get("pcm_card") == "lahainayupikidp"
                and config.get("pcm_device") == "MultiMedia5"
                and config.get("sample_rate") == RATE
                and config.get("channels") == 6
                and "mixer" in config):
            return path
    return None


MIC_CONFIG = _find_mic_config()


class CaptureError(Exception):
    pass


def run_adb(serial, *args, **kwargs):
    kwargs.setdefault("timeout", 8)
    try:
        return subprocess.run(
            ["adb", "-s", serial, *args], check=True, text=True, **kwargs
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise CaptureError(f"ADB failed: {detail}") from exc


def shell(serial, *args, **kwargs):
    return run_adb(serial, "shell", shlex.join(str(a) for a in args), **kwargs)


def mixer_values(serial, card, name):
    result = shell(serial, "tinymix", "-D", card, "-v", name, capture_output=True)
    if result.stderr.strip() or not result.stdout.strip():
        raise CaptureError(f"Cannot read mixer {name}: {result.stderr.strip()}")
    return result.stdout.strip().split()


def mixer_set(serial, card, name, values):
    result = shell(serial, "tinymix", "-D", card, name, *values, capture_output=True)
    if result.stderr.strip():
        raise CaptureError(f"Cannot set mixer {name}: {result.stderr.strip()}")
    _check_mixer(name, values, mixer_values(serial, card, name))


def _check_mixer(name, values, actual):
    expected = [
        {"0": "Off", "1": "On"}.get(str(v), str(v))
        if i < len(actual) and actual[i] in ("Off", "On") else str(v)
        for i, v in enumerate(values)
    ]
    if actual[:len(expected)] != expected:
        raise CaptureError(f"Mixer {name}: expected {expected}, got {actual}")


def _tinymix_batch(serial, card, commands):
    """Run tinymix commands in one ADB call; return (exit code, output) for each."""
    marker = f"@@{uuid.uuid4().hex}"
    script = []
    for index, arguments in enumerate(commands):
        script.append(f"echo {marker} {index}; "
                      f"{shlex.join(['tinymix', '-D', str(card), *map(str, arguments)])} 2>&1; "
                      f"echo {marker} rc $?")
    output = run_adb(serial, "shell", "; ".join(script), capture_output=True,
                     timeout=8 + len(commands)).stdout
    results = [None] * len(commands)
    index, lines = None, []
    for line in output.splitlines():
        fields = line.split()
        if fields[:1] == [marker] and len(fields) == 2:
            index, lines = int(fields[1]), []
        elif fields[:2] == [marker, "rc"] and index is not None:
            results[index] = (int(fields[2]), "\n".join(lines).strip())
            index = None
        elif index is not None:
            lines.append(line)
    if any(result is None for result in results):
        raise CaptureError("Incomplete response from batched tinymix commands")
    return results


def mixer_values_many(serial, card, names):
    """Read several mixer controls in one ADB call."""
    values = {}
    for name, (code, text) in zip(names, _tinymix_batch(serial, card, [["-v", n] for n in names])):
        if code != 0 or not text:
            raise CaptureError(f"Cannot read mixer {name}: {text}")
        values[name] = text.split()
    return values


def mixer_set_many(serial, card, items):
    """Set and verify (name, values) pairs in order; every item is attempted."""
    commands = [[name, *map(str, values)] for name, values in items]
    commands += [["-v", name] for name, _values in items]
    results = _tinymix_batch(serial, card, commands)
    errors = []
    for (name, values), (code, text), (read_code, actual) in zip(
            items, results[:len(items)], results[len(items):]):
        try:
            if code != 0 or text:
                raise CaptureError(f"Cannot set mixer {name}: {text}")
            if read_code != 0 or not actual:
                raise CaptureError(f"Cannot read mixer {name}: {actual}")
            _check_mixer(name, values, actual.split())
        except CaptureError as exc:
            errors.append(str(exc))
    if errors:
        raise CaptureError("; ".join(errors))


def connected_serial():
    result = subprocess.run(["adb", "devices", "-l"], capture_output=True,
                            text=True, check=True, timeout=8)
    devices = []
    for line in result.stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "device":
            devices.append(fields[0])
    if len(devices) != 1:
        raise CaptureError(
            "Expected one connected device; specify --serial if multiple devices are connected"
        )
    return devices[0]


def locate_pcm(serial):
    cards = shell(serial, "cat", "/proc/asound/cards", capture_output=True).stdout
    matches = re.findall(r"^\s*(\d+)\s+\[lahainayupikidp\s*\]", cards, re.MULTILINE)
    if len(matches) != 1:
        raise CaptureError("Could not uniquely identify the six-channel audio card")
    card = int(matches[0])
    devices = shell(serial, "cat", "/proc/asound/pcm", capture_output=True).stdout
    pattern = rf"^\s*0*{card}-(\d+): MultiMedia5\b.*playback \d+.*capture \d+"
    matches = re.findall(pattern, devices, re.MULTILINE)
    if len(matches) != 1:
        raise CaptureError("Could not uniquely identify the MultiMedia5 duplex PCM")
    return card, int(matches[0])


def make_chirp(low, high):
    if not 10_000 <= low < high < RATE / 2:
        raise ValueError("Chirp band must start at or above 10 kHz and end below 24 kHz")
    samples = int(RATE * 0.005)
    t = np.arange(samples) / RATE
    return chirp(t, f0=low, f1=high, t1=0.005) * tukey(samples, 0.5)


def chirp_period(low, high, level_percent=1):
    """Return one 100 ms period of four packed S16 playback channels."""
    if not 0 < level_percent <= 20:
        raise ValueError("Playback level must be greater than 0 and no more than 20 percent")
    burst = make_chirp(low, high)
    level = round(32767 * level_percent / 100)
    if level < 1:
        raise ValueError("Playback level is too low to represent in 16-bit audio")
    pcm = np.zeros((RATE // 10, 4), dtype="<i2")
    pcm[:len(burst), :] = (burst[:, None] * level).astype("<i2")
    return pcm


def make_playback(path, low, high, level_percent=1, duration_seconds=2):
    if not isinstance(duration_seconds, int) or not 1 <= duration_seconds <= 10:
        raise ValueError("Playback duration must be an integer from 1 to 10 seconds")
    pcm = np.tile(chirp_period(low, high, level_percent), (duration_seconds * 10, 1))
    with wave.open(str(path), "wb") as wav:
        # Two ALSA S32 channels carry the four packed S16 playback channels.
        wav.setnchannels(2)
        wav.setsampwidth(4)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())


@contextmanager
def microphone_route(serial, stop_event=None):
    """Temporarily enable the raw microphone route and restore it."""
    def check_stopped():
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("Capture stopped")

    check_stopped()
    if shell(serial, "id", "-u", capture_output=True).stdout.strip() != "0":
        raise CaptureError("ADB must be root; run 'adb -s SERIAL root' first")
    card, device = locate_pcm(serial)
    route = "MultiMedia5 Mixer TX_CDC_DMA_TX_3"
    speaker = "PRI_MI2S_RX Audio Mixer MultiMedia5"
    if MIC_CONFIG is None:
        raise CaptureError("No compatible six-channel microphone configuration was found")
    settings = [next(iter(s.items())) for s in json.loads(MIC_CONFIG.read_text())["mixer"]["init"]]
    names = [name for name, _value in settings]
    current = mixer_values_many(serial, card, list(dict.fromkeys([route, speaker, *names])))
    if any(v != "Off" for v in current[route]):
        raise CaptureError("Microphone route is already in use; stop audio capture first")
    if "On" not in current[speaker]:
        raise CaptureError("Speaker route is not active; configure audio first")
    original = {name: current[name] for name in names}

    touched = []
    try:
        check_stopped()
        # Mark everything first: a failed batch may have applied some settings.
        touched = names
        mixer_set_many(serial, card, [(name, [str(value)]) for name, value in settings])
        yield card, device
    finally:
        errors = []
        if touched:
            restore = [(name, [{"Off": "0", "On": "1"}.get(v, v) for v in original[name]])
                       for name in reversed(touched)]
            try:
                mixer_set_many(serial, card, restore)
            except (CaptureError, subprocess.SubprocessError, OSError) as exc:
                errors.append(f"restore mixer: {exc}")
        if errors:
            active_error = sys.exc_info()[1]
            message = "; ".join(errors)
            if active_error is not None:
                message = f"{active_error}; cleanup: {message}"
            raise CaptureError(message)


def capture(serial, output, low, high, level_percent=1):
    if output.exists():
        raise CaptureError(f"Output already exists: {output}")
    if not output.parent.is_dir():
        raise CaptureError(f"Output directory does not exist: {output.parent}")

    remote_prefix = f"/data/local/tmp/audiodevice-capture-{uuid.uuid4().hex}"
    remote_chirp = remote_prefix + "-chirp.wav"
    remote_mics = remote_prefix + "-mics.wav"
    recording = None
    with tempfile.TemporaryDirectory(prefix="audiodevice-capture-") as directory:
        playback_file = Path(directory) / "chirp.wav"
        capture_file = Path(directory) / "mics.wav"
        make_playback(playback_file, low, high, level_percent)
        with microphone_route(serial) as (card, device):
            cleanup_errors = []
            try:
                run_adb(serial, "push", str(playback_file), remote_chirp, capture_output=True)
                recording = subprocess.Popen(
                    ["adb", "-s", serial, "shell", "tinycap", remote_mics,
                     "-D", str(card), "-d", str(device), "-c", "6",
                     "-r", str(RATE), "-b", "24", "-p", "192", "-n", "4", "-T", "4"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                time.sleep(0.7)
                if recording.poll() is not None:
                    raise CaptureError(f"Capture stopped early: {recording.communicate()[0]}")
                shell(serial, "tinyplay", remote_chirp, "-D", card, "-d", device,
                      "-p", "192", "-n", "4", timeout=10)
                log = recording.communicate(timeout=10)[0]
                if recording.returncode != 0 or not re.search(r"Captured [1-9]\d* frames", log):
                    raise CaptureError(f"Capture failed: {log.strip()}")
                run_adb(serial, "pull", remote_mics, str(capture_file), capture_output=True)
                with wave.open(str(capture_file), "rb") as wav:
                    if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (6, RATE, 4):
                        raise CaptureError("Unexpected microphone WAV format")
                    frames = wav.getnframes()
                    if frames < 3 * RATE:
                        raise CaptureError("Microphone recording is too short")
                    if len(wav.readframes(frames)) != frames * 6 * 4:
                        raise CaptureError("Microphone recording is truncated")
            finally:
                if recording is not None and recording.poll() is None:
                    try:
                        recording.communicate(timeout=6)
                    except subprocess.TimeoutExpired:
                        cleanup_errors.append("capture process did not stop in time")
                        try:
                            recording.terminate()
                            recording.wait(timeout=5)
                        except (subprocess.SubprocessError, OSError) as exc:
                            cleanup_errors.append(f"stop capture process: {exc}")
                try:
                    shell(serial, "rm", "-f", remote_chirp, remote_mics, capture_output=True)
                except (CaptureError, subprocess.SubprocessError, OSError) as exc:
                    cleanup_errors.append(f"remove temporary device files: {exc}")
                if cleanup_errors:
                    active_error = sys.exc_info()[1]
                    message = "; ".join(cleanup_errors)
                    if active_error is not None:
                        message = f"{active_error}; cleanup: {message}"
                    raise CaptureError(message)
        with capture_file.open("rb") as source:
            with output.open("xb") as destination:
                try:
                    while data := source.read(1024 * 1024):
                        destination.write(data)
                except (OSError, KeyboardInterrupt):
                    output.unlink()
                    raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True, help="ADB device serial")
    parser.add_argument("--band", type=float, nargs=2, default=(18_000, 22_000),
                        metavar=("LOW_HZ", "HIGH_HZ"))
    parser.add_argument("--level", type=float, default=1,
                        help="chirp peak as percent of 16-bit full scale (default 1, max 20)")
    parser.add_argument("output", type=Path, help="new WAV file for six microphone channels")
    args = parser.parse_args()
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        capture(args.serial, args.output, *args.band, args.level)
    except KeyboardInterrupt:
        print("Capture interrupted", file=sys.stderr)
        return 130
    except (CaptureError, ValueError, OSError, subprocess.SubprocessError,
            wave.Error, json.JSONDecodeError) as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1
    print(f"Recorded {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
