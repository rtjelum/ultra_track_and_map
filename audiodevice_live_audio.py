"""Live Audiodevice microphone streaming over ADB."""

import os
import select
import shlex
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

import audiodevice_capture as capture


CHANNELS = 6
FRAME_BYTES = CHANNELS * 4
WAV_HEADER_BYTES = 44
MAX_PLAYBACK_BYTES = (0xFFFFFFFF - 36) // 8 * 8
STARTUP_TIMEOUT_SECONDS = 5


class RemoteProcess:
    def __init__(self, serial, arguments, *, binary=False, error_path=None, root=False):
        self.serial = serial
        self.root = root
        self.arguments = [str(arg) for arg in arguments]
        command = "echo $$; exec " + shlex.join(self.arguments)
        if error_path is not None:
            command += " 2>" + shlex.quote(error_path)
        transport = "exec-out" if binary else "shell"
        prefix = "su 0 " if root else ""
        self.process = subprocess.Popen(
            ["adb", "-s", serial, transport, prefix + "sh -c " + shlex.quote(command)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        self.pid = None
        try:
            line = bytearray()
            deadline = time.monotonic() + 8
            while not line.endswith(b"\n"):
                timeout = max(0, deadline - time.monotonic())
                ready, _, _ = select.select([self.process.stdout], [], [], timeout)
                if not ready:
                    raise capture.CaptureError(f"Timed out starting {self.arguments[0]}")
                part = os.read(self.process.stdout.fileno(), 1)
                if not part or len(line) > 16:
                    raise capture.CaptureError(f"Invalid process ID from {self.arguments[0]}")
                line.extend(part)
            self.pid = int(line)
            if self.pid <= 1:
                raise capture.CaptureError(f"Invalid process ID from {self.arguments[0]}")
        except (ValueError, OSError, capture.CaptureError):
            self.process.terminate()
            self.process.wait(timeout=5)
            raise

    def output(self):
        try:
            out, err = self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired as exc:
            raise capture.CaptureError(f"{self.arguments[0]} did not exit") from exc
        return (out + err).decode(errors="replace").strip()

    def stop(self):
        if self.pid is None:
            return
        result = subprocess.run(
            ["adb", "-s", self.serial, "shell", "-T",
             *(["su", "0"] if self.root else []), "cat", f"/proc/{self.pid}/cmdline"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
        )
        if result.returncode == 0:
            argv = result.stdout.rstrip(b"\0").split(b"\0")
            expected = [arg.encode() for arg in self.arguments]
            if not argv or Path(os.fsdecode(argv[0])).name != Path(self.arguments[0]).name:
                raise capture.CaptureError(f"Refusing to stop unexpected process {self.pid}")
            if len(expected) > 1 and not all(arg in argv for arg in expected[1:]):
                raise capture.CaptureError(f"Refusing to stop changed process {self.pid}")
            try:
                capture.shell(self.serial, *(["su", "0"] if self.root else []),
                              "kill", "-INT", self.pid, capture_output=True)
            except capture.CaptureError:
                exists = subprocess.run(
                    ["adb", "-s", self.serial, "shell", "test", "-d", f"/proc/{self.pid}"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
                )
                if exists.returncode == 0:
                    raise
                if exists.returncode != 1 or exists.stderr:
                    raise capture.CaptureError(
                        f"Cannot verify process {self.pid}: "
                        f"{exists.stderr.decode(errors='replace').strip()}"
                    )
        elif b"No such file" not in result.stderr:
            raise capture.CaptureError(
                f"Cannot inspect process {self.pid}: {result.stderr.decode(errors='replace').strip()}"
            )
        try:
            self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired as exc:
            self.process.terminate()
            self.process.wait(timeout=5)
            raise capture.CaptureError(f"{self.arguments[0]} did not stop on the device") from exc


def wav_header(data_bytes):
    """Header for Audiodevice's two-channel S32 playback stream."""
    return (b"RIFF" + struct.pack("<I", min(36 + data_bytes, 0xFFFFFFFF)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 2, capture.RATE, capture.RATE * 8, 8, 32)
            + b"data" + struct.pack("<I", data_bytes))


def stream(serial, low, high, level_percent, duration_seconds, stop_event, on_status, on_samples):
    """Emit live six-channel float samples until stopped or the optional duration ends.

    Playback and capture run through device FIFOs, so an unbounded session does
    not accumulate audio on the device. The device route is always released.
    """
    if duration_seconds is not None and (not isinstance(duration_seconds, int)
                                         or duration_seconds < 1):
        raise ValueError("Session duration must be a positive number of seconds")
    period = capture.chirp_period(low, high, level_percent)
    if duration_seconds is None:
        data_bytes = MAX_PLAYBACK_BYTES
    else:
        data_bytes = duration_seconds * capture.RATE * 8
        if data_bytes > MAX_PLAYBACK_BYTES:
            raise ValueError("Session duration is too long for one playback stream")

    prefix = f"/data/local/tmp/audiodevice-live-{uuid.uuid4().hex}"
    remote_header = prefix + "-header.bin"
    remote_period = prefix + "-period.bin"
    play_fifo = prefix + "-play.fifo"
    mics_fifo = prefix + "-mics.fifo"
    remote_reader_log = prefix + "-reader.log"
    remote_files = (remote_header, remote_period, play_fifo, mics_fifo, remote_reader_log)
    processes = []

    def check_stopped():
        if stop_event.is_set():
            raise InterruptedError("Live capture stopped")

    with tempfile.TemporaryDirectory(prefix="audiodevice-live-") as directory:
        header_file = Path(directory) / "header.bin"
        period_file = Path(directory) / "period.bin"
        header_file.write_bytes(wav_header(data_bytes))
        period_file.write_bytes(period.tobytes())
        on_status("Opening Audiodevice microphone route...")
        with capture.microphone_route(serial, stop_event) as (card, device):
            errors = []
            try:
                check_stopped()
                capture.run_adb(serial, "push", str(header_file), remote_header,
                                capture_output=True)
                capture.run_adb(serial, "push", str(period_file), remote_period,
                                capture_output=True)
                capture.shell(serial, "mkfifo", play_fifo, mics_fifo, capture_output=True)
                reader = RemoteProcess(serial, ["cat", mics_fifo], binary=True,
                                       error_path=remote_reader_log)
                processes.append(reader)
                recorder = RemoteProcess(
                    serial, ["tinycap", mics_fifo, "-D", card, "-d", device,
                             "-c", CHANNELS, "-r", capture.RATE, "-b", 24,
                             "-p", 192, "-n", 4],
                )
                processes.append(recorder)
                check_stopped()
                # Restarting playback on this duplex PCM disrupts capture, so one
                # stream carries the whole session (at most ~3.1 hours of WAV data).
                script = (f"{{ cat {shlex.quote(remote_header)}; "
                          f"while cat {shlex.quote(remote_period)}; do :; done; }} "
                          f"> {shlex.quote(play_fifo)}")
                processes.append(RemoteProcess(serial, ["sh", "-c", script]))
                player = RemoteProcess(
                    serial, ["tinyplay", play_fifo, "-D", card, "-d", device,
                             "-p", 192, "-n", 4],
                )
                processes.append(player)
                started = time.monotonic()
                playback_finished = None
                header_remaining = WAV_HEADER_BYTES
                received = False
                if duration_seconds is None:
                    on_status("Chirps active until Stop; keep the background still for calibration")
                else:
                    on_status("Chirps active; keep the background still for calibration")
                pending = bytearray()
                while True:
                    check_stopped()
                    ready, _, _ = select.select([reader.process.stdout], [], [], 0.2)
                    if ready:
                        chunk = os.read(reader.process.stdout.fileno(), 262144)
                        if not chunk:
                            log = capture.shell(serial, "cat", remote_reader_log,
                                                capture_output=True).stdout.strip()
                            detail = recorder.output() if recorder.process.poll() is not None else log
                            raise capture.CaptureError(f"Microphone stream ended: {detail}")
                        if header_remaining:
                            skipped = min(header_remaining, len(chunk))
                            header_remaining -= skipped
                            chunk = chunk[skipped:]
                        pending.extend(chunk)
                        count = len(pending) // FRAME_BYTES * FRAME_BYTES
                        if count:
                            received = True
                            samples = np.frombuffer(pending[:count], dtype="<i4").reshape(-1, CHANNELS)
                            on_samples(samples.astype(np.float32) / float(2**31))
                            del pending[:count]
                    elif not received and time.monotonic() - started > STARTUP_TIMEOUT_SECONDS:
                        raise capture.CaptureError("Microphone capture produced no samples")
                    if recorder.process.poll() is not None:
                        raise capture.CaptureError(f"Microphone capture stopped: {recorder.output()}")
                    if player.process.poll() is not None and playback_finished is None:
                        log = player.output()
                        if player.process.returncode != 0 or "Playing sample:" not in log:
                            raise capture.CaptureError(f"Playback failed: {log}")
                        expected = (MAX_PLAYBACK_BYTES / (capture.RATE * 8)
                                    if duration_seconds is None else duration_seconds)
                        if time.monotonic() - started < expected - 1:
                            raise capture.CaptureError(f"Playback ended too early: {log}")
                        playback_finished = time.monotonic()
                    if playback_finished is not None and time.monotonic() - playback_finished > 0.5:
                        break
                    if (duration_seconds is not None
                            and time.monotonic() - started > duration_seconds + 3):
                        raise capture.CaptureError("Timed out waiting for playback")
            finally:
                # Stop the reader before tinycap: once the host stops reading,
                # tinycap blocks on the full FIFO and cannot handle SIGINT.
                order = list(reversed(processes))
                if len(processes) >= 2:
                    order.remove(processes[0])
                    order.insert(order.index(processes[1]), processes[0])
                for process in order:
                    try:
                        process.stop()
                    except (capture.CaptureError, subprocess.SubprocessError, OSError) as exc:
                        errors.append(str(exc))
                try:
                    capture.shell(serial, "rm", "-f", *remote_files, capture_output=True)
                except (capture.CaptureError, subprocess.SubprocessError, OSError) as exc:
                    errors.append(f"remove temporary device files: {exc}")
                if errors:
                    active_error = sys.exc_info()[1]
                    message = "; ".join(errors)
                    if active_error is not None:
                        message = f"{active_error}; cleanup: {message}"
                    raise capture.CaptureError(message)
