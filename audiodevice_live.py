#!/usr/bin/env python3
"""Experimental live Audiodevice chirp tracker with a Python GUI."""

import argparse
import queue
import signal
import subprocess
import sys
import threading
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

import audiodevice_capture as capture
import audiodevice_live_audio as audio
import audiodevice_tracking as tracker


COLORS = ("#ffae42", "#fa6d6d", "#8bd17c", "#44c2ad", "#89aaff", "#d599f0")


connected_serial = capture.connected_serial


class LiveDemo:
    def __init__(self, serial, band, level, duration=None):
        self.serial = serial
        self.band = band
        self.level = level
        self.duration = duration
        self.messages = queue.Queue()
        self.stop_event = threading.Event()
        self.recalibrate_event = threading.Event()
        self.worker = None
        self.stopping = False
        self.failed = False
        self.closed = False
        self.history = deque(maxlen=30)

        model = tracker.AudiodeviceTracker(*band)
        self.figure = plt.figure(figsize=(13, 7.5), facecolor="#101722")
        self.figure.suptitle("Audiodevice live chirp tracker — EXPERIMENTAL",
                             color="white", fontsize=15, y=0.97)
        grid = self.figure.add_gridspec(
            2, 2, left=0.07, right=0.97, bottom=0.18, top=0.88,
            width_ratios=(1.0, 1.15), height_ratios=(1.0, 0.75),
            wspace=0.25, hspace=0.30,
        )
        self.map_axis = self.figure.add_subplot(grid[:, 0])
        self.signal_axis = self.figure.add_subplot(grid[0, 1])
        self.info_axis = self.figure.add_subplot(grid[1, 1])
        for axis in (self.map_axis, self.signal_axis, self.info_axis):
            axis.set_facecolor("#1a2533")
            axis.tick_params(colors="#dce4ec")
            for spine in axis.spines.values():
                spine.set_color("#718096")

        extent = (model.laterals[0] - 0.025, model.laterals[-1] + 0.025,
                  model.depths[0] - 0.025, model.depths[-1] + 0.025)
        self.heat = self.map_axis.imshow(
            np.zeros((len(model.depths), len(model.laterals))),
            origin="lower", extent=extent, cmap="magma", vmin=0, vmax=2,
            interpolation="nearest", alpha=0.8, aspect="auto",
        )
        self.map_axis.plot([-0.19, 0.24], [0, 0], color="#e2e8f0", linewidth=4)
        self.map_axis.scatter(tracker.MIC_POSITIONS[:, 1], tracker.MIC_POSITIONS[:, 0],
                              color="#4cd1bf", s=32, label="Microphones")
        self.map_axis.scatter([tracker.TX_POSITION[1]], [tracker.TX_POSITION[0]],
                              marker="*", color="#ffd36b", s=130, label="Speaker")
        self.trail, = self.map_axis.plot([], [], color="#4cd1ff", alpha=0.65,
                                         linewidth=1.5, label="Recent candidates")
        self.marker, = self.map_axis.plot([], [], "o", color="#40dcff", markersize=11,
                                          markeredgecolor="white", label="Candidate")
        self.map_axis.set_xlim(model.laterals[0], model.laterals[-1])
        self.map_axis.set_ylim(-0.15, model.depths[-1])
        self.map_axis.set_aspect("equal", adjustable="box")
        self.map_axis.set_xlabel("Left / right of device centre (m)", color="white")
        self.map_axis.set_ylabel("Distance in front of speaker (m)", color="white")
        self.map_axis.set_title("Change evidence — not a room map", color="white")
        self.map_axis.legend(loc="upper right", fontsize=7, framealpha=0.75)

        self.trace_count = min(tracker.PERIOD, int(6 * capture.RATE / tracker.SPEED_OF_SOUND))
        self.trace_x = np.arange(self.trace_count) * tracker.SPEED_OF_SOUND / capture.RATE
        self.lines = []
        for ch, color in enumerate(COLORS):
            line, = self.signal_axis.plot([], [], color=color, linewidth=1, label=f"Mic {ch}")
            self.lines.append(line)
        self.signal_axis.set_xlim(0, self.trace_x[-1])
        self.signal_axis.set_ylim(0, 1.05)
        self.signal_axis.set_xlabel("Excess path from reference mic (m)", color="white")
        self.signal_axis.set_ylabel("Per-mic normalized envelope", color="white")
        self.signal_axis.set_title("Matched-filter response", color="white")
        self.signal_axis.grid(alpha=0.2)
        self.signal_axis.legend(ncol=3, fontsize=8, loc="upper right")

        self.info_axis.set_axis_off()
        self.info = self.info_axis.text(
            0.03, 0.92, "", transform=self.info_axis.transAxes, va="top",
            color="#dce4ec", fontsize=10, linespacing=1.5,
        )
        self.status = self.figure.text(0.07, 0.125, "Ready — press Start.",
                                       color="#87e0cb", fontsize=11)
        self.figure.text(
            0.07, 0.095,
            f"{band[0]/1000:g}–{band[1]/1000:g} kHz | {level:g}% digital peak | "
            f"{'runs until Stop' if duration is None else f'{duration} s session'}. "
            "Sound pressure is unknown; high frequencies may be inaudible.",
            color="#ffcc80", fontsize=9,
        )
        self.figure.text(
            0.07, 0.071,
            "Keep people and animals clear unless the acoustic output has been measured safe.",
            color="#ffcc80", fontsize=9,
        )
        self.start_button = Button(self.figure.add_axes([0.60, 0.02, 0.11, 0.045]), "Start")
        self.cal_button = Button(self.figure.add_axes([0.72, 0.02, 0.13, 0.045]), "Recalibrate")
        self.stop_button = Button(self.figure.add_axes([0.86, 0.02, 0.11, 0.045]), "Stop")
        self.start_button.on_clicked(self.start)
        self.cal_button.on_clicked(self.recalibrate)
        self.stop_button.on_clicked(self.stop)
        self.figure.canvas.mpl_connect("close_event", self._closed)
        self.timer = self.figure.canvas.new_timer(interval=100)
        self.timer.add_callback(self.poll)
        self.timer.start()
        self._set_info()

    def _set_info(self, frame=None):
        lines = [
            f"Device: {self.serial}",
            "Remove the moving target during calibration.",
            "Position estimates use supplied geometry and an",
            "assumed direct-path timing reference; they are",
            "not calibrated or ground truth. Hotter colors",
            "indicate stronger change, not a room map.",
        ]
        if frame is not None:
            lines.extend(("", f"Sync peak / median: {frame.sync_ratio:.1f}",
                          f"Channels supporting peak: {frame.support} / 6"))
            if frame.position is not None:
                depth, lateral = frame.position
                lines.append(f"Candidate: {depth:.2f} m forward, {lateral:+.2f} m right")
        self.info.set_text("\n".join(lines))

    def _clear_position(self, clear_history=False):
        self.marker.set_data([], [])
        if clear_history:
            self.history.clear()
            self.trail.set_data([], [])

    def start(self, _event=None):
        if self.worker is not None:
            return
        self.stop_event.clear()
        self.recalibrate_event.clear()
        self.stopping = False
        self.failed = False
        self._clear_position(clear_history=True)
        self.heat.set_data(np.zeros_like(self.heat.get_array()))
        self._set_info()
        self.status.set_text("Preparing device — playback has not started yet.")
        self.worker = threading.Thread(target=self._run, name="audiodevice-live", daemon=False)
        self.worker.start()
        self.figure.canvas.draw_idle()

    def _run(self):
        reported = False
        try:
            model = tracker.AudiodeviceTracker(*self.band)
            synchronized = False
            calibrated = False

            def on_status(text):
                self.messages.put(("status", text))

            def on_samples(samples):
                nonlocal synchronized, calibrated
                if self.recalibrate_event.is_set():
                    model.reset_calibration()
                    self.recalibrate_event.clear()
                    synchronized = False
                    calibrated = False
                for frame in model.push(samples):
                    if frame.state == "calibrating":
                        synchronized = True
                        calibrated |= frame.calibration == tracker.CALIBRATION_UPDATES
                    self.messages.put(("frame", frame))

            audio.stream(self.serial, *self.band, self.level, self.duration,
                         self.stop_event, on_status, on_samples)
        except InterruptedError:
            self.messages.put(("done", "Stopped; device cleanup finished."))
            reported = True
        except (capture.CaptureError, subprocess.SubprocessError, OSError, ValueError,
                RuntimeError) as exc:
            self.messages.put(("error", str(exc)))
            print(f"Live demo failed: {exc}", file=sys.stderr)
            self.messages.put(("done", ""))
            reported = True
        else:
            if not synchronized:
                status = "Session ended without a stable chirp reference — no position."
            elif not calibrated:
                status = "Session ended before background calibration completed."
            else:
                status = "Session finished; press Start to run again."
            self.messages.put(("done", status))
            reported = True
        finally:
            if not reported:
                self.messages.put(("error", "Unexpected worker failure; see the terminal."))
                self.messages.put(("done", ""))

    def recalibrate(self, _event=None):
        if self.worker is None or self.stopping:
            return
        self.recalibrate_event.set()
        self._clear_position(clear_history=True)
        self.status.set_text("Recalibrating — remove the target and keep the room still.")
        self.figure.canvas.draw_idle()

    def stop(self, _event=None):
        if self.worker is None:
            return
        self.stopping = True
        self.stop_event.set()
        self.status.set_text("Stopping playback and restoring the device...")
        self.figure.canvas.draw_idle()

    def _show_frame(self, frame):
        self._set_info(frame)
        if frame.envelopes is not None:
            for ch, line in enumerate(self.lines):
                envelope = frame.envelopes[:self.trace_count, ch]
                maximum = float(np.max(envelope))
                line.set_data(self.trace_x, envelope / maximum if maximum > 0 else envelope)
            self.signal_axis.set_title(
                "Matched-filter response" if frame.state == "calibrating"
                else "Change from calibrated background", color="white"
            )
        else:
            for line in self.lines:
                line.set_data([], [])
        if frame.scores is not None:
            self.heat.set_data(frame.scores)
            self.heat.set_clim(0, max(2.0, frame.threshold * 1.5))
        else:
            self.heat.set_data(np.zeros_like(self.heat.get_array()))
        if frame.position is None:
            self._clear_position(clear_history=frame.state in ("syncing", "calibrating"))
        else:
            depth, lateral = frame.position
            self.marker.set_color("#ffae42" if frame.state == "candidate" else "#40dcff")
            self.marker.set_data([lateral], [depth])
            self.history.append((lateral, depth))
            self.trail.set_data(*zip(*self.history))
        if frame.state == "syncing":
            self.status.set_text("Waiting for a stable chirp reference — no position.")
        elif frame.state == "calibrating":
            self.status.set_text(
                f"Calibrating {frame.calibration}/{tracker.CALIBRATION_UPDATES}: "
                "keep the target out and the room still."
            )
        elif frame.state == "no_target":
            self.status.set_text("No sufficiently supported change — no position.")
        elif frame.state == "candidate":
            self.status.set_text("Unconfirmed candidate — position accuracy is unverified.")
        else:
            self.status.set_text("Experimental candidate — position accuracy is unverified.")

    def poll(self):
        changed = False
        while True:
            try:
                kind, value = self.messages.get_nowait()
            except queue.Empty:
                break
            changed = True
            if kind == "frame":
                if not self.stopping:
                    self._show_frame(value)
            elif kind == "status":
                if not self.stopping:
                    self.status.set_text(value)
            elif kind == "error":
                self.failed = True
                self._clear_position()
                self.status.set_text(f"Error: {value}")
            elif kind == "done":
                self.worker = None
                self.stopping = False
                self._clear_position()
                if not self.failed:
                    self.status.set_text(value)
        if changed and not self.closed:
            self.figure.canvas.draw_idle()

    def _closed(self, _event):
        self.closed = True
        self.timer.stop()
        self.stop_event.set()

    def shutdown(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=30)
            if self.worker.is_alive():
                print("Waiting for Audiodevice cleanup to finish...", file=sys.stderr)
                self.worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="ADB serial; auto-detect one connected Audiodevice if omitted")
    parser.add_argument("--band", type=float, nargs=2, default=(18_000, 22_000),
                        metavar=("LOW_HZ", "HIGH_HZ"))
    parser.add_argument("--level", type=float, default=1,
                        help="digital peak percentage (default 1; maximum 20)")
    parser.add_argument("--seconds", type=int,
                        help="optional session length in seconds (default: run until Stop)")
    args = parser.parse_args()
    if not 0 < args.level <= 20:
        parser.error("--level must be greater than 0 and no more than 20")
    if args.seconds is not None and args.seconds < 5:
        parser.error("--seconds must be at least 5")
    try:
        capture.make_chirp(*args.band)
        if matplotlib.get_backend().lower() in {"agg", "pdf", "ps", "svg", "cairo", "template"}:
            raise capture.CaptureError("An interactive Matplotlib display is required")
        serial = args.serial or connected_serial()
        demo = LiveDemo(serial, args.band, args.level, args.seconds)
        previous_handler = signal.getsignal(signal.SIGTERM)
        def interrupt(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        try:
            plt.show()
        finally:
            demo.shutdown()
            signal.signal(signal.SIGTERM, previous_handler)
    except KeyboardInterrupt:
        print("Live demo interrupted.", file=sys.stderr)
        return 130
    except (capture.CaptureError, subprocess.SubprocessError, OSError, ValueError,
            ImportError) as exc:
        print(f"Unable to start live demo: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
