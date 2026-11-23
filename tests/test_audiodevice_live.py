import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import audiodevice_capture as capture
import audiodevice_live as gui
import audiodevice_live_audio as audio
import audiodevice_tracking as tracking


class AudiodeviceLiveTests(unittest.TestCase):
    def test_gui_clears_a_position_when_evidence_is_lost(self):
        app = gui.LiveDemo("serial", (18000, 22000), 1, 10)
        try:
            scores = np.ones_like(app.heat.get_array())
            envelopes = np.ones((tracking.PERIOD, 6))
            candidate = tracking.TrackingFrame("candidate", 9.0, 4, envelopes, scores,
                                               1.5, (0.85, 0.1), 4)
            app._show_frame(candidate)
            self.assertEqual(list(app.marker.get_xdata()), [0.1])
            self.assertIn("Unconfirmed candidate", app.status.get_text())
            tracked = tracking.TrackingFrame("tracking", 9.0, 4, envelopes, scores,
                                             1.5, (0.85, 0.1), 4)
            app._show_frame(tracked)
            self.assertEqual(list(app.marker.get_xdata()), [0.1])
            app._show_frame(tracking.TrackingFrame("syncing", 0.0, 0))
            self.assertEqual(len(app.marker.get_xdata()), 0)
            self.assertEqual(len(app.lines[0].get_xdata()), 0)
            self.assertFalse(np.any(app.heat.get_array()))
            self.assertIn("no position", app.status.get_text())
        finally:
            plt.close(app.figure)
            app.shutdown()

    def test_stop_signals_the_worker_and_finishes_cleanup(self):
        entered = threading.Event()

        def fake_stream(_serial, _low, _high, _level, _duration, stop, status, _samples):
            status("Streaming")
            entered.set()
            if not stop.wait(3):
                raise RuntimeError("Stop was not delivered")
            raise InterruptedError("Stopped")

        app = gui.LiveDemo("serial", (18000, 22000), 1, 10)
        try:
            with patch.object(audio, "stream", side_effect=fake_stream):
                app.start()
                self.assertTrue(entered.wait(3))
                app.stop()
                app.worker.join(timeout=3)
                self.assertFalse(app.worker.is_alive())
                app.poll()
                self.assertIsNone(app.worker)
                self.assertIn("cleanup finished", app.status.get_text())
        finally:
            plt.close(app.figure)
            app.shutdown()

    def test_completed_session_reports_missing_sync(self):
        def fake_stream(_serial, _low, _high, _level, _duration, _stop, _status, samples):
            samples(np.zeros((tracking.PINGS_PER_UPDATE * tracking.PERIOD + 240, 6)))

        app = gui.LiveDemo("serial", (18000, 22000), 1, 5)
        try:
            with patch.object(audio, "stream", side_effect=fake_stream):
                app.start()
                app.worker.join(timeout=3)
                self.assertFalse(app.worker.is_alive())
                app.poll()
                self.assertIn("without a stable chirp reference", app.status.get_text())
                self.assertEqual(len(app.marker.get_xdata()), 0)
        finally:
            plt.close(app.figure)
            app.shutdown()

    def test_completed_session_reports_incomplete_calibration(self):
        def fake_stream(_serial, _low, _high, _level, _duration, _stop, _status, samples):
            samples(np.zeros((1, 6)))

        model = Mock()
        model.push.return_value = [tracking.TrackingFrame("calibrating", 6.0, 1)]
        app = gui.LiveDemo("serial", (18000, 22000), 1, 5)
        try:
            with (patch.object(tracking, "AudiodeviceTracker", return_value=model),
                  patch.object(audio, "stream", side_effect=fake_stream)):
                app.start()
                app.worker.join(timeout=3)
                self.assertFalse(app.worker.is_alive())
                app.poll()
                self.assertIn("before background calibration completed", app.status.get_text())
        finally:
            plt.close(app.figure)
            app.shutdown()

    def test_cancel_stops_each_device_process_and_restores_route(self):
        events = []
        stop = threading.Event()

        @contextmanager
        def route(_serial, _stop):
            events.append("route opened")
            try:
                yield 1, 9
            finally:
                events.append("route restored")

        class FakeProcess:
            def __init__(self, _serial, args, **_kwargs):
                self.name = args[0]
                self.process = SimpleNamespace(poll=lambda: None)

            def stop(self):
                events.append(f"stopped {self.name}")

        def fake_shell(_serial, *args, **_kwargs):
            if args[0] == "stat":
                return SimpleNamespace(stdout="4096", stderr="")
            if args[0] == "rm":
                events.append("temporary files removed")
            return SimpleNamespace(stdout="", stderr="")

        def on_status(text):
            if text.startswith("Chirps active"):
                stop.set()

        with (patch.object(capture, "microphone_route", side_effect=route),
              patch.object(capture, "run_adb"),
              patch.object(capture, "shell", side_effect=fake_shell),
              patch.object(audio, "RemoteProcess", FakeProcess),
              patch.object(audio.time, "sleep")):
            with self.assertRaises(InterruptedError):
                audio.stream("serial", 18000, 22000, 1, 5, stop, on_status, lambda _s: None)
        self.assertEqual(events, ["route opened", "stopped tinyplay", "stopped sh",
                                  "stopped cat", "stopped tinycap", "temporary files removed",
                                  "route restored"])

    def test_refuses_to_signal_an_unrelated_remote_process(self):
        remote = audio.RemoteProcess.__new__(audio.RemoteProcess)
        remote.serial = "serial"
        remote.root = False
        remote.arguments = ["tinyplay", "/data/local/tmp/test.wav"]
        remote.pid = 1234
        remote.process = Mock()
        result = SimpleNamespace(returncode=0, stdout=b"other\0/data/local/tmp/test.wav\0",
                                 stderr=b"")
        with (patch.object(audio.subprocess, "run", return_value=result),
              patch.object(capture, "shell") as shell):
            with self.assertRaisesRegex(capture.CaptureError, "unexpected process"):
                remote.stop()
            shell.assert_not_called()


if __name__ == "__main__":
    unittest.main()
