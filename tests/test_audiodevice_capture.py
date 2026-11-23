import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import audiodevice_capture as capture


class AudiodeviceCaptureTests(unittest.TestCase):
    def test_auto_detects_only_connected_adb_device_without_product_filter(self):
        output = SimpleNamespace(
            stdout=("List of devices attached\n"
                    "device-1\tdevice product:unbranded model:target\n"),
        )
        with patch.object(capture.subprocess, "run", return_value=output):
            self.assertEqual(capture.connected_serial(), "device-1")

        output.stdout += "device-2\tdevice product:other model:phone\n"
        with patch.object(capture.subprocess, "run", return_value=output):
            with self.assertRaisesRegex(capture.CaptureError, "specify --serial"):
                capture.connected_serial()

    def test_playback_levels_are_capped_and_packed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chirp-1.wav"
            capture.make_playback(path, 12000, 16000, 1)
            with wave.open(str(path), "rb") as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()),
                                 (2, 4, 48000))
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").reshape(-1, 4)
        self.assertGreater(np.max(np.abs(pcm)), 0)
        self.assertLessEqual(np.max(np.abs(pcm)), round(32767 * .01))
        np.testing.assert_array_equal(pcm[:, 0], pcm[:, 3])
        np.testing.assert_array_equal(pcm[:4800], pcm[4800:9600])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chirp-20.wav"
            capture.make_playback(path, 18000, 22000, 20)
            with wave.open(str(path), "rb") as wav:
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        self.assertLessEqual(np.max(np.abs(pcm)), round(32767 * .20))
        self.assertGreater(np.max(np.abs(pcm)), 6000)

    def test_rejects_chirps_below_requested_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                capture.make_playback(Path(directory) / "chirp.wav", 9999, 16000)

    def test_caps_playback_level_at_twenty_percent(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                capture.make_playback(Path(directory) / "chirp.wav", 18000, 22000, 20.1)
            with self.assertRaises(ValueError):
                capture.make_playback(Path(directory) / "chirp.wav", 18000, 22000, 0.00001)

    def test_mixer_checks_numeric_and_boolean_values(self):
        result = SimpleNamespace(stderr="", stdout="")
        with patch.object(capture, "shell", return_value=result):
            with patch.object(capture, "mixer_values", return_value=["0"]):
                capture.mixer_set("serial", 1, "volume", ["0"])
            with patch.object(capture, "mixer_values", return_value=["Off", "On"]):
                capture.mixer_set("serial", 1, "route", ["0", "1"])
            with patch.object(capture, "mixer_values", return_value=["0"]):
                with self.assertRaises(capture.CaptureError):
                    capture.mixer_set("serial", 1, "volume", ["78"])

    def test_partial_mixer_setup_is_restored_on_failure(self):
        settings = {"mixer": {"init": [{"volume": 78}, {"mux": "DMIC1"}]}}
        values = {
            "MultiMedia5 Mixer TX_CDC_DMA_TX_3": ["Off", "Off"],
            "PRI_MI2S_RX Audio Mixer MultiMedia5": ["On", "Off"],
            "volume": ["0"],
            "mux": ["ZERO"],
        }
        calls = []

        def fake_shell(_serial, *args, **_kwargs):
            text = "0" if args[:2] == ("id", "-u") else "audiodevice"
            return SimpleNamespace(stdout=text, stderr="")

        def fake_set_many(_serial, _card, items):
            calls.append(list(items))
            if ("mux", ["DMIC1"]) in items:
                raise capture.CaptureError("forced setup failure")

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "mic.json"
            config.write_text(json.dumps(settings))
            output = Path(directory) / "output.wav"
            with (patch.object(capture, "MIC_CONFIG", config),
                  patch.object(capture, "shell", side_effect=fake_shell),
                  patch.object(capture, "run_adb"),
                  patch.object(capture, "locate_pcm", return_value=(1, 9)),
                  patch.object(capture, "mixer_values_many",
                               side_effect=lambda _s, _c, names: {n: values[n] for n in names}),
                  patch.object(capture, "mixer_set_many", side_effect=fake_set_many)):
                with self.assertRaisesRegex(capture.CaptureError, "forced setup failure"):
                    capture.capture("serial", output, 18000, 22000)
            self.assertFalse(output.exists())
        self.assertEqual(calls, [[("volume", ["78"]), ("mux", ["DMIC1"])],
                                 [("mux", ["ZERO"]), ("volume", ["0"])]])

    def test_batched_mixer_set_reports_each_failed_control(self):
        output = ("@@m 0\n@@m rc 0\n@@m 1\nError: invalid value\n@@m rc 22\n"
                  "@@m 2\n78\n@@m rc 0\n@@m 3\n0\n@@m rc 0\n")
        with (patch.object(capture.uuid, "uuid4", return_value=SimpleNamespace(hex="m")),
              patch.object(capture, "run_adb", return_value=SimpleNamespace(stdout=output))
              as adb):
            with self.assertRaisesRegex(capture.CaptureError, "Cannot set mixer mux") as error:
                capture.mixer_set_many("serial", 1, [("volume", ["78"]), ("mux", ["DMIC1"])])
        self.assertNotIn("volume", str(error.exception))
        self.assertEqual(adb.call_count, 1)


if __name__ == "__main__":
    unittest.main()
