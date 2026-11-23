import os
import tempfile
import unittest

import numpy as np

import audiodevice_recording as recording


class RecordingTests(unittest.TestCase):
    def test_round_trip_is_lossless_and_keeps_chunk_and_scan_layout(self):
        rng = np.random.default_rng(1)
        samples = (rng.integers(-2 ** 22, 2 ** 22, (700, 6)) / recording.SCALE).astype(np.float32)
        chunks = [(10.0, samples[:300]), (10.05, samples[300:])]
        scans = [(10.01, {"points": [[1.5, 2.25, 120.0]], "hz": 10.0, "timestamp_ms": 7,
                          "bad_packets": 1}),
                 (10.11, {"points": [], "hz": 9.5, "timestamp_ms": 107, "bad_packets": 0})]
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "r")
            recording.save(stem, chunks, scans, track=[[0.0, 1.0, 0.5, 9]], meta={"band_hz": [1, 2]})
            rec = recording.load(stem)
        self.assertTrue(np.array_equal(rec.samples, samples))
        self.assertEqual([len(c) for _, c in rec.chunks()], [300, 400])
        self.assertEqual(rec.scans, scans)
        self.assertEqual(rec.meta, {"band_hz": [1, 2]})
        self.assertEqual(rec.track.tolist(), [[0.0, 1.0, 0.5, 9.0]])
        self.assertAlmostEqual(rec.sample_time(700), 0.05)
        self.assertAlmostEqual(rec.sample_time(500), 0.025)

    def test_rejects_samples_beyond_24_bit_full_scale(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            recording.save(os.path.join(tmp, "r"), [(0.0, np.ones((4, 6), np.float32))], [])

    def test_player_replays_chunks_and_scans_on_one_clock(self):
        import threading
        samples = np.zeros((900, 6), np.float32)
        chunks = [(5.0, samples[:300]), (5.1, samples[300:600]), (5.2, samples[600:])]
        scans = [(5.05, {"points": [[1.0, 2.0, 3.0]], "hz": 10.0, "timestamp_ms": 1,
                         "bad_packets": 0}),
                 (5.15, {"points": [], "hz": 10.0, "timestamp_ms": 2, "bad_packets": 0})]
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "r")
            recording.save(stem, chunks, scans, meta={"band_hz": [18000, 22000]})
            player = recording.Player(stem, speed=4.0)
        self.assertEqual(player.band, (18000, 22000))
        got, seen, status = [], [], []
        stop = threading.Event()
        lidar = threading.Thread(target=player.lidar, args=(None, None, None, stop, seen.append))
        lidar.start()
        player.audio(None, 0, 0, 0, None, stop, status.append, got.append)
        import time
        time.sleep(0.1)
        stop.set()
        lidar.join()
        self.assertEqual([len(c) for c in got], [300, 300, 300])
        self.assertEqual([s["timestamp_ms"] for s in seen], [1, 2])
        self.assertIn("Replaying", status[0])
        with self.assertRaises(InterruptedError):
            player.audio(None, 0, 0, 0, None, stop, status.append, got.append)


if __name__ == "__main__":
    unittest.main()
