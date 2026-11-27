import unittest

import numpy as np

import audiodevice_capture as capture
import audiodevice_tracking as live
import tracking


def synthetic_audio(target=None, seed=42):
    rng = np.random.default_rng(seed)
    stride = live.PINGS_PER_UPDATE * live.PERIOD
    chirp = capture.make_chirp(18_000, 22_000)
    samples = rng.normal(0, 0.002, (10 * stride + len(chirp) - 1, 6)).astype(np.float32)
    direct = (np.linalg.norm(live.MIC_POSITIONS - live.TX_POSITION, axis=1)
              * capture.RATE / live.SPEED_OF_SOUND)
    wall = (tracking.bistatic_paths(np.array([1.5, -0.4]), live.MIC_POSITIONS,
                                    live.TX_POSITION)
            * capture.RATE / live.SPEED_OF_SOUND)
    target_paths = None
    if target is not None:
        target_paths = (tracking.bistatic_paths(np.asarray(target), live.MIC_POSITIONS,
                                                live.TX_POSITION)
                        * capture.RATE / live.SPEED_OF_SOUND)
    pulse_samples = np.arange(len(chirp))
    for start in range(1200, len(samples) - len(chirp) - 1000, live.PERIOD):
        for ch in range(6):
            echoes = [(direct[ch], 0.35), (wall[ch], 0.05)]
            if target_paths is not None and start >= 6 * stride:
                echoes.append((target_paths[ch], 0.08))
            for delay, amplitude in echoes:
                arrival = start + delay
                indices = np.arange(int(arrival), min(int(arrival) + len(chirp) + 2, len(samples)))
                pulse = np.interp(indices - arrival, pulse_samples, chirp, left=0, right=0)
                samples[indices, ch] += (amplitude * pulse).astype(np.float32)
    return samples


def process_chunks(tracker, samples):
    frames = []
    for start in range(0, len(samples), 6203):
        frames.extend(tracker.push(samples[start:start + 6203]))
    return frames


class AudiodeviceTrackingTests(unittest.TestCase):
    def test_silence_never_produces_a_position(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(0, 0.002, (7 * live.PINGS_PER_UPDATE * live.PERIOD, 6))
        frames = process_chunks(live.AudiodeviceTracker(), samples)
        self.assertTrue(frames)
        self.assertTrue(all(frame.state == "syncing" and frame.position is None for frame in frames))

    def test_static_background_does_not_create_a_confirmed_track(self):
        frames = process_chunks(live.AudiodeviceTracker(), synthetic_audio())
        self.assertIn("calibrating", [frame.state for frame in frames])
        self.assertNotIn("tracking", [frame.state for frame in frames])
        self.assertTrue(all(frame.position is None or frame.state == "candidate"
                            for frame in frames))

    def test_tracks_first_echo_update_after_calibration(self):
        target = np.array([0.85, 0.1])
        frames = process_chunks(live.AudiodeviceTracker(), synthetic_audio(target))
        self.assertIsNotNone(frames[6 * live.PINGS_PER_UPDATE].position)
        positions = [frame.position for frame in frames if frame.position is not None]
        self.assertTrue(positions, [frame.state for frame in frames])
        self.assertLess(np.linalg.norm(np.array(positions[-1]) - target), 0.15)

    def test_updates_position_immediately_after_a_large_target_move(self):
        stride = live.PINGS_PER_UPDATE * live.PERIOD
        initial = synthetic_audio([0.85, 0.1])
        target = np.array([1.35, -0.35])
        moved = synthetic_audio(target)
        samples = np.vstack((initial[:8 * stride], moved[8 * stride:]))
        frames = process_chunks(live.AudiodeviceTracker(), samples)
        n = live.PINGS_PER_UPDATE  # updates advance one ping; window k*n starts at k*stride
        self.assertIsNotNone(frames[7 * n].position)
        self.assertIsNotNone(frames[8 * n].position)
        self.assertGreater(np.linalg.norm(np.array(frames[8 * n].position)
                                          - np.array(frames[7 * n].position)), 0.35)
        self.assertLess(np.linalg.norm(np.array(frames[8 * n].position) - target),
                        np.linalg.norm(np.array(frames[7 * n].position) - target))

    def test_loss_of_chirps_clears_the_track(self):
        samples = synthetic_audio([0.85, 0.1])
        noise = np.random.default_rng(4).normal(
            0, 0.002, (3 * live.PINGS_PER_UPDATE * live.PERIOD, 6))
        frames = process_chunks(live.AudiodeviceTracker(), np.vstack((samples, noise)))
        self.assertIn("tracking", [frame.state for frame in frames])
        self.assertEqual(frames[-1].state, "syncing")
        self.assertIsNone(frames[-1].position)

    def test_updates_every_ping_while_folding_five(self):
        samples = synthetic_audio()
        frames = process_chunks(live.AudiodeviceTracker(), samples)
        required = live.PINGS_PER_UPDATE * live.PERIOD + len(capture.make_chirp(18_000, 22_000)) - 1
        self.assertEqual(len(frames), (len(samples) - required) // live.HOP + 1)
        calibrating = [i for i, f in enumerate(frames) if f.state == "calibrating"]
        # Calibration still spans CALIBRATION_UPDATES non-overlapping five-ping groups.
        self.assertEqual(len(calibrating), (live.CALIBRATION_UPDATES - 1) * live.PINGS_PER_UPDATE + 1)

    def test_notch_restores_sync_under_in_band_tone(self):
        import audiodevice_mapper as mapper

        samples = synthetic_audio()
        t = np.arange(len(samples)) / capture.RATE
        samples += (2.0 * np.sin(2 * np.pi * 21_000 * t))[:, None].astype(np.float32)
        plain = process_chunks(live.AudiodeviceTracker(), samples)
        self.assertNotIn("calibrating", [frame.state for frame in plain])
        notched = process_chunks(live.AudiodeviceTracker(notch_hz=[21_000]), samples)
        self.assertIn("calibrating", [frame.state for frame in notched])
        plain_mapper = process_chunks(mapper.LiveMapper(), samples)
        self.assertNotIn("calibrating", [frame.state for frame in plain_mapper])
        notched_mapper = process_chunks(mapper.LiveMapper(notch_hz=[21_000]), samples)
        self.assertIn("calibrating", [frame.state for frame in notched_mapper])

    def test_raw_mode_maps_static_echoes_without_calibration(self):
        tracker = live.AudiodeviceTracker(raw=True)
        frames = process_chunks(tracker, synthetic_audio())
        self.assertNotIn("calibrating", [frame.state for frame in frames])
        raw = [frame for frame in frames if frame.state == "raw"]
        self.assertTrue(raw)
        self.assertTrue(all(frame.position is None and frame.threshold is None for frame in raw))
        peak = np.unravel_index(int(np.argmax(raw[-1].scores)), raw[-1].scores.shape)
        found = np.array([tracker.depths[peak[0]], tracker.laterals[peak[1]]])
        self.assertLess(np.linalg.norm(found - [1.5, -0.4]), 0.15)

    def test_notch_configuration_and_chunk_continuity(self):
        self.assertEqual(live.notch_frequencies(None), [21_000])
        self.assertEqual(live.notch_frequencies([0]), [])
        self.assertEqual(live.notch_frequencies([0, 19_000, 20_000]), [19_000, 20_000])
        for invalid in (-1, 24_000, float("nan")):
            with self.assertRaises(ValueError):
                live.NotchFilter([invalid])
        samples = np.random.default_rng(42).normal(size=(2000, 6)).astype(np.float32)
        whole = live.NotchFilter([21_000]).push(samples)
        filt = live.NotchFilter([21_000])
        chunks = np.concatenate([filt.push(samples[:731]), filt.push(samples[731:])])
        np.testing.assert_array_equal(chunks, whole)

    def test_rejects_bad_sample_shapes(self):
        tracker = live.AudiodeviceTracker()
        with self.assertRaises(ValueError):
            tracker.push(np.zeros((10, 5)))
        with self.assertRaises(ValueError):
            tracker.push(np.full((10, 6), np.nan))


if __name__ == "__main__":
    unittest.main()
