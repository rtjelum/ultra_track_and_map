import os
import unittest
from unittest.mock import patch

import numpy as np

import audiodevice_mapper as mapper
import audiodevice_tracking as tracking

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LiveMapperTests(unittest.TestCase):
    def setUp(self):
        self.live = mapper.LiveMapper()

    def test_states_map_to_the_gui_states(self):
        f = self.live._frame
        self.assertEqual(f(mapper.MapperFrame("locking", lock_quality=2.0)).state, "syncing")
        cal = f(mapper.MapperFrame("calibrating", lock_quality=1.0, cal_progress=1.0))
        self.assertEqual((cal.state, cal.calibration), ("calibrating", tracking.CALIBRATION_UPDATES))
        env = np.ones((mapper.WINDOW, 3))
        heat = np.zeros((len(self.live.depths), len(self.live.laterals)))
        found = f(mapper.MapperFrame("tracking", 1.0, 1.0, position=(2.0, -0.3), threshold=2.1,
                                     envelopes=env, heat=heat))
        self.assertEqual((found.state, found.position, found.support), ("tracking", (2.0, -0.3), 3))
        self.assertEqual(found.envelopes.shape, (mapper.WINDOW, 6))
        self.assertTrue(np.all(found.envelopes[:, [0, 2, 4]] == 1))
        self.assertFalse(np.any(found.envelopes[:, [1, 3, 5]]))
        self.assertIs(found.scores, heat)
        walled = f(mapper.MapperFrame("calibrating", 1.0, 1.0, facing_wall_m=4.3))
        self.assertEqual(walled.facing_wall_m, 4.3)
        lost = f(mapper.MapperFrame("searching", 1.0, 1.0, position=None, envelopes=env, heat=heat,
                                    facing_wall_m=4.3))
        self.assertEqual(lost.facing_wall_m, 4.3)
        skipped = f(mapper.MapperFrame("searching", 0.1, 1.0))      # sync check skipped the ping
        self.assertIs(skipped.scores, heat)
        self.assertTrue(np.all(skipped.envelopes[:, [0, 2, 4]] == 1))
        self.assertEqual((lost.state, lost.position), ("no_target", None))

    def test_notch_filters_mapper_input_across_chunks(self):
        live = mapper.LiveMapper(notch_hz=[21_000])
        t = np.arange(6000) / 48_000
        samples = np.tile(np.sin(2 * np.pi * 21_000 * t)[:, None], (1, 6))
        with patch.object(live.mapper, "push", return_value=[]) as push:
            live.push(samples[:1700])
            live.push(samples[1700:])
        filtered = np.concatenate([call.args[0] for call in push.call_args_list])
        self.assertLess(np.max(np.abs(filtered[4000:])), 0.01)
        with self.assertRaises(ValueError):
            live.push(np.full((10, 6), np.nan))

    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, "recordings", "walk1.flac")),
                         "needs recordings/walk1")
    def test_tracks_the_walker_in_walk1(self):
        import audiodevice_recording as recording
        rec = recording.load(os.path.join(ROOT, "recordings", "walk1"))
        states, positions = [], []
        for _, chunk in rec.chunks():
            for frame in self.live.push(chunk):
                states.append(frame.state)
                if frame.position is not None:
                    positions.append(frame.position)
        self.assertIn("calibrating", states)
        self.assertGreater(states.count("tracking"), 100)
        depth = np.array(positions)[:, 0]
        self.assertTrue(0.5 < depth.min() and depth.max() < 4.0)


if __name__ == "__main__":
    unittest.main()
