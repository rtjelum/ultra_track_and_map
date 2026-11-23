import unittest

import numpy as np

import recording_sim as sim
import tracker as T


class MatchedFilterTests(unittest.TestCase):
    def setUp(self):
        self.mf = sim.MF(T.Chirp())

    def test_echo_gives_its_amplitude_at_its_delay(self):
        x = np.zeros((2000, 1))
        x[700:700 + self.mf.chirp.n, 0] = 0.3 * self.mf.chirp.burst
        z = np.abs(self.mf(x)[0])
        self.assertEqual(int(np.argmax(z)), 700)
        self.assertAlmostEqual(z.max(), 0.3, places=6)

    def test_clean_recovers_two_echoes(self):
        x = np.zeros((3000, 1))
        n = self.mf.chirp.n
        x[500:500 + n, 0] += 1e-3 * self.mf.chirp.burst
        x[1400:1400 + n, 0] -= 4e-4 * self.mf.chirp.burst
        z = self.mf(x)[0]
        taps, resid = sim.clean(z, self.mf, 0, 3000, 1e-6)
        strong = sorted(k for k, a in taps.items() if abs(a) > 1e-4)
        self.assertEqual(strong, [500, 1400])
        self.assertLess(np.abs(resid).max(), 1e-6)


class GeometryTests(unittest.TestCase):
    def test_segments_from_an_l_shaped_scan(self):
        wall = np.column_stack([np.full(60, 3.0), np.linspace(-1.5, 1.5, 60)])
        side = np.column_stack([np.linspace(2.95, 1.0, 40), np.full(40, 1.5)])
        segs = sim.extract_segments(np.vstack([wall, side]))
        lengths = sorted(float(np.linalg.norm(b - a)) for a, b in segs)
        self.assertEqual(len(lengths), 2)
        np.testing.assert_allclose(lengths, [1.95, 3.0], atol=0.06)

    def test_specular_path_matches_the_image_source(self):
        wall = (np.array([3.0, -2.0]), np.array([3.0, 2.0]))
        tx, mics = np.array([0.0, -0.2]), np.array([[0.0, 0.1], [0.0, 0.3]])
        echoes = sim.specular_echoes([wall], tx, mics, lam=0.017, order=1)
        self.assertEqual(len(echoes), 2)
        for m, path, w, nb, legs in echoes:
            image = np.array([6.0, -0.2])
            self.assertAlmostEqual(path, np.linalg.norm(image - mics[m]), places=9)
            self.assertEqual((nb, w), (1, 1.0))
            self.assertAlmostEqual(legs[1][0], 3.0)

    def test_short_segment_off_the_reflection_point_gives_no_echo(self):
        seg = (np.array([3.0, 1.0]), np.array([3.0, 1.2]))
        echoes = sim.specular_echoes([seg], np.zeros(2), np.array([[0.0, 0.1]]), lam=0.017, order=1)
        self.assertEqual(echoes, [])

    def test_lidar_angles_are_clockwise_towards_the_right(self):
        xy = sim.lidar_xy([[0.0, 2.0, 0], [90.0, 1.0, 0]])
        np.testing.assert_allclose(xy, [[2.0, 0.0], [0.0, 1.0]], atol=1e-12)

    def test_distance_to_path_legs(self):
        d = sim._seg_dist(np.array([1.0, 0.5]), np.array([[0.0, 0.0]]), np.array([[2.0, 0.0]]))
        self.assertAlmostEqual(d[0], 0.5)


if __name__ == "__main__":
    unittest.main()
