import struct
import threading
import unittest
from unittest.mock import Mock, patch

import audiodevice_lidar as lidar


def packet(first, last, distance=1234):
    data = struct.pack("<BBHH", 0x54, 0x2C, 3600, first)
    data += struct.pack("<HB", distance, 180) * 12
    data += struct.pack("<HH", last, 123)
    return data + bytes([lidar.crc8(data)])


class LD19Tests(unittest.TestCase):
    def test_fragmented_packets_wrap_angles_and_convert_millimetres(self):
        decoder = lidar.LD19Decoder()
        data = packet(35900, 100)
        for byte in data:
            self.assertEqual(decoder.push(bytes([byte])), [])
        scans = decoder.push(packet(100, 300) + packet(35900, 100))
        self.assertEqual(len(scans), 1)
        self.assertAlmostEqual(scans[0]["hz"], 10)
        self.assertTrue(all(p[1:] == [1.234, 180] for p in scans[0]["points"]))
        self.assertTrue(all(0 <= p[0] < 360 for p in scans[0]["points"]))
        self.assertLess(scans[0]["points"][0][0], 1)

    def test_corruption_resynchronizes_and_zero_ranges_are_omitted(self):
        decoder = lidar.LD19Decoder()
        damaged = bytearray(packet(100, 300))
        damaged[8] ^= 1
        scans = decoder.push(b"noise" + damaged + packet(35900, 100, 0)
                             + packet(100, 300) + packet(35900, 100, 0))
        self.assertEqual(len(scans), 1)
        self.assertEqual(scans[0]["bad_packets"], 1)
        self.assertEqual(len(scans[0]["points"]), 12)
        decoder.push(b"x" * 100000)
        self.assertLessEqual(len(decoder.buffer), 1)

    def test_serial_settings_restored_if_reader_cannot_start(self):
        with patch.object(lidar.capture, "shell", return_value=Mock(stdout="saved\n")) as shell:
            with patch.object(lidar.audio, "RemoteProcess", side_effect=OSError("reader failed")):
                with self.assertRaisesRegex(OSError, "reader failed"):
                    lidar.stream("serial", "/dev/ttyUSB0", True, threading.Event(), Mock())
        self.assertEqual(shell.call_args.args, (
            "serial", "su", "0", "stty", "-F", "/dev/ttyUSB0", "saved"))

    def test_invalid_port_never_runs_adb(self):
        with patch.object(lidar.capture, "shell") as shell:
            with self.assertRaises(ValueError):
                lidar.stream("serial", "/dev/ttyUSB0; echo bad", False, threading.Event(), Mock())
            shell.assert_not_called()


if __name__ == "__main__":
    unittest.main()
