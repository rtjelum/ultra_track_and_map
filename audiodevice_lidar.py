"""LD-19 packets and bounded-memory serial streaming from an ADB target."""

import os
import re
import select
import struct
import time

import audiodevice_capture as capture
import audiodevice_live_audio as audio


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ (0x4D if crc & 0x80 else 0)) & 0xFF
    return crc


class LD19Decoder:
    """Emit complete revolutions as [clockwise degrees, metres, confidence]."""

    def __init__(self):
        self.buffer = bytearray()
        self.points = []
        self.previous_angle = None
        self.synchronized = False
        self.bad_packets = 0

    def push(self, data: bytes) -> list[dict]:
        self.buffer.extend(data)
        scans = []
        while len(self.buffer) >= 47:
            start = self.buffer.find(b"\x54\x2c")
            if start < 0:
                self.buffer[:] = self.buffer[-1:]
                break
            del self.buffer[:start]
            if len(self.buffer) < 47:
                break
            packet = self.buffer[:47]
            if crc8(packet[:-1]) != packet[-1]:
                self.bad_packets += 1
                del self.buffer[0]
                continue
            del self.buffer[:47]
            speed, first = struct.unpack_from("<HH", packet, 2)
            last, timestamp = struct.unpack_from("<HH", packet, 42)
            span = (last - first) % 36000
            if first >= 36000 or last >= 36000 or not speed or span > speed * 12 / 4500 * 150:
                self.bad_packets += 1
                continue
            for i in range(12):
                angle = ((first + span * i / 11) / 100) % 360
                distance, confidence = struct.unpack_from("<HB", packet, 6 + i * 3)
                if self.previous_angle is not None and self.previous_angle > 340 and angle < 20:
                    if self.synchronized:
                        scans.append({"points": self.points, "hz": round(speed / 360, 2),
                                      "timestamp_ms": timestamp, "bad_packets": self.bad_packets})
                    self.points = []
                    self.synchronized = True
                self.previous_angle = angle
                if self.synchronized and distance:
                    self.points.append([round(angle, 3), distance / 1000, confidence])
                if len(self.points) > 10000:
                    raise capture.CaptureError("LD-19 revolution boundary missing")
        return scans


def stream(serial, port, root, stop_event, on_scan):
    if not re.fullmatch(r"/dev/tty(?:USB|ACM|HS|MSM)\d+", port):
        raise ValueError("LiDAR port must be a /dev/ttyUSB, ttyACM, ttyHS or ttyMSM device")
    prefix = ["su", "0"] if root else []
    saved = capture.shell(serial, *prefix, "stty", "-F", port, "-g",
                          capture_output=True).stdout.strip()
    if not saved:
        raise capture.CaptureError("Cannot save LiDAR serial settings")
    reader = None
    try:
        capture.shell(serial, *prefix, "stty", "-F", port, "230400", "raw", "-echo",
                      "cs8", "-parenb", "-cstopb", "-crtscts", "clocal",
                      "min", "1", "time", "0", capture_output=True)
        reader = audio.RemoteProcess(serial, ["cat", port], binary=True, root=root)
        decoder = LD19Decoder()
        last_scan = time.monotonic()
        while not stop_event.is_set():
            ready, _, _ = select.select([reader.process.stdout], [], [], 0.25)
            if ready:
                data = os.read(reader.process.stdout.fileno(), 4096)
                if not data:
                    raise capture.CaptureError(f"LiDAR serial reader ended: {reader.output()}")
                for scan in decoder.push(data):
                    last_scan = time.monotonic()
                    on_scan(scan)
            if time.monotonic() - last_scan > 3:
                raise capture.CaptureError(
                    f"No complete LD-19 scan for 3 seconds ({decoder.bad_packets} rejected packets)")
    finally:
        try:
            if reader is not None:
                reader.stop()
        finally:
            capture.shell(serial, *prefix, "stty", "-F", port, saved, capture_output=True)
