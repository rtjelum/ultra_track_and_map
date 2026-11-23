"""Record and replay synchronised Audiodevice ultrasound and LD-19 LiDAR data.

A recording ``STEM`` is two files:

* ``STEM.flac``: six-channel 48 kHz capture, lossless 24-bit PCM.
* ``STEM.npz``: host ``time.monotonic()`` and length of every capture chunk,
  every LiDAR scan (time, points ``[angle_deg, range_m, intensity]``, rate,
  device timestamp, bad packets), an optional ground-truth walker ``track``
  with rows ``(t, depth_m, right_m, points)`` relative to the first chunk,
  and JSON metadata (band, level, notes).

    python audiodevice_recording.py record recordings/walk2 --seconds 30 --lidar-port /dev/ttyUSB0 --lidar-root
    python audiodevice_recording.py replay recordings/walk1
"""
import argparse
import json
import threading
import time
from dataclasses import dataclass, field

import numpy as np

RATE = 48_000
SCALE = 2 ** 23


@dataclass
class Recording:
    samples: np.ndarray
    chunk_times: np.ndarray
    chunk_lengths: np.ndarray
    scans: list
    track: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def chunks(self):
        """Yield ``(host_time, samples)`` exactly as they were captured."""
        starts = np.concatenate([[0], np.cumsum(self.chunk_lengths)])
        for t, a, b in zip(self.chunk_times, starts[:-1], starts[1:]):
            yield float(t), self.samples[a:b]

    def sample_time(self, index):
        """Host time in seconds since the first chunk for a stream sample index.

        Each chunk time is its arrival, so it is assigned to the chunk's last sample.
        """
        ends = np.cumsum(self.chunk_lengths)
        return np.interp(index, ends, self.chunk_times - self.chunk_times[0])


def save(stem, chunks, scans, *, track=None, meta=None):
    import soundfile

    samples = np.concatenate([c for _, c in chunks]) if chunks else np.zeros((0, 6), np.float32)
    pcm = np.round(np.asarray(samples, np.float64) * SCALE)
    if np.any(np.abs(pcm) >= SCALE):
        raise ValueError("Samples exceed 24-bit full scale")
    soundfile.write(f"{stem}.flac", pcm.astype(np.int32) << 8, RATE,
                    subtype="PCM_24", compression_level=1.0)
    points = [np.asarray(s["points"], np.float32).reshape(-1, 3) for _, s in scans]
    np.savez_compressed(
        f"{stem}.npz",
        chunk_times=np.array([t for t, _ in chunks], np.float64),
        chunk_lengths=np.array([len(c) for _, c in chunks], np.int64),
        scan_times=np.array([t for t, _ in scans], np.float64),
        scan_counts=np.array([len(p) for p in points], np.int64),
        scan_points=np.concatenate(points) if points else np.zeros((0, 3), np.float32),
        scan_hz=np.array([s.get("hz", np.nan) for _, s in scans], np.float64),
        scan_timestamp_ms=np.array([s.get("timestamp_ms", -1) for _, s in scans], np.int64),
        scan_bad_packets=np.array([s.get("bad_packets", 0) for _, s in scans], np.int64),
        track=np.zeros((0, 4)) if track is None else np.asarray(track, np.float64),
        meta=json.dumps(meta or {}),
    )


def load(stem):
    import soundfile

    pcm, rate = soundfile.read(f"{stem}.flac", dtype="int32", always_2d=True)
    if rate != RATE:
        raise ValueError(f"Expected {RATE} Hz, got {rate}")
    with np.load(f"{stem}.npz") as d:
        offsets = np.concatenate([[0], np.cumsum(d["scan_counts"])])
        scans = [(float(t), {"points": d["scan_points"][a:b].tolist(), "hz": float(hz),
                             "timestamp_ms": int(ts), "bad_packets": int(bad)})
                 for t, a, b, hz, ts, bad in zip(d["scan_times"], offsets[:-1], offsets[1:],
                                                 d["scan_hz"], d["scan_timestamp_ms"],
                                                 d["scan_bad_packets"])]
        track = d["track"] if len(d["track"]) else None
        rec = Recording((pcm >> 8).astype(np.float32) / SCALE, d["chunk_times"],
                        d["chunk_lengths"], scans, track, json.loads(str(d["meta"])))
    if rec.chunk_lengths.sum() != len(rec.samples):
        raise ValueError("Chunk lengths do not match the audio length")
    return rec


def record(stem, seconds, serial, low, high, level, lidar_port=None, lidar_root=False, note=""):
    import audiodevice_capture as capture
    import audiodevice_lidar as lidar
    import audiodevice_live_audio as audio

    serial = serial or capture.connected_serial()
    chunks, scans = [], []
    stop, lidar_stop = threading.Event(), threading.Event()
    reader = None
    if lidar_port:
        reader = threading.Thread(target=lidar.stream, daemon=True, args=(
            serial, lidar_port, lidar_root, lidar_stop,
            lambda s: scans.append((time.monotonic(), s))))
        reader.start()
    try:
        audio.stream(serial, low, high, level, seconds, stop, print,
                     lambda x: chunks.append((time.monotonic(), np.array(x, np.float32))))
    finally:
        lidar_stop.set()
        if reader:
            reader.join(timeout=10)
    save(stem, chunks, scans, meta={"band_hz": [low, high], "level_percent": level,
                                     "lidar_port": lidar_port, "note": note})
    print(f"{stem}: {sum(len(c) for _, c in chunks) / RATE:.1f} s audio, {len(scans)} scans")


class Player:
    """Plays a recording in real time (× ``speed``) in place of the device.

    ``audio`` has the signature of ``audiodevice_live_audio.stream`` and
    ``lidar`` that of ``audiodevice_lidar.stream``, so the GUIs can use them
    unchanged. Both follow one clock, which restarts when the audio starts, so
    LiDAR scans stay aligned with the audio as they were recorded."""

    def __init__(self, stem, speed=1.0):
        self.stem = stem
        self.rec = load(stem)
        self.speed = float(speed)
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        self.band = tuple(self.rec.meta.get("band_hz", (18_000, 22_000)))
        self.seconds = len(self.rec.samples) / RATE
        self._t0 = self.rec.chunk_times[0] if len(self.rec.chunk_times) else 0.0
        self._scan_times = np.array([t for t, _ in self.rec.scans]) - self._t0
        self._origin = None
        self._lock = threading.Lock()

    def now(self):
        """Recording time (s since the first chunk) of the playback clock."""
        with self._lock:
            if self._origin is None:
                self._origin = time.monotonic()
            return (time.monotonic() - self._origin) * self.speed

    def _wait_until(self, t, stop_event):
        while True:
            ahead = t - self.now()
            if ahead <= 0:
                return
            if stop_event.wait(min(ahead / self.speed, 0.05)):
                raise InterruptedError

    def audio(self, _serial, _low, _high, _level, duration, stop_event, on_status, on_samples):
        with self._lock:
            self._origin = None
        on_status(f"Replaying {self.stem} ({self.seconds:.0f} s"
                  f"{'' if self.speed == 1 else f' at ×{self.speed:g}'}) — no audio is played.")
        for t, chunk in self.rec.chunks():
            due = t - self._t0
            if duration is not None and due > duration:
                break
            self._wait_until(due, stop_event)
            on_samples(chunk)

    def lidar(self, _serial, _port, _root, stop_event, on_scan):
        last = -1
        while not stop_event.wait(0.02):
            i = int(np.searchsorted(self._scan_times, self.now(), side="right")) - 1
            if i >= 0 and i != last and i < len(self.rec.scans):
                on_scan(self.rec.scans[i][1])
                last = i


def angle_offset(rec):
    """The mapper's angle calibration is for the real device; simulated audio
    (recording_sim.py) has no such bias."""
    return 0.0 if rec.meta.get("simulated_from") else None


def evaluate(rec, lidar_delay=0.1, **mapper_args):
    """Run :class:`audiodevice_mapper.AudiodeviceMapper` on a loaded recording and
    compare with the LiDAR track. Returns a dict of metrics plus per-ping ``rows``
    ``(t, depth, right)`` (NaN where nothing is tracked)."""
    import audiodevice_mapper

    low, high = rec.meta.get("band_hz", (18_000, 22_000))
    mapper_args.setdefault("angle_offset_deg", angle_offset(rec))
    mapper = audiodevice_mapper.AudiodeviceMapper(low, high, **mapper_args)
    rows, start, pings, frame = [], time.perf_counter(), 0, None
    for _, chunk in rec.chunks():
        for frame in mapper.push(chunk):
            pings += 1
            if frame.state in ("locking", "calibrating"):
                continue
            rows.append((rec.sample_time(frame.sample_index), *(frame.position or (np.nan, np.nan))))
    out = dict(seconds=len(rec.samples) / RATE, pings=pings,
               ms_per_ping=1e3 * (time.perf_counter() - start) / max(pings, 1),
               facing_wall_m=frame.facing_wall_m if frame else None,
               rows=np.array(rows).reshape(-1, 3))
    if not rows or rec.track is None:
        return out
    t, depth, right = out["rows"].T
    track = rec.track[~np.isnan(rec.track[:, 1])]
    truth = [np.interp(t + lidar_delay, track[:, 0], track[:, k], left=np.nan, right=np.nan)
             for k in (1, 2)]
    present, found = ~np.isnan(truth[0]), ~np.isnan(depth)
    both = present & found
    range_err = np.abs(np.hypot(depth, right) - np.hypot(*truth))[both]
    lateral_err = np.abs(right - truth[1])[both]
    out.update(present=int(present.sum()), absent=int((~present).sum()),
               tracked=float(found[present].mean()) if present.any() else np.nan,
               false_tracks=float(found[~present].mean()) if (~present).any() else np.nan,
               range_err=range_err, lateral_err=lateral_err)
    return out


def replay(stem, lidar_delay=0.1):
    """Run :class:`audiodevice_mapper.AudiodeviceMapper` and compare with the LiDAR track."""
    m = evaluate(load(stem), lidar_delay)
    print(f"{m['seconds']:.1f} s, {m['pings']} pings, {m['ms_per_ping']:.1f} ms/ping, "
          f"facing wall {m['facing_wall_m']}")
    if "present" not in m:
        return
    re, le = m["range_err"], m["lateral_err"]
    if m["present"]:
        print(f"target present {m['present']} pings: tracked {100 * m['tracked']:.0f}%, "
              f"range error median {np.median(re):.2f} m p90 {np.percentile(re, 90):.2f} m, "
              f"lateral error median {np.median(le):.2f} m p90 {np.percentile(le, 90):.2f} m")
    if m["absent"]:
        print(f"target absent {m['absent']} pings: false tracks {100 * m['false_tracks']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="Record ultrasound and optional LiDAR")
    rec.add_argument("stem", help="Output path without extension")
    rec.add_argument("--seconds", type=int, required=True)
    rec.add_argument("--serial", help="ADB serial; auto-detect one connected Audiodevice if omitted")
    rec.add_argument("--band", type=float, nargs=2, default=(18_000, 22_000), metavar=("LOW", "HIGH"))
    rec.add_argument("--level", type=float, default=20)
    rec.add_argument("--lidar-port", help="LD-19 serial port on target, e.g. /dev/ttyUSB0")
    rec.add_argument("--lidar-root", action="store_true")
    rec.add_argument("--note", default="")
    rep = sub.add_parser("replay", help="Run the mapper on a recording")
    rep.add_argument("stem")
    args = parser.parse_args()
    if args.command == "record":
        record(args.stem, args.seconds, args.serial, *args.band, args.level,
               args.lidar_port, args.lidar_root, args.note)
    else:
        replay(args.stem)


if __name__ == "__main__":
    main()
