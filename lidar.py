"""
LD06 / LD19 2D lidar (LDROBOT protocol) — reader and live scan viewer.

    python lidar.py                      # auto-detect /dev/cu.usbserial-*
    python lidar.py --port /dev/cu.usbserial-0001 --range 6

Viewer keys:  + / -  or scroll  zoom     space  pause     s  save scan (.npz)
              t  trail on/off            q  quit

Protocol: 230400 baud, 8N1, 47-byte packets
    0x54 0x2C | speed u16 (deg/s) | start angle u16 (0.01 deg)
    | 12 × (distance u16 mm, intensity u8) | end angle u16 | timestamp u16 (ms)
    | CRC-8 (poly 0x4D)
Angles increase clockwise seen from above; the viewer shows them
counter-clockwise-positive in x/y (x = 0°, y = 90° counter-clockwise),
i.e. the standard maths frame used by the rest of this project.
"""

import argparse
import glob
import struct
import threading
import time

import numpy as np

PKT_LEN = 47
N_PTS = 12
HEADER = b"\x54\x2c"

_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = ((_c << 1) ^ 0x4D) & 0xFF if _c & 0x80 else (_c << 1) & 0xFF
    _CRC_TABLE.append(_c)


def crc8(data):
    c = 0
    for b in data:
        c = _CRC_TABLE[c ^ b]
    return c


def parse_packet(p):
    """47 bytes → (speed deg/s, angles deg (12,), dist mm (12,), intensity (12,))."""
    speed, a0 = struct.unpack_from("<HH", p, 2)
    a1, _stamp = struct.unpack_from("<HH", p, 42)
    raw = np.frombuffer(p, np.uint8, 3 * N_PTS, 6).reshape(N_PTS, 3)
    dist = raw[:, 0].astype(np.uint16) | (raw[:, 1].astype(np.uint16) << 8)
    a0, a1 = a0 / 100.0, a1 / 100.0
    span = (a1 - a0) % 360.0
    ang = (a0 + span * np.arange(N_PTS) / (N_PTS - 1)) % 360.0
    return speed, ang, dist.astype(float), raw[:, 2].astype(float)


class Scan:
    """One full revolution. angle_deg is the lidar's own (clockwise) angle."""

    def __init__(self, angle_deg, dist_mm, intensity, t, speed_dps):
        self.angle_deg, self.dist_mm, self.intensity = angle_deg, dist_mm, intensity
        self.t, self.speed_dps = t, speed_dps

    def xy(self, min_mm=1.0):
        """Valid points in metres, maths frame (x forward, y left)."""
        ok = self.dist_mm >= min_mm
        a = -np.radians(self.angle_deg[ok])
        r = self.dist_mm[ok] / 1000.0
        return np.column_stack([r * np.cos(a), r * np.sin(a)]), self.intensity[ok]


class LD06:
    """Background reader. `latest()` returns the newest complete Scan (or None)."""

    def __init__(self, port=None, baud=230400):
        import serial                       # pyserial
        self.port = port or find_port()
        self.ser = serial.Serial(self.port, baud, timeout=0.1)
        self._lock = threading.Lock()
        self._scan = None
        self.n_scans = self.crc_errors = self.packets = 0
        self.scan_hz = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest(self):
        with self._lock:
            return self._scan

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _run(self):
        buf = bytearray()
        parts, last_ang, last_t, speed = [], None, None, 0.0
        self.ser.reset_input_buffer()
        while not self._stop.is_set():
            try:
                buf += self.ser.read(self.ser.in_waiting or 1)
            except Exception:
                if self._stop.is_set():
                    return
                raise
            while True:
                i = buf.find(HEADER)
                if i < 0:
                    del buf[:-1]
                    break
                if i:
                    del buf[:i]
                if len(buf) < PKT_LEN:
                    break
                p = bytes(buf[:PKT_LEN])
                if crc8(p[:-1]) != p[-1]:
                    self.crc_errors += 1
                    del buf[:1]                 # resync on the next header
                    continue
                del buf[:PKT_LEN]
                self.packets += 1
                speed, ang, dist, inten = parse_packet(p)
                # a new revolution starts when the start angle wraps around
                if last_ang is not None and ang[0] < last_ang - 180.0 and parts:
                    now = time.time()
                    a, d, s = (np.concatenate(x) for x in zip(*parts))
                    scan = Scan(a, d, s, now, speed)
                    with self._lock:
                        self._scan = scan
                        self.n_scans += 1
                        if last_t is not None:
                            dt = now - last_t
                            self.scan_hz = 0.8 * self.scan_hz + 0.2 / dt if self.scan_hz else 1 / dt
                    last_t, parts = now, []
                parts.append((ang, dist, inten))
                last_ang = ang[0]


def find_port():
    ports = sorted(glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.SLAB*")
                   + glob.glob("/dev/cu.wchusbserial*") + glob.glob("/dev/ttyUSB*"))
    if not ports:
        raise SystemExit("no USB-serial port found (is the lidar plugged in?)")
    return ports[0]


# ---------------------------------------------------------------- viewer

def view(port=None, max_range=6.0):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    lidar = LD06(port)
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title(f"Lidar — {lidar.port}")
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#111")
    ax.set_aspect("equal")
    ax.tick_params(colors="#aaa")
    for s in ax.spines.values():
        s.set_color("#444")
    ax.set_xlabel("x (m)  — lidar 0°", color="#aaa")
    ax.set_ylabel("y (m)", color="#aaa")

    rings = []
    state = dict(range=max_range, paused=False, trail=True)

    def draw_rings():
        for r in rings:
            r.remove()
        rings.clear()
        R = state["range"]
        step = 0.5 if R <= 3 else 1.0 if R <= 8 else 2.0
        for r in np.arange(step, R + 1e-9, step):
            c = plt.Circle((0, 0), r, fill=False, color="#333", lw=0.8)
            ax.add_patch(c)
            rings.append(c)
            rings.append(ax.text(r * 0.707, r * 0.707, f"{r:g} m", color="#555",
                                 fontsize=8))
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        fig.canvas.draw_idle()

    draw_rings()
    ax.plot([0, 0.25], [0, 0], color="#f80", lw=2)          # 0° heading marker
    ax.plot(0, 0, "o", color="#f80", ms=6)
    trail = ax.scatter([], [], s=2, c="#2a5", alpha=0.25, animated=True)
    pts = ax.scatter([], [], s=4, c=[], cmap="plasma", vmin=0, vmax=255, animated=True)
    info = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
                   color="#ddd", family="monospace", fontsize=9, animated=True)
    hist = []
    shown = dict(scan=None)

    def update(_):
        scan = shown["scan"] if state["paused"] else lidar.latest()
        if scan is not None:
            xy, inten = scan.xy()
            if scan is not shown["scan"] and not state["paused"]:
                hist.append(xy)
                del hist[:-10]
            shown["scan"] = scan
            pts.set_offsets(xy)
            pts.set_array(inten)
            trail.set_offsets(np.vstack(hist[:-1]) if state["trail"] and len(hist) > 1
                              else np.empty((0, 2)))
            r = np.hypot(*xy.T) if len(xy) else np.array([np.nan])
            info.set_text(
                f"{lidar.scan_hz:4.1f} Hz  {scan.speed_dps / 360:4.1f} rev/s  "
                f"{len(xy):3d}/{len(scan.dist_mm)} pts\n"
                f"range {np.nanmin(r):.2f}–{np.nanmax(r):.2f} m   "
                f"scans {lidar.n_scans}  crc err {lidar.crc_errors}"
                + ("\nPAUSED" if state["paused"] else ""))
        else:
            info.set_text(f"waiting for data on {lidar.port} …")
        return trail, pts, info

    def zoom(f):
        state["range"] = float(np.clip(state["range"] * f, 0.5, 12.0))
        draw_rings()

    def on_key(e):
        if e.key in ("+", "="):
            zoom(1 / 1.25)
        elif e.key in ("-", "_"):
            zoom(1.25)
        elif e.key == " ":
            state["paused"] = not state["paused"]
        elif e.key == "t":
            state["trail"] = not state["trail"]
        elif e.key == "s" and shown["scan"] is not None:
            sc = shown["scan"]
            fn = time.strftime("lidar_scan_%Y%m%d_%H%M%S.npz")
            np.savez(fn, angle_deg=sc.angle_deg, dist_mm=sc.dist_mm,
                     intensity=sc.intensity, t=sc.t)
            print("saved", fn)

    def on_scroll(e):
        zoom(1 / 1.15 if e.button == "up" else 1.15)

    plt.rcParams["keymap.save"] = []       # free 's' for our own save
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.colorbar(pts, ax=ax, fraction=0.035, pad=0.02).set_label("intensity", color="#aaa")
    fig.tight_layout()
    anim = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
    try:
        plt.show()
    finally:
        del anim
        lidar.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (default: first /dev/cu.usbserial*)")
    ap.add_argument("--range", type=float, default=6.0, help="initial view radius, m")
    a = ap.parse_args()
    view(a.port, a.range)
