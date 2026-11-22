#!/usr/bin/env python3
"""
Room simulator + live visualizer for the ultrasonic tracker.

Simulates a rectangular room (direct sound, wall and corner echoes, a moving
object with Doppler, noise) for any mic-array layout placed anywhere in the
room, and runs the tracker (tracker.py) and wall mapping (wallmap.py) on the
simulated recordings — the same code that would run on hardware.

Array and placement:
  --array NAME|FILE    layout (bar6, line4, tri3, ring8, or a JSON file; see arrays.py)
  --pose X Y DEG       where the array's mounting point is and which way it
                       faces (0° = +x). If the mounting point is on a wall and
                       the array faces into the room, that wall is treated as
                       the known mounting wall; otherwise it is free-standing.

Panels:
  Left   — room top-down: true pos (yellow ●), Kalman estimate (cyan ◆),
           single-ping fix (orange ×), velocity (green), wall map (pink)
  Top-R  — per-mic residual echo level
  Bot-R  — trajectory + speed

Usage:
    python simulator.py                              # live visualizer
    python simulator.py --eval 300                   # headless accuracy report
    python simulator.py --array ring8 --pose 1.5 1.0 30 --eval 300
    python simulator.py --array line4 --pose 1.5 0 90  # on the bottom wall
    python simulator.py --room 4.5 3.2
    python simulator.py --latency 23.7 --jitter 5    # unknown playback/record latency
"""

import argparse

import arrays

_ap = argparse.ArgumentParser(description="Ultrasonic tracker simulator")
_ap.add_argument("--eval", type=int, metavar="N", default=0,
                 help="run N frames headless and print accuracy statistics")
_ap.add_argument("--seed", type=int, default=None, help="random seed")
_ap.add_argument("--fs", type=int, default=48_000, help="sample rate in Hz (default 48000)")
_ap.add_argument("--band", type=float, nargs=2, metavar=("F0", "F1"),
                 default=(18_000, 22_000), help="chirp start/end frequency in Hz")
_ap.add_argument("--room", type=float, nargs=2, metavar=("W", "D"),
                 default=(3.0, 2.0), help="room size in metres (default 3 × 2)")
_ap.add_argument("--array", default="bar6",
                 help=f"mic layout: {', '.join(arrays.PRESETS)} or a JSON file (default bar6)")
_ap.add_argument("--pose", type=float, nargs=3, metavar=("X", "Y", "DEG"), default=None,
                 help="array mounting point (m) and facing angle (deg, 0 = +x); "
                      "default: centre of the left wall, facing into the room")
_ap.add_argument("--speaker", choices=("auto", "cardioid", "omni"), default="auto",
                 help="speaker directivity (auto: cardioid if wall-mounted, else omni)")
_ap.add_argument("--latency", type=float, default=0.0, metavar="MS",
                 help="simulated playback→recording latency in ms (default 0)")
_ap.add_argument("--jitter", type=float, default=0.0, metavar="MS",
                 help="extra random latency per ping, uniform 0..MS (default 0)")
_ap.add_argument("--no-sync", action="store_true",
                 help="don't measure/compensate the latency (assume it is zero)")
_ap.add_argument("--object-multipath", action="store_true",
                 help="EXPERIMENTAL: also simulate object↔wall bounces and "
                      "track with the wall-map multipath model (not yet robust)")
ARGS = _ap.parse_args()

import numpy as np
import matplotlib
if ARGS.eval:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import deque

import tracker as T
import wallmap

if ARGS.seed is not None:
    np.random.seed(ARGS.seed)

# ─────────────────────────────── constants ────────────────────────────────────
C         = T.C
NOISE_STD = 0.004       # fixed noise floor (mic + ADC), same units as echo amplitude
DIRECT_GAIN = 0.05      # speaker → mic leakage, amplitude = DIRECT_GAIN / distance
WALL_GAIN   = 0.30      # wall echo amplitude = WALL_GAIN · WALL_REFL^order / path
WALL_REFL   = 0.7       # pressure reflection coefficient per bounce
OBJ_GAIN    = 0.65      # object echo amplitude = OBJ_GAIN / path²
TRAIL_LEN      = 80
ANIM_INTERVAL  = 100    # ms per frame
FRAME_DT       = ANIM_INTERVAL / 1000.0

try:
    CHIRP = T.Chirp(fs=ARGS.fs, f0=ARGS.band[0], f1=ARGS.band[1])
except ValueError as e:
    _ap.error(str(e))
FS = CHIRP.fs

# ─────────────────────────────── room + array ─────────────────────────────────
ROOM_W, ROOM_D = map(float, ARGS.room)
TRUE_WALLS = [                                        # normals point into the room
    wallmap.Wall(np.array([ 1.0, 0.0]), 0.0),         # left
    wallmap.Wall(np.array([-1.0, 0.0]), -ROOM_W),     # right
    wallmap.Wall(np.array([ 0.0, 1.0]), 0.0),         # bottom
    wallmap.Wall(np.array([ 0.0,-1.0]), -ROOM_D),     # top
]
WALL_NAMES = ["left", "right", "bottom", "top"]

try:
    LAYOUT = arrays.load(ARGS.array)
except ValueError as e:
    _ap.error(str(e))
POSE = tuple(ARGS.pose) if ARGS.pose else (0.0, ROOM_D / 2, 0.0)
MIC_POS, TX_POS, FACING = arrays.place(LAYOUT, *POSE)
N_MICS = len(MIC_POS)
for p in (*MIC_POS, TX_POS):
    if not all(w.signed_distance(p) > 0 for w in TRUE_WALLS):
        _ap.error(f"array at pose {POSE} sticks out of the {ROOM_W}×{ROOM_D} m room")

# mounted = mounting point on a wall and facing straight into the room
MOUNT_IDX = next((i for i, w in enumerate(TRUE_WALLS)
                  if abs(w.signed_distance(np.array(POSE[:2]))) < 0.02
                  and w.normal @ FACING > np.cos(np.radians(5))), None)
SPEAKER = (ARGS.speaker if ARGS.speaker != "auto"
           else "cardioid" if MOUNT_IDX is not None else "omni")


def tx_gain(direction):
    """Speaker directivity. Cardioid: 1 on axis, 0.5 sideways, 0 backwards."""
    if SPEAKER == "omni":
        return 1.0
    d = np.asarray(direction, float)
    return 0.5 * (1.0 + float(d @ FACING) / max(np.linalg.norm(d), 1e-9))


# ─────────────────────────── acoustic simulation ──────────────────────────────
# reception window: long enough for the corner echoes (wall mapping) + burst
SIM_SAMPLES = int(FS * (2 * np.hypot(ROOM_W, ROOM_D) / C + CHIRP.duration)) + 500


def _add_chirp(out, ch, delay_s, amp, scale=1.0):
    """Add one echo arriving at an exact (fractional) delay, frequency-scaled
    by `scale` (Doppler). Evaluated analytically, not rounded to samples."""
    n0 = int(np.floor(delay_s * FS))
    n = np.arange(n0, n0 + CHIRP.n + 2)
    n = n[(n >= 0) & (n < out.shape[0])]
    t = n / FS - delay_s
    ok = (t >= 0) & (t < CHIRP.duration)
    t, n = t[ok], n[ok]
    phase = 2 * np.pi * (CHIRP.f0 * scale * t + 0.5 * (CHIRP.bw / CHIRP.duration) * scale * t**2)
    out[n, ch] += (np.cos(phase) * CHIRP.window(t) * amp).astype(np.float32)


def _image_sources(src, max_order=2):
    """Image sources up to max_order (exact for a rectangular room):
    [(position, order, L)] — L maps an arrival direction back to the emission
    direction at the real source (for the speaker directivity)."""
    found = {}
    frontier = [(src, -1, np.eye(2))]
    for order in range(1, max_order + 1):
        nxt = []
        for p, last, L in frontier:
            for wi, w in enumerate(TRUE_WALLS):
                if wi == last:
                    continue
                q = wallmap.reflect(p, w)
                Ln = L @ (np.eye(2) - 2 * np.outer(w.normal, w.normal))
                key = tuple(np.round(q, 6))
                if key not in found:
                    found[key] = (q, order, Ln)
                    nxt.append((q, wi, Ln))
        frontier = nxt
    return list(found.values())


def _static_signal():
    """Direct speaker→mic sound + wall/corner echoes (identical every ping)."""
    out = np.zeros((SIM_SAMPLES, N_MICS), dtype=np.float32)
    for i, mp in enumerate(MIC_POS):
        d = float(np.linalg.norm(mp - TX_POS))
        _add_chirp(out, i, d / C, tx_gain(mp - TX_POS) * DIRECT_GAIN / max(d, 0.02))
    for q, order, L in _image_sources(TX_POS):
        for i, mp in enumerate(MIC_POS):
            d = float(np.linalg.norm(mp - q))
            _add_chirp(out, i, d / C, tx_gain(L @ (mp - q)) * WALL_GAIN * WALL_REFL**order / d)
    return out


STATIC_SIGNAL = _static_signal()


# playback→recording latency: the recording starts this long before the
# chirp leaves the speaker (fixed part + random jitter per ping)
MAX_LATENCY_S = max(0.05, (ARGS.latency + ARGS.jitter) / 1e3 + 0.005)
REC_SAMPLES = SIM_SAMPLES + (0 if ARGS.no_sync else int(MAX_LATENCY_S * FS))
if ARGS.latency < 0 or ARGS.jitter < 0:
    _ap.error("--latency and --jitter must be >= 0")


def _delayed(sig, delay):
    """sig delayed by `delay` (fractional) samples into a REC_SAMPLES buffer."""
    n = 1 << (REC_SAMPLES + CHIRP.n).bit_length()
    X = np.fft.rfft(sig, n=n, axis=0)
    X *= np.exp(-2j * np.pi * np.fft.rfftfreq(n) * delay)[:, None]
    return np.fft.irfft(X, n=n, axis=0)[:REC_SAMPLES].astype(np.float32)


def simulate_received(obj_pos, obj_vel, latency_s=0.0):
    """One ping: static room + object echo (Doppler) [+ object↔wall
    multipath], recorded `latency_s` early, + noise."""
    out = STATIC_SIGNAL.copy()
    d_tx = float(np.linalg.norm(obj_pos - TX_POS))
    u_tx = (obj_pos - TX_POS) / max(d_tx, 1e-6)
    g_tx = tx_gain(obj_pos - TX_POS)
    for i, mp in enumerate(MIC_POS):
        d_mic = float(np.linalg.norm(obj_pos - mp))
        d_total = d_tx + d_mic
        amp = g_tx * min(OBJ_GAIN / max(d_total**2, 0.01), 3.0)
        v_r = float(obj_vel @ -u_tx) + float(obj_vel @ ((mp - obj_pos) / max(d_mic, 1e-6)))
        _add_chirp(out, i, d_total / C, amp, 1.0 + v_r / C)

    # object ↔ wall multipath: TX→obj→wall→mic and TX→wall→obj→mic (not via
    # the mounting wall: right behind the array, that bounce merges with the
    # direct echo)
    eps = 1e-4
    for wi, w in enumerate(TRUE_WALLS if ARGS.object_multipath else []):
        if wi == MOUNT_IDX:
            continue
        tx_img = wallmap.reflect(TX_POS, w)
        for i, mp in enumerate(MIC_POS):
            mic_img = wallmap.reflect(mp, w)
            for path in (lambda q: np.linalg.norm(q - TX_POS) + np.linalg.norm(q - mic_img),
                         lambda q: np.linalg.norm(q - tx_img) + np.linalg.norm(q - mp)):
                d_total = float(path(obj_pos))
                rate = (float(path(obj_pos + obj_vel * eps)) - d_total) / eps
                _add_chirp(out, i, d_total / C, g_tx * WALL_REFL * OBJ_GAIN
                           / max(d_total**2, 0.01), 1.0 - rate / C)

    if latency_s or REC_SAMPLES != SIM_SAMPLES:
        out = _delayed(out, latency_s * FS)
    out += (np.random.randn(*out.shape) * NOISE_STD).astype(np.float32)
    return out


# ─────────────────────────── object motion ────────────────────────────────────
def obj_pos_at(t):
    x = ROOM_W * 0.35 + (ROOM_W * 0.55) * (0.5 + 0.5 * np.sin(2 * np.pi * t / 5.0 - np.pi / 2))
    y = ROOM_D / 2 + (ROOM_D / 2 - 0.25) * np.sin(2 * np.pi * t / 10.0)
    return np.array([x, y])


def obj_vel_at(t, dt=0.005):
    return (obj_pos_at(t + dt) - obj_pos_at(t - dt)) / (2 * dt)


# ─────────────────────────────── tracker ──────────────────────────────────────
MOUNT_WALL = (None if MOUNT_IDX is None else
              wallmap.Wall(TRUE_WALLS[MOUNT_IDX].normal.copy(), TRUE_WALLS[MOUNT_IDX].offset))
TRACKER = T.Tracker(MIC_POS, TX_POS, CHIRP, SIM_SAMPLES, frame_dt=FRAME_DT,
                    mount_wall=MOUNT_WALL, multipath=ARGS.object_multipath,
                    async_map=not ARGS.eval, sync=not ARGS.no_sync,
                    max_latency_s=MAX_LATENCY_S)
CAL_FRAMES = T.Tracker.CAL_FRAMES

_min_clear = min(np.min(np.linalg.norm(MIC_POS - obj_pos_at(t), axis=1))
                 for t in np.arange(0, 10, 0.05))
print(f"array {LAYOUT['name']} ({N_MICS} mics) at ({POSE[0]:.2f}, {POSE[1]:.2f}) m "
      f"facing {POSE[2]:.0f}°, "
      + (f"mounted on the {WALL_NAMES[MOUNT_IDX]} wall" if MOUNT_IDX is not None
         else "free-standing") + f", {SPEAKER} speaker")
if _min_clear < 0.3:
    print(f"  note: the simulated path passes within {_min_clear*100:.0f} cm of a mic")


def step(frame):
    """Simulate + track one frame. Returns the tracker output + ground truth."""
    t = frame * FRAME_DT
    pos, vel = obj_pos_at(t), obj_vel_at(t)
    lat = (ARGS.latency + np.random.uniform(0, ARGS.jitter)) / 1e3
    return dict(TRACKER.process(simulate_received(pos, vel, lat)), t=t,
                true_pos=pos, true_vel=vel, true_latency_ms=lat * 1e3)


# ─────────────────────────── figure layout ────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(16, 9), facecolor="#111111")
fig.canvas.manager.set_window_title("Ultrasonic tracker — simulator")
gs = GridSpec(2, 2, figure=fig, left=0.04, right=0.97, top=0.93, bottom=0.07,
              wspace=0.30, hspace=0.40, width_ratios=[1.6, 1])
ax_room  = fig.add_subplot(gs[:, 0])
ax_sigs  = fig.add_subplot(gs[0, 1])
ax_track = fig.add_subplot(gs[1, 1])
MIC_COLORS = [plt.cm.hsv(i / N_MICS) for i in range(N_MICS)]


def _style(ax, title):
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="#777777")
    for s in ax.spines.values():
        s.set_edgecolor("#444444")


def _draw_room(ax, labels):
    ax.set_xlim(-0.1, ROOM_W + 0.1)
    ax.set_ylim(-0.22 if labels else 0, ROOM_D + (0.2 if labels else 0))
    ax.set_aspect("equal")
    ax.set_xlabel("x  (m)", color="#aaaaaa")
    ax.set_ylabel("y  (m)", color="#aaaaaa")
    ax.add_patch(mpatches.Rectangle((0, 0), ROOM_W, ROOM_D, lw=2,
                                    edgecolor="#55aaff", facecolor="none"))
    for g in np.arange(0, ROOM_W + 0.01, 0.5):
        ax.axvline(g, color="#ffffff06", linewidth=0.5)
    for g in np.arange(0, ROOM_D + 0.01, 0.5):
        ax.axhline(g, color="#ffffff06", linewidth=0.5)
    ax.add_patch(mpatches.Polygon(arrays.outline(LAYOUT, *POSE), closed=True,
                                  lw=1.2, edgecolor="#888888", facecolor="#1e1e3a", zorder=2))
    for i, (mx, my) in enumerate(MIC_POS):
        ax.plot(mx, my, "o", color=MIC_COLORS[i], ms=7 if labels else 4, zorder=5)
    ax.plot(*TX_POS, "s", color="#f39c12", ms=8 if labels else 4, zorder=5)
    if labels:
        ax.annotate("", TX_POS + 0.18 * FACING, TX_POS, zorder=5,
                    arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.5))
        ax.text(*(TX_POS + 0.24 * FACING), "TX", color="#f39c12", fontsize=8,
                ha="center", va="center")


_style(ax_room, f"Room view  ({ROOM_W:g} m × {ROOM_D:g} m)  |  array {LAYOUT['name']}  |  "
                f"LFM chirp {CHIRP.f0/1e3:.0f}–{CHIRP.f1/1e3:.0f} kHz")
_draw_room(ax_room, labels=True)
true_trail, = ax_room.plot([], [], "-", color="#ffcc00", alpha=0.35, lw=1.5, zorder=3)
true_dot,   = ax_room.plot([], [], "o", color="#ffcc00", ms=14, zorder=6,
                           markeredgecolor="white", markeredgewidth=1.5)
est_trail,  = ax_room.plot([], [], "-", color="#00ccff", alpha=0.35, lw=1.5, zorder=3)
est_dot,    = ax_room.plot([], [], "D", color="#00ccff", ms=11, zorder=7,
                           markeredgecolor="white", markeredgewidth=1.5)
raw_trail,  = ax_room.plot([], [], "-", color="#ff6b35", alpha=0.25, lw=1.0, zorder=2)
raw_dot,    = ax_room.plot([], [], "x", color="#ff6b35", ms=9, zorder=6, markeredgewidth=2.0)
vel_quiver = ax_room.quiver(0, 0, 0, 0, color="#44ff88", scale=8, scale_units="xy",
                            width=0.005, zorder=8, headwidth=5)
wall_line, = ax_room.plot([], [], "--", color="#ff66cc", lw=2.0, zorder=4)
_box = dict(boxstyle="round,pad=0.4", facecolor="#000000cc", edgecolor="#444444")
wall_txt = ax_room.text(0.98, 0.03, "", transform=ax_room.transAxes, color="#ff66cc",
                        fontsize=8.5, va="bottom", ha="right", fontfamily="monospace",
                        bbox=_box, zorder=10)
info_txt = ax_room.text(0.02, 0.97, "", transform=ax_room.transAxes, color="white",
                        fontsize=8.5, va="top", fontfamily="monospace", bbox=_box, zorder=10)
ax_room.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in (
    ("#ffcc00", "True pos"), ("#00ccff", "Kalman est."), ("#ff6b35", "Raw est."),
    ("#44ff88", "Velocity"), ("#ff66cc", "Wall map"))],
    loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white", ncol=2)

HISTORY = 60
env_history = np.zeros((HISTORY, N_MICS))
_style(ax_sigs, "Residual echo level per mic (background removed)")
ax_sigs.set_xlabel("Frame →", color="#aaaaaa")
ax_sigs.set_ylabel("Norm. amplitude", color="#aaaaaa")
sig_lines = [ax_sigs.plot([], [], "-", color=MIC_COLORS[i], lw=1.5, label=f"M{i+1}",
                          alpha=0.9)[0] for i in range(N_MICS)]
ax_sigs.legend(loc="upper right", fontsize=7, framealpha=0.3, ncol=4, labelcolor="white")
ax_sigs.set_xlim(0, HISTORY - 1)
ax_sigs.set_ylim(0, 1.15)               # fixed: lines are normalised to ≤ 1

_style(ax_track, "Position tracking  +  speed")
_draw_room(ax_track, labels=False)
tr_true_sc = ax_track.scatter([], [], s=18, c="#ffcc00", alpha=0.4, zorder=3)
tr_est_sc  = ax_track.scatter([], [], s=18, c="#00ccff", alpha=0.4, zorder=3)
tr_raw_sc  = ax_track.scatter([], [], s=12, c="#ff6b35", alpha=0.3, zorder=2)
tr_vel_q   = ax_track.quiver(0, 0, 0, 0, color="#44ff88", scale=8, scale_units="xy",
                             width=0.008, zorder=8, headwidth=5)
spd_txt = ax_track.text(0.98, 0.97, "", transform=ax_track.transAxes, color="white",
                        fontsize=8.5, va="top", ha="right", fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#00000099",
                                  edgecolor="#444444"))
fig.suptitle(f"Ultrasonic tracker  —  {N_MICS} mics  |  LFM chirp {CHIRP.f0/1e3:.0f}–"
             f"{CHIRP.f1/1e3:.0f} kHz  |  {FS/1e3:.0f} kHz SR  |  matched filter + TOF",
             color="white", fontsize=13, y=0.98)


# ─────────────────────────── drawing ──────────────────────────────────────────
ANIMATED = [true_trail, true_dot, est_trail, est_dot, raw_trail, raw_dot, vel_quiver,
            wall_line, wall_txt, info_txt, *sig_lines, tr_true_sc, tr_est_sc,
            tr_raw_sc, tr_vel_q, spd_txt]
TRUE_HIST = deque(maxlen=TRAIL_LEN)
EST_HIST  = deque(maxlen=TRAIL_LEN)
RAW_HIST  = deque(maxlen=TRAIL_LEN)


def wall_report(m):
    if m is None:
        return "wall map: pending"
    depth, width = np.ptp(m.polygon, axis=0) if len(m.polygon) else (np.nan, np.nan)
    n = sum(not w.known for w in m.walls)
    return "\n".join([f"── WALL MAP ({n} walls{'' if m.closed else ', open'}) ──",
                      f"  room  {depth:.3f} × {width:.3f} m" if m.closed else "  room  not closed",
                      f"  true  {ROOM_W:.3f} × {ROOM_D:.3f} m"])


def draw_walls(m):
    if m is None or not m.closed:
        wall_line.set_data([], [])
    else:
        poly = np.vstack([m.polygon, m.polygon[:1]])
        wall_line.set_data(poly[:, 0], poly[:, 1])
    wall_txt.set_text(wall_report(m))


def _update(frame):
    s = step(frame)
    draw_walls(TRACKER.wall_map)
    if s["calibrating"]:
        info_txt.set_text(f"calibrating static background…\n"
                          f"  {s['cal_progress']*100:.0f} %  (keep the room as is)\n"
                          f"  latency {s['latency_ms']:.3f} ms (true {s['true_latency_ms']:.3f})")
        info_txt.set_color("#cccccc")
        return
    pos, vel = s["true_pos"], s["true_vel"]
    TRUE_HIST.append(pos.copy())
    true_trail.set_data(*np.array(TRUE_HIST).T)
    true_dot.set_data([pos[0]], [pos[1]])

    env_history[:-1] = env_history[1:]
    env_history[-1] = s["envs"]
    norm = env_history / (env_history.max() + 1e-9)
    for i, ln in enumerate(sig_lines):
        ln.set_data(np.arange(HISTORY), norm[:, i])

    snr_db = 20 * np.log10(np.median(s["snr"]))
    if not s["detected"] or not TRACKER.kf.started:
        info_txt.set_text(f"nothing detected   (median echo SNR {snr_db:.0f} dB)\n"
                          "monitoring…")
        info_txt.set_color("#cccccc")
        return
    p, v = s["pos"], s["vel"]
    EST_HIST.append(p.copy())
    if s["pos_est"] is not None:
        RAW_HIST.append(s["pos_est"].copy())
    est_trail.set_data(*np.array(EST_HIST).T)
    est_dot.set_data([p[0]], [p[1]])
    if RAW_HIST:
        raw_trail.set_data(*np.array(RAW_HIST).T)
        raw_dot.set_data([RAW_HIST[-1][0]], [RAW_HIST[-1][1]])
    for q, off in ((vel_quiver, [p]), (tr_vel_q, [p])):
        q.set_offsets(off)
        q.set_UVC(v[0], v[1])
    tr_true_sc.set_offsets(np.array(TRUE_HIST))
    tr_est_sc.set_offsets(np.array(EST_HIST))
    if RAW_HIST:
        tr_raw_sc.set_offsets(np.array(RAW_HIST))

    speed, true_speed = float(np.linalg.norm(v)), float(np.linalg.norm(vel))
    err = float(np.linalg.norm(p - pos))
    true_range = float(np.linalg.norm(pos - TRACKER.centre))
    moving = speed > 0.1
    info_txt.set_text("\n".join([
        "MOVING" if moving else "PRESENT (still)",
        f"  distance {s['range_m']:.2f} m   (true {true_range:.2f})",
        f"  speed    {speed:.2f} m/s (true {true_speed:.2f})",
        f"  pos   ({p[0]:.2f}, {p[1]:.2f}) m",
        f"  true  ({pos[0]:.2f}, {pos[1]:.2f}) m",
        f"  error {err*100:.1f} cm",
        f"  mics used {int(s['used'].sum())}/{N_MICS}   SNR {snr_db:.0f} dB",
        f"  latency  {s['latency_ms']:.3f} ms (true {s['true_latency_ms']:.3f})",
        "" if s["accepted"] else "  (measurement gated out)"]))
    info_txt.set_color("#ff4444" if moving else "#ffcc66")
    spd_txt.set_text(f"speed (Kalman)  {speed:.2f} m/s\n"
                     f"speed (true)    {true_speed:.2f} m/s\n"
                     f"pos error  {err*100:.1f} cm\n"
                     f"fit resid  {s['cost']*1000:.1f} mm")


def update(frame):
    """Animation step. Returns the artists that change (for blitting:
    only these are redrawn each frame; the rest of the figure is cached)."""
    _update(frame)
    return ANIMATED


# ─────────────────────────────── eval ─────────────────────────────────────────
def run_eval(n_frames, warmup=CAL_FRAMES + 20):
    """Headless accuracy report."""
    pe_raw, pe_kf, ve_kf, rng_err, ex, ey, lat_err = [], [], [], [], [], [], []
    rejected = missed = held = 0
    for f in range(n_frames):
        s = step(f)
        lat_err.append(s["latency_ms"] - s["true_latency_ms"])
        held += not s["sync_ok"]
        if f < warmup or s["calibrating"]:
            continue
        if not s["detected"] or s["pos_est"] is None:
            missed += 1
            continue
        e = s["pos"] - s["true_pos"]
        pe_kf.append(np.linalg.norm(e)); ex.append(e[0]); ey.append(e[1])
        pe_raw.append(np.linalg.norm(s["pos_est"] - s["true_pos"]))
        ve_kf.append(np.linalg.norm(s["vel"] - s["true_vel"]))
        rng_err.append(s["range_m"] - np.linalg.norm(s["true_pos"] - TRACKER.centre))
        rejected += not s["accepted"]
    cm = lambda a: np.asarray(a) * 100
    print(f"frames evaluated: {len(pe_kf)}  (after {warmup} warm-up), "
          f"not detected: {missed}")
    if pe_kf:
        pe_raw, pe_kf = np.array(pe_raw), np.array(pe_kf)
        print(f"raw position     mean {cm(pe_raw.mean()):5.1f} cm  "
              f"p95 {cm(np.percentile(pe_raw, 95)):5.1f} cm  max {cm(pe_raw.max()):5.1f} cm")
        print(f"Kalman position  mean {cm(pe_kf.mean()):5.1f} cm  "
              f"p95 {cm(np.percentile(pe_kf, 95)):5.1f} cm  max {cm(pe_kf.max()):5.1f} cm")
        print(f"  RMS x {cm(np.sqrt(np.mean(np.square(ex)))):.1f} cm   "
              f"RMS y {cm(np.sqrt(np.mean(np.square(ey)))):.1f} cm   "
              f"distance error RMS {cm(np.sqrt(np.mean(np.square(rng_err)))):.1f} cm")
        print(f"Kalman velocity  mean {np.mean(ve_kf):.3f} m/s  "
              f"p95 {np.percentile(ve_kf, 95):.3f} m/s")
        print(f"gated-out measurements: {rejected}")
    le = np.abs(lat_err) * 1e3
    print(f"latency sync     " + ("off" if ARGS.no_sync else
          f"error mean {le.mean():.2f} µs  max {le.max():.2f} µs "
          f"(1 sample = {1e6/FS:.1f} µs), held {held} pings"))

    m = TRACKER.wall_map
    print()
    print(wall_report(m).replace("──", "").strip())
    if m is None:
        return
    measured = [w for w in m.walls if not w.known]
    for wi, (name, tw) in enumerate(zip(WALL_NAMES, TRUE_WALLS)):
        if wi == MOUNT_IDX:
            continue                                  # known (mounting wall)
        near = [w for w in measured
                if np.degrees(np.arccos(np.clip(w.normal @ tw.normal, -1, 1))) < 15]
        if not near:
            print(f"  {name:7s} NOT FOUND")
            continue
        w = min(near, key=lambda w: abs(w.offset - tw.offset))
        ang = np.degrees(np.arccos(np.clip(w.normal @ tw.normal, -1, 1)))
        raw = abs((w.raw_angle_deg - tw.angle_deg + 180) % 360 - 180)
        print(f"  {name:7s} distance error {(w.offset - tw.offset)*1000:+6.1f} mm   "
              f"angle error {ang:.2f}° (raw fit {raw:.2f}°)")
    extra = len(measured) - (4 - (MOUNT_IDX is not None))
    if extra > 0:
        print(f"  + {extra} spurious wall(s)")


if __name__ == "__main__" and ARGS.eval:
    run_eval(ARGS.eval)
elif __name__ == "__main__":
    ani = animation.FuncAnimation(fig, update, init_func=lambda: ANIMATED,
                                  interval=ANIM_INTERVAL, blit=True,
                                  cache_frame_data=False)
    plt.show()
