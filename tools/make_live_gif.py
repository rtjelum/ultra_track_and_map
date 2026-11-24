#!/usr/bin/env python3
"""Render the Audiodevice live demo (audiodevice_live.py) replaying a recording
into a GIF, with the recording's LiDAR scan and walker drawn on the map.

    python tools/make_live_gif.py                       # walk1, the walk, real time
    python tools/make_live_gif.py --start 0 --end 55    # the whole recording
    python tools/make_live_gif.py recordings/other --out docs/other.gif

The GUI is driven headless and in step with the recording (no threads, no
device, no audio): each recorded chunk goes through the same LiveMapper the
live demo uses, and the window is captured every --step seconds of recording.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import audiodevice_live as L
import audiodevice_recording as R
import compare_view as CV
import gifsave

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("stem", nargs="?", default=os.path.join(ROOT, "recordings", "walk1"))
ap.add_argument("--start", type=float, default=4.0, help="first frame (s of recording)")
ap.add_argument("--end", type=float, default=30.5, help="last frame (s of recording)")
ap.add_argument("--step", type=float, default=0.2, help="recording seconds per frame")
ap.add_argument("--speed", type=float, default=1.0, help="playback speed of the GIF")
ap.add_argument("--width", type=int, default=800, help="GIF width in pixels")
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "live_walk1.gif"))
args = ap.parse_args()

rec = R.load(args.stem)
band = tuple(rec.meta.get("band_hz", (18_000, 22_000)))
name = os.path.basename(args.stem)
demo = L.LiveDemo(f"recording {name} (replay)", band,
                  rec.meta.get("level_percent", 20), None, stream=lambda *a: None)
demo.timer.stop()
model = L.mapper.LiveMapper(*band)

# LiDAR overlay on the live map (the GUI itself has none)
t0 = rec.chunk_times[0]
scan_t = np.array([t for t, _ in rec.scans]) - t0
scan_xy = [CV.lidar_xy(s["points"]) for _, s in rec.scans]
ax = demo.map_axis
scan_art, = ax.plot([], [], ".", color="#9aa5b1", ms=2.5, zorder=1, label="LiDAR scan")
truth_art, = ax.plot([], [], "o", color="#3ccf5a", ms=13, mfc="none", mew=2,
                     label="LiDAR walker (truth)")
truth_trail, = ax.plot([], [], "-", color="#3ccf5a", lw=1.2, alpha=0.6)
ax.get_legend().remove()                  # legend beside the map, not over it
handles, labels = ax.get_legend_handles_labels()
demo.info_axis.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.6,
                      facecolor="#1a2533", edgecolor="#718096", labelcolor="#dce4ec")
demo.figure.texts[0].set_text("Audiodevice live chirp tracker — replay of "
                              f"{name} with LiDAR ground truth")
# fit the map to the room the LiDAR sees during the shown span, so it doesn't jump
shown = [xy for t, xy in zip(scan_t, scan_xy) if args.start <= t <= args.end]
if shown:
    pts = np.vstack(shown)
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(min(x0, lo[1] - 0.2), max(x1, hi[1] + 0.2))
    ax.set_ylim(y0, max(y1, hi[0] + 0.2))


def overlay(t):
    i = int(np.searchsorted(scan_t, t, side="right")) - 1
    sxy = scan_xy[i] if i >= 0 else np.empty((0, 2))
    scan_art.set_data(sxy[:, 1], sxy[:, 0])
    trail = CV.truth_at(rec, np.arange(max(0.0, t - CV.TRAIL_S), t + 1e-9, 0.05))
    truth_trail.set_data(trail[:, 1], trail[:, 0])
    now = trail[-1] if len(trail) else (np.nan, np.nan)
    truth_art.set_data([now[1]], [now[0]])


frames, next_t, last = [], args.start, None
for t, chunk in rec.chunks():
    t -= t0
    for frame in model.push(chunk):
        last = frame
        demo._show_frame(frame)
    if t > args.end:
        break
    if t >= next_t and last is not None:
        overlay(t)
        demo.status.set_text(f"t = {t:5.1f} s   " + demo.status.get_text().split("   ")[-1])
        frames.append(gifsave.grab(demo.figure, args.width))
        next_t += args.step

gifsave.save(frames, args.out, 1000 * args.step / args.speed, keys=[
    "#101722", "#1a2533", "#ffae42", "#fa6d6d", "#8bd17c", "#44c2ad", "#89aaff", "#d599f0",
    "#40dcff", "#4cd1ff", "#ffd36b", "#f2a7c3", "#3ccf5a", "#9aa5b1", "#87e0cb", "#ffcc80",
    "#ffffff"])
