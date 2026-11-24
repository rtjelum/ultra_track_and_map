#!/usr/bin/env python3
"""Render compare_view.py (recording vs simulation, LiDAR as truth) into a GIF.

    python tools/make_compare_gif.py                      # walk1 vs walk1_sim, the walk
    python tools/make_compare_gif.py --no-sim             # the recording only
    python tools/make_compare_gif.py recordings/other --out docs/other_compare.gif

Needs the simulated recording (default STEM_sim; make it with
``python recording_sim.py STEM``). By default it covers the walk, as the
viewer's loop does; --start/--end pick another span.
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

import compare_view as CV
import gifsave

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("stem", nargs="?", default=os.path.join(ROOT, "recordings", "walk1"))
ap.add_argument("--sim", help="the simulated recording (default STEM_sim)")
ap.add_argument("--no-sim", action="store_true", help="show the recording only")
ap.add_argument("--start", type=float, help="first frame, s (default: the walk)")
ap.add_argument("--end", type=float, help="last frame, s (default: the walk)")
ap.add_argument("--step", type=float, default=0.2, help="recording seconds per frame")
ap.add_argument("--speed", type=float, default=1.0, help="playback speed of the GIF")
ap.add_argument("--width", type=int, default=960, help="GIF width in pixels")
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "compare_walk1.gif"))
args = ap.parse_args()

sim_stem = None if args.no_sim else args.sim or f"{args.stem}_sim"
if sim_stem and not os.path.exists(sim_stem + ".flac"):
    raise SystemExit(f"{sim_stem}.flac not found — run: python recording_sim.py {args.stem}")

fig, draw, t_end, walk = CV.build(args.stem, sim_stem)
fig.loop_band.set_visible(False)
start = walk[0] if args.start is None else args.start
end = walk[1] if args.end is None else min(args.end, t_end)
clock = fig.text(0.5, 0.04, "", ha="center", fontsize=13, family="monospace")
name = os.path.basename(args.stem)
fig.text(0.04, 0.04, f"{name}: " + ("recording" if args.no_sim else "recording vs simulation")
         + ", AudiodeviceMapper, LiDAR as ground truth", fontsize=11, color="#aaa")

frames = []
for t in np.arange(start, end + 1e-9, args.step):
    draw(t)
    clock.set_text(f"t = {t:5.1f} s")
    frames.append(gifsave.grab(fig, args.width))

gifsave.save(frames, args.out, 1000 * args.step / args.speed, keys=[
    "#15171a", "#1d2024", CV.REAL_C, CV.SIM_C, CV.TRUTH_C, "#8a8f98", "#c9a", "#e55",
    "#59f", "#dddddd", "#444444", "#ffffff"], colors=160)
