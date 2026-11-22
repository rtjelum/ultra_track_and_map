#!/usr/bin/env python3
"""Render the simulator headless into docs/simulation.gif (for the README).

    python tools/make_gif.py            # 10 s of tracking after calibration
    python tools/make_gif.py --frames 150 --seed 3
    python tools/make_gif.py --out ring.gif -- --array ring8 --pose 1.5 0 90
                                         # args after -- go to simulator.py
"""
import argparse
import contextlib
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=100, help="frames to record (10 per second)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--width", type=int, default=960, help="GIF width in pixels")
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "simulation.gif"))
ap.add_argument("sim_args", nargs="*", help="extra simulator.py arguments (after --)")
args = ap.parse_args()

# --eval selects the non-interactive backend and synchronous wall mapping
sys.argv = ["simulator.py", "--eval", "1", "--seed", str(args.seed), *args.sim_args]
sys.path.insert(0, ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    import simulator as S
from PIL import Image

for f in range(S.CAL_FRAMES):            # skip the background calibration
    S.update(f)

frames = []
for f in range(S.CAL_FRAMES, S.CAL_FRAMES + args.frames):
    S.update(f)
    S.fig.canvas.draw()
    img = Image.frombuffer("RGBA", S.fig.canvas.get_width_height(),
                           S.fig.canvas.buffer_rgba()).convert("RGB")
    w, h = img.size
    frames.append(img.resize((args.width, round(h * args.width / w)), Image.LANCZOS))

# one palette for all frames, built from several frames + the plot's key
# colours so thin coloured markers don't get merged into grey
keys = ["#ffcc00", "#00ccff", "#ff6b35", "#44ff88", "#ff66cc", "#55aaff", "#f39c12",
        "#e74c3c", "#2ecc71", "#3498db", "#ffffff", "#1a1a2e", "#111111"]
w, h = frames[0].size
picks = sorted({0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1})
sheet = Image.new("RGB", (w, h * len(picks) + 40))
for i, k in enumerate(picks):
    sheet.paste(frames[k], (0, h * i))
sw = w // len(keys)
for i, c in enumerate(keys):
    sheet.paste(Image.new("RGB", (sw, 40), c), (i * sw, h * len(picks)))
pal = sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]

os.makedirs(os.path.dirname(args.out), exist_ok=True)
q[0].save(args.out, save_all=True, append_images=q[1:],
          duration=int(S.ANIM_INTERVAL), loop=0, optimize=True)
print(f"wrote {args.out}  ({w}×{h}, {len(q)} frames, "
      f"{os.path.getsize(args.out) / 1024:.0f} KB)")
