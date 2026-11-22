"""
Mic-array layouts and their placement in a room.

A layout is given in the array's own frame:
    x = forward (the direction the speaker faces, away from the mounting surface)
    y = to the left of forward
    origin = the mounting point (on the wall surface when wall-mounted)

so the same layout can be put anywhere with a pose (x, y, facing angle).

Custom layouts: a JSON file
    {"mics": [[x, y], ...], "tx": [x, y]}
in metres, array frame. At least 3 mics; they must not all lie on one line
with each other if the array is free-standing (see Tracker's warning).
"""

import json
import os

import numpy as np

_r3 = 0.15 / np.sqrt(3)            # circumradius of a 15 cm triangle

PRESETS = {
    "bar6": dict(
        desc="6 mics, 2×3 grid on a 50×10 cm bar, TX at the top end (original)",
        mics=[[0.05 + dx, -0.25 + dy] for dy in (0.0, 0.1, 0.2) for dx in (0.0, 0.1)],
        tx=[0.05, 0.20]),
    "line4": dict(
        desc="4 mics in a 45 cm line along the wall, TX in the middle "
             "(collinear: needs the mounting wall to tell front from back)",
        mics=[[0.03, y] for y in (-0.225, -0.075, 0.075, 0.225)],
        tx=[0.03, 0.0]),
    "tri3": dict(
        desc="3 mics on a 15 cm triangle, TX in the centre (minimum layout)",
        mics=[[0.10 + _r3 * np.cos(a), _r3 * np.sin(a)]
              for a in np.radians([0.0, 120.0, 240.0])],
        tx=[0.10, 0.0]),
    "ring8": dict(
        desc="8 mics on a 10 cm circle, TX in the centre (compact; works free-standing)",
        mics=[[0.06 + 0.05 * np.cos(a), 0.05 * np.sin(a)]
              for a in np.radians(np.arange(0, 360, 45))],
        tx=[0.06, 0.0]),
}


def load(name_or_path):
    """Preset name or JSON file → dict(mics=(n, 2) array, tx=(2,) array, name)."""
    if name_or_path in PRESETS:
        d, name = PRESETS[name_or_path], name_or_path
    elif os.path.isfile(name_or_path):
        with open(name_or_path) as f:
            d = json.load(f)
        name = os.path.splitext(os.path.basename(name_or_path))[0]
    else:
        raise ValueError(f"unknown array '{name_or_path}' "
                         f"(presets: {', '.join(PRESETS)}, or a JSON file)")
    mics = np.asarray(d["mics"], float).reshape(-1, 2)
    tx = np.asarray(d["tx"], float).reshape(2)
    if len(mics) < 3:
        raise ValueError("need at least 3 mics for 2D positioning")
    return dict(name=name, mics=mics, tx=tx, desc=d.get("desc", ""))


def rotation(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def place(layout, x, y, facing_deg):
    """Layout → room coordinates. Returns (mic_pos (n, 2), tx_pos (2,),
    facing unit vector (2,))."""
    R, o = rotation(facing_deg), np.array([x, y], float)
    return o + layout["mics"] @ R.T, o + R @ layout["tx"], R @ np.array([1.0, 0.0])


def outline(layout, x, y, facing_deg, pad=0.02):
    """Rectangle around the array (for drawing), room coordinates (4, 2)."""
    pts = np.vstack([layout["mics"], layout["tx"]])
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    lo[0] = min(lo[0], 0.0)             # include the mounting point
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return np.array([x, y]) + box @ rotation(facing_deg).T
