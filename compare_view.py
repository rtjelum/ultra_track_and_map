"""
Side-by-side replay: a recording and its simulation, both through the
AudiodeviceMapper (the tracker `audiodevice_recording.py replay` scores),
with the LiDAR as ground truth.

    python compare_view.py recordings/walk1                 # vs recordings/walk1_sim
    python compare_view.py recordings/walk1 --sim other_sim --speed 2

Keys: space pause · ←/→ ∓1 s · +/- speed · drag the time slider · q quit

Both recordings are tracked first (a few seconds), then played back in sync.
Panels: the scene for each (LiDAR now, walker truth and trail, mapper
estimate and trail, facing wall), range over time, and the mapper's
integrated range profile with its detection threshold.
"""

import argparse
import os
import time

import numpy as np

import audiodevice_mapper as M
import audiodevice_recording as R
import audiodevice_tracking as G

LIDAR_DELAY = 0.1
TRAIL_S = 2.0
REAL_C, SIM_C, TRUTH_C = "#ff8c1a", "#2ea8ff", "#3ccf5a"


def run_mapper(rec):
    """Mapper over a whole recording → per-ping arrays."""
    low, high = rec.meta.get("band_hz", (18_000, 22_000))
    mapper = M.AudiodeviceMapper(low, high, angle_offset_deg=R.angle_offset(rec))
    t, state, pos, score, thr, prof, wall = [], [], [], [], [], [], []
    for _, chunk in rec.chunks():
        for f in mapper.push(chunk):
            if f.sample_index < 0:
                continue
            t.append(rec.sample_time(f.sample_index))
            state.append(f.state)
            pos.append(f.position or (np.nan, np.nan))
            score.append(f.score)
            thr.append(f.threshold)
            prof.append(f.profile if f.profile is not None else np.full(len(mapper.ranges), np.nan))
            wall.append(f.facing_wall_m if f.facing_wall_m is not None else np.nan)
    return dict(t=np.array(t), state=np.array(state), pos=np.array(pos, float),
                score=np.array(score), thr=np.array(thr), prof=np.array(prof),
                wall=np.array(wall, float), ranges=mapper.ranges)


def truth_at(rec, t):
    tr = rec.track
    if tr is None:
        return np.full((len(np.atleast_1d(t)), 2), np.nan)
    ok = ~np.isnan(tr[:, 1])
    q = np.atleast_1d(t) + LIDAR_DELAY
    out = np.column_stack([np.interp(q, tr[ok, 0], tr[ok, k], left=np.nan, right=np.nan)
                           for k in (1, 2)])
    j = np.clip(np.searchsorted(tr[ok, 0], q), 1, ok.sum() - 1)
    gap = np.minimum(np.abs(tr[ok, 0][j] - q), np.abs(tr[ok, 0][j - 1] - q))
    out[gap > 0.35] = np.nan
    return out


def lidar_xy(points):
    p = np.asarray(points, float).reshape(-1, 3)
    a = np.radians(p[:, 0])
    return np.column_stack([p[:, 1] * np.cos(a), p[:, 1] * np.sin(a)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="the real recording, e.g. recordings/walk1")
    ap.add_argument("--sim", help="the simulated recording (default STEM_sim)")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    sim_stem = a.sim or f"{a.stem}_sim"
    if not os.path.exists(sim_stem + ".flac"):
        raise SystemExit(f"{sim_stem}.flac not found — run: python recording_sim.py {a.stem}")

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    print("tracking both recordings …", flush=True)
    real_rec, sim_rec = R.load(a.stem), R.load(sim_stem)
    runs = {"recording": run_mapper(real_rec), "simulation": run_mapper(sim_rec)}
    t_end = min(r["t"][-1] for r in runs.values())
    t0_chunk = real_rec.chunk_times[0]
    scan_t = np.array([t for t, _ in real_rec.scans]) - t0_chunk
    scan_xy = [lidar_xy(s["points"]) for _, s in real_rec.scans]
    for r in runs.values():
        r["truth"] = truth_at(real_rec, r["t"])
        r["err"] = np.linalg.norm(r["pos"] - r["truth"], axis=1)
    truth_t = np.arange(0, t_end, 0.05)
    truth_xy = truth_at(real_rec, truth_t)
    print(f"ready: {t_end:.0f} s", flush=True)

    plt.rcParams.update({"figure.facecolor": "#15171a", "axes.facecolor": "#1d2024",
                         "axes.edgecolor": "#555", "text.color": "#ddd",
                         "axes.labelcolor": "#bbb", "xtick.color": "#999",
                         "ytick.color": "#999", "grid.color": "#333"})
    plt.rcParams["keymap.back"] = []
    plt.rcParams["keymap.forward"] = []
    fig = plt.figure(figsize=(16, 9.5))
    fig.canvas.manager.set_window_title(f"{a.stem}: recording vs simulation")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1], left=0.04, right=0.98,
                          top=0.95, bottom=0.11, hspace=0.25, wspace=0.18)
    maps = {"recording": fig.add_subplot(gs[0, 0]), "simulation": fig.add_subplot(gs[0, 1])}
    ax_rt = fig.add_subplot(gs[0, 2])
    ax_pr = fig.add_subplot(gs[1, :])
    colors = {"recording": REAL_C, "simulation": SIM_C}

    art = {}
    for name, ax in maps.items():
        ax.set_aspect("equal")
        ax.set_xlim(-2.1, 2.6)
        ax.set_ylim(-0.4, 4.7)
        ax.grid(alpha=0.5)
        ax.set_xlabel("right (m)")
        ax.set_ylabel("depth (m)")
        ax.set_title(name, color=colors[name], fontsize=13)
        ax.plot(G.MIC_POSITIONS[:, 1], G.MIC_POSITIONS[:, 0], "^", c="#e55", ms=5)
        ax.plot(G.TX_POSITION[1], G.TX_POSITION[0], "s", c="#59f", ms=6)
        d = dict(
            scan=ax.plot([], [], ".", c="#8a8f98", ms=2)[0],
            wall=ax.axhline(np.nan, c="#c9a", ls="--", lw=1),
            truth_trail=ax.plot([], [], "-", c=TRUTH_C, lw=1.5, alpha=0.6)[0],
            truth=ax.plot([], [], "o", c=TRUTH_C, ms=11, mfc="none", mew=2)[0],
            est_trail=ax.plot([], [], "-", c=colors[name], lw=1.5, alpha=0.6)[0],
            est=ax.plot([], [], "o", c=colors[name], ms=8)[0],
            info=ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace",
                         fontsize=9, bbox=dict(fc="#000a", ec="none")),
        )
        art[name] = d
    maps["recording"].plot([], [], "o", c=TRUTH_C, mfc="none", mew=2, label="LiDAR walker")
    maps["recording"].plot([], [], ".", c="#8a8f98", label="LiDAR scan")
    maps["recording"].plot([], [], "--", c="#c9a", label="facing wall (mapper)")
    maps["recording"].legend(loc="lower right", fontsize=8, facecolor="#000", edgecolor="none")

    # range over time (static, with a cursor)
    ax_rt.plot(truth_t, np.linalg.norm(truth_xy, axis=1), c=TRUTH_C, lw=2, label="LiDAR")
    for name, r in runs.items():
        ax_rt.plot(r["t"], np.linalg.norm(r["pos"], axis=1), ".", c=colors[name], ms=3, label=name)
    ax_rt.set_xlim(0, t_end)
    ax_rt.set_ylim(0, 4.5)
    ax_rt.set_xlabel("time (s)")
    ax_rt.set_ylabel("range (m)")
    ax_rt.grid(alpha=0.5)
    ax_rt.legend(fontsize=8, facecolor="#000", edgecolor="none", loc="upper right")
    cursor = ax_rt.axvline(0, c="#fff", lw=1)
    summary = []
    for name, r in runs.items():
        present = ~np.isnan(r["truth"][:, 0]) & (r["state"] != "calibrating") & (r["state"] != "locking")
        tracked = present & ~np.isnan(r["pos"][:, 0])
        summary.append(f"{name}: tracked {100 * tracked.sum() / max(present.sum(), 1):.0f}%, "
                       f"median error {np.nanmedian(r['err'][tracked]):.2f} m")
    ax_rt.set_title("\n".join(summary), fontsize=9)

    # range profile
    rng_axis = runs["recording"]["ranges"]
    prof_lines = {n: ax_pr.plot(rng_axis, np.zeros_like(rng_axis), c=colors[n], lw=1.5, label=n)[0]
                  for n in runs}
    thr_lines = {n: ax_pr.axhline(np.nan, c=colors[n], ls=":", lw=1) for n in runs}
    truth_line = ax_pr.axvline(np.nan, c=TRUTH_C, lw=2, alpha=0.7, label="LiDAR range")
    ax_pr.set_xlim(rng_axis[0], rng_axis[-1])
    ax_pr.set_ylim(0, 10)
    ax_pr.set_xlabel("range (m)")
    ax_pr.set_ylabel("integrated score")
    ax_pr.set_title("mapper range profile (dotted: detection threshold)", fontsize=10)
    ax_pr.grid(alpha=0.5)
    ax_pr.legend(fontsize=8, facecolor="#000", edgecolor="none", loc="upper right")

    ax_sl = fig.add_axes([0.08, 0.03, 0.74, 0.025], facecolor="#2a2d31")
    slider = Slider(ax_sl, "time (s)", 0, t_end, valinit=0, color="#555")
    slider.label.set_color("#ddd")
    slider.valtext.set_color("#ddd")
    state = dict(t=0.0, playing=True, speed=a.speed, wall=time.monotonic(), setting=False)
    status = fig.text(0.90, 0.035, "", fontsize=10, family="monospace")

    def draw(t):
        i_scan = int(np.searchsorted(scan_t, t, side="right")) - 1
        sxy = scan_xy[i_scan] if i_scan >= 0 else np.empty((0, 2))
        j0 = int(np.searchsorted(truth_t, t - TRAIL_S))
        j1 = int(np.searchsorted(truth_t, t, side="right"))
        tt = truth_xy[j0:j1]
        for name, r in runs.items():
            d = art[name]
            i = int(np.searchsorted(r["t"], t, side="right")) - 1
            d["scan"].set_data(sxy[:, 1], sxy[:, 0])
            d["truth_trail"].set_data(tt[:, 1], tt[:, 0])
            last = tt[-1] if len(tt) else (np.nan, np.nan)
            d["truth"].set_data([last[1]], [last[0]])
            if i < 0:
                d["est"].set_data([], [])
                d["est_trail"].set_data([], [])
                d["info"].set_text("waiting for lock …")
                prof_lines[name].set_ydata(np.full_like(rng_axis, np.nan))
                continue
            k0 = int(np.searchsorted(r["t"], t - TRAIL_S))
            p = r["pos"][k0:i + 1]
            d["est_trail"].set_data(p[:, 1], p[:, 0])
            d["est"].set_data([r["pos"][i, 1]], [r["pos"][i, 0]])
            d["wall"].set_ydata([r["wall"][i]] * 2)
            err = r["err"][i]
            d["info"].set_text(f"{r['state'][i]:11s} t {t:5.1f} s\n"
                               f"score {r['score'][i]:4.1f} / {r['thr'][i]:3.1f}\n"
                               f"error {'   –' if np.isnan(err) else f'{err:4.2f}'} m")
            prof_lines[name].set_ydata(r["prof"][i])
            thr_lines[name].set_ydata([r["thr"][i]] * 2)
        truth_line.set_xdata([np.linalg.norm(tt[-1])] * 2 if len(tt) else [np.nan] * 2)
        cursor.set_xdata([t, t])
        status.set_text(f"{'▶' if state['playing'] else '❚❚'} ×{state['speed']:g}")
        state["setting"] = True
        slider.set_val(t)
        state["setting"] = False

    def tick(_):
        now = time.monotonic()
        if state["playing"]:
            state["t"] += (now - state["wall"]) * state["speed"]
            if state["t"] >= t_end:
                state["t"], state["playing"] = t_end, False
        state["wall"] = now
        draw(state["t"])

    def on_slider(v):
        if not state["setting"]:
            state["t"] = float(v)

    def on_key(e):
        if e.key == " ":
            state["playing"] = not state["playing"]
            if state["playing"] and state["t"] >= t_end:
                state["t"] = 0.0
        elif e.key == "right":
            state["t"] = min(t_end, state["t"] + 1)
        elif e.key == "left":
            state["t"] = max(0.0, state["t"] - 1)
        elif e.key in ("+", "="):
            state["speed"] = min(8.0, state["speed"] * 2)
        elif e.key in ("-", "_"):
            state["speed"] = max(0.125, state["speed"] / 2)

    slider.on_changed(on_slider)
    fig.canvas.mpl_connect("key_press_event", on_key)
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, tick, interval=100, cache_frame_data=False)
    plt.show()
    del anim


if __name__ == "__main__":
    main()
