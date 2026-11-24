"""Shared GIF helpers for the tools/make_*_gif.py scripts."""
import os

from PIL import Image


def grab(fig, width):
    """The figure as an RGB image, ``width`` pixels wide."""
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                           fig.canvas.buffer_rgba()).convert("RGB")
    w, h = img.size
    return img.resize((width, round(h * width / w)), Image.LANCZOS)


def save(frames, out, duration_ms, keys=(), colors=96):
    """Write ``frames`` as a looping GIF with one shared palette, built from a
    few frames plus the plot's key colours so thin markers keep their colour."""
    w, h = frames[0].size
    picks = sorted({0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1})
    sheet = Image.new("RGB", (w, h * len(picks) + 40))
    for i, k in enumerate(picks):
        sheet.paste(frames[k], (0, h * i))
    if keys:
        sw = w // len(keys)
        for i, c in enumerate(keys):
            sheet.paste(Image.new("RGB", (sw, 40), c), (i * sw, h * len(picks)))
    pal = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    q[0].save(out, save_all=True, append_images=q[1:], duration=int(round(duration_ms)),
              loop=0, optimize=True)
    print(f"wrote {out}  ({w}×{h}, {len(q)} frames, {os.path.getsize(out) / 1024:.0f} KB)")
