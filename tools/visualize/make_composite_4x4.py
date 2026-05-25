"""Build a 4x4 composite GIF from 16 hand-picked trace-overlay GIFs.

Layout (rows top-to-bottom, cols left-to-right):
  Row 1: Bridge  — 1:1 aspect
  Row 2: Fractal — 5:4 aspect
  Row 3: BC-Z    — 5:4 aspect
  Row 4: DROID   — 16:9 aspect

Style notes:
- **No letterbox.** Each row uses its own aspect-correct cell height. All
  four cells in a row share the same input aspect ratio so they tile
  perfectly without any internal padding.
- **Hairline 1 px gutters** between cells (just enough to separate visually,
  not the thick black margins the earlier version had).
- Frames are resampled linearly to a uniform N_OUT so all 16 clips play
  inside one looped sequence.
"""

from PIL import Image
from pathlib import Path

PICKS = [
    [("bridge",  5317), ("bridge",  10663), ("bridge",  21302), ("bridge",  15995)],
    [("fractal", 17431),("fractal", 34901), ("fractal", 43630), ("fractal", 61060)],
    [("bcz",     4331), ("bcz",     12997), ("bcz",     25940), ("bcz",     30275)],
    [("droid",   5699), ("droid",   11349), ("droid",   33979), ("droid",   39547)],
]
GIF_ROOT = Path("/projects/u6gs/spikefly.u6gs/trace_viz_gifs")
CELL_W = 160
GUTTER = 1
N_OUT = 40
FPS = 10


def load_frames(path: Path):
    im = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(im.copy().convert("RGB"))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return frames


def sample_indices(n_in: int, n_out: int):
    if n_in <= 1:
        return [0] * n_out
    return [min(n_in - 1, int(round(t * (n_in - 1) / (n_out - 1)))) for t in range(n_out)]


def main():
    print("loading 16 GIFs ...")
    all_frames = []
    row_heights = []
    for row in PICKS:
        row_frames = []
        sample_img = None
        for ds, ep in row:
            p = GIF_ROOT / ds / f"episode_{ep:06d}.gif"
            fr = load_frames(p)
            print(f"  {ds:8s} ep{ep:06d}  {len(fr)} frames  {fr[0].size}")
            row_frames.append(fr)
            if sample_img is None:
                sample_img = fr[0]
        sw, sh = sample_img.size
        row_h = int(round(CELL_W * sh / sw))
        row_heights.append(row_h)
        all_frames.append(row_frames)

    cols, rows = 4, 4
    W = cols * CELL_W + (cols + 1) * GUTTER
    H = sum(row_heights) + (rows + 1) * GUTTER
    print(f"composite canvas: {W}x{H}  (row heights: {row_heights})  "
          f"{N_OUT} frames @ {FPS} fps")

    idx_map = [[sample_indices(len(all_frames[r][c]), N_OUT) for c in range(cols)]
               for r in range(rows)]

    out_frames = []
    for t in range(N_OUT):
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        y = GUTTER
        for r in range(rows):
            x = GUTTER
            for c in range(cols):
                src_idx = idx_map[r][c][t]
                cell = all_frames[r][c][src_idx].resize((CELL_W, row_heights[r]), Image.LANCZOS)
                canvas.paste(cell, (x, y))
                x += CELL_W + GUTTER
            y += row_heights[r] + GUTTER
        out_frames.append(canvas)
        if t % 16 == 0:
            print(f"  rendered frame {t}/{N_OUT}")

    out_path = Path("/projects/u6gs/spikefly.u6gs/trace_viz_gifs/composite_4x4.gif")
    print(f"saving {out_path} ...")
    out_frames[0].save(
        out_path,
        save_all=True,
        append_images=out_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"OK: {out_path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
