"""Render a video of a Gaussian splat (real splat rendering, with gsplat).

The camera swings back and forth around the object (--arc 120 degrees by default; 360 =
full orbit) while zooming in and back out, so the video shows the whole object and
then its detail. Scans usually cover one side only, and a full orbit then shows the
uncaptured back. The object frame comes from its solid splats: the camera turns around
the axis along which the object is thinnest (the vertical axis for a ship hull). The
sign of that axis is a guess: check with --still and add --flip if the view is upside
down, or set --up.

Reads every format clean_splat.py reads (.ply, compressed .ply, .splat).
Needs a CUDA GPU, PyTorch, gsplat and ffmpeg (see README).

Usage:
  python render_video.py cleaned.ply check.mp4 --still check.png      # one frame, to check the view
  python render_video.py cleaned.ply cleaned.mp4 --auto                 # start at the best view
  python render_video.py cleaned.ply cleaned.mp4 --flip --arc 360 --zoom 0.6 --duration 20
"""
import argparse
import subprocess

import numpy as np
import torch
from gsplat import rasterization

from clean_splat import SH_C0, load, attrs


def splat_tensors(vertex, dev):
    a = attrs(vertex)
    names = vertex.dtype.names
    n_rest = sum(p.startswith("f_rest_") for p in names)
    k = 1 + n_rest // 3
    sh = np.zeros((len(vertex), k, 3), np.float32)
    if "f_dc_0" in names:
        sh[:, 0] = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], 1)
    else:
        sh[:, 0] = (a["rgb"] - 0.5) / SH_C0
    if n_rest:  # INRIA layout: all coefficients of red, then green, then blue
        rest = np.stack([vertex[f"f_rest_{i}"] for i in range(n_rest)], 1).reshape(-1, 3, k - 1)
        sh[:, 1:] = rest.transpose(0, 2, 1)
    scale = a["scale"]
    if scale.shape[1] == 2:  # 2D splats: flat third axis
        scale = np.c_[scale, np.full(len(scale), 1e-6)]
    quat = a["rot"] / np.maximum(np.linalg.norm(a["rot"], axis=1, keepdims=True), 1e-12)
    t = lambda x: torch.tensor(np.asarray(x, np.float32), device=dev)
    return a, dict(means=t(a["xyz"]), quats=t(quat), scales=t(scale), opacities=t(a["opacity"]),
                   colors=t(sh)), int(round(np.sqrt(k))) - 1


def object_frame(a, up_arg):
    xyz, op = a["xyz"], a["opacity"]
    solid = xyz[op >= np.percentile(op, 90)]
    centre = np.median(solid, 0)
    _, _, axes = np.linalg.svd(solid - centre, full_matrices=False)
    if up_arg == "auto":
        up = axes[2]
    else:
        up = np.eye(3)["xyz".index(up_arg[-1])] * (-1 if up_arg[0] == "-" else 1)
    radius = np.percentile(np.linalg.norm(solid - centre, axis=1), 98)
    # start the orbit looking at the long side of the object
    side = axes[1] - up * (axes[1] @ up)
    if np.linalg.norm(side) < 1e-6:
        side = axes[0] - up * (axes[0] @ up)
    return centre, up, side / np.linalg.norm(side), radius


def look_at(eye, target, up):
    """World-to-camera matrix, OpenCV convention (x right, y down, z forward)."""
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    m = np.eye(4)
    m[:3, :3] = np.stack([r, d, f])
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def camera_path(n, centre, up, side, radius, fov, zoom, elevation, arc, start):
    """Yields (world-to-camera matrix, focal length multiplier) per frame.
    arc >= 360: orbit all the way round; otherwise swing back and forth over `arc` degrees.
    Zoom narrows the field of view (like a zoom lens), so the camera never enters the object."""
    dist = radius / np.tan(np.radians(fov) / 2) * 0.8
    other = np.cross(up, side)
    for i in range(n):
        t = i / max(n - 1, 1)
        if arc >= 360:
            az = np.radians(start + arc * t)
        else:
            az = np.radians(start + arc / 2 * np.sin(2 * np.pi * t))
        el = np.radians(elevation) * (1 + 0.5 * np.sin(2 * np.pi * t))
        direction = np.cos(el) * (np.cos(az) * side + np.sin(az) * other) + np.sin(el) * up
        focal = 1 / (1 - zoom * np.sin(np.pi * t) ** 2)  # full view -> closest at half way -> full view
        yield look_at(centre + dist * direction, centre, up), focal


def find_good_view(render_small, centre, up, side, radius, fov, step=15):
    """Render a grid of angles at low resolution and score each by image sharpness (mean squared
    gradient). Angles the capture cameras saw look sharp; unseen sides look blurry or hazy.
    Returns (start, elevation, arc): the best angle, and a swing around it (60-120 degrees) that
    stays within angles scoring at least half as well."""
    azs, els = np.arange(0, 360, step), np.arange(-60, 61, step)
    score = np.zeros((len(els), len(azs)))
    for i, el in enumerate(els):
        for j, az in enumerate(azs):
            g = render_small(next(camera_path(1, centre, up, side, radius, fov, 0, el, 0, az))).mean(-1)
            score[i, j] = ((g[:, 1:] - g[:, :-1])[:-1] ** 2 + (g[1:] - g[:-1])[:, :-1] ** 2).mean().item()
    i, j = np.unravel_index(score.argmax(), score.shape)
    good = score[i] >= 0.5 * score[i, j]
    half = 0
    while half * step < 60 and good[(j - half - 1) % len(azs)] and good[(j + half + 1) % len(azs)]:
        half += 1
    return float(azs[j]), float(els[i]), float(max(2 * half * step, 60))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", help=".mp4")
    ap.add_argument("--duration", type=float, default=15, help="seconds")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080", help="WIDTHxHEIGHT")
    ap.add_argument("--fov", type=float, default=50, help="vertical field of view, degrees")
    ap.add_argument("--zoom", type=float, default=0.15, help="zoom-in half way: 0 = none, 0.25 = 1.3x, 0.5 = 2x, 0.75 = 4x")
    ap.add_argument("--elevation", type=float, default=20, help="camera height above the object's middle plane, degrees")
    ap.add_argument("--arc", type=float, help="120 = full orbit; less = swing back and forth over this many degrees "
                                              "(default 120, or chosen by --auto)")
    ap.add_argument("--auto", action="store_true", help="choose --start, --elevation and --arc from the sharpest views")
    ap.add_argument("--start", type=float, default=0, help="starting angle of the camera around the object, degrees")
    ap.add_argument("--up", default="auto", choices=["auto", "x", "y", "z", "-x", "-y", "-z"],
                    help="world axis that points up; auto = object's thinnest axis")
    ap.add_argument("--flip", action="store_true", help="turn the view upside down (if auto 'up' is wrong)")
    ap.add_argument("--bg", default="0,0,0", help="background colour r,g,b in 0-1")
    ap.add_argument("--still", help="only render the first frame to this .png, to check the view")
    args = ap.parse_args()

    dev = "cuda"
    w, h = map(int, args.size.lower().split("x"))
    a, splats, sh_degree = splat_tensors(load(args.input).vertex, dev)
    centre, up, side, radius = object_frame(a, args.up)
    if args.flip:
        up = -up
    bg = torch.tensor([float(c) for c in args.bg.split(",")], device=dev)
    n = 1 if args.still else int(round(args.duration * args.fps))
    print(f"{len(a['xyz'])} splats, SH degree {sh_degree}, {n} frames at {w}x{h}")

    def render_float(cam, w, h):
        view, focal = cam
        f = h / 2 / np.tan(np.radians(args.fov) / 2) * focal
        K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=torch.float32, device=dev)[None]
        with torch.no_grad():
            img, alpha, _ = rasterization(viewmats=torch.tensor(view, dtype=torch.float32, device=dev)[None], Ks=K,
                                          width=w, height=h, sh_degree=sh_degree, **splats)
        return (img[0] + (1 - alpha[0]) * bg).clamp(0, 1)

    def render(cam):
        return (render_float(cam, w, h) * 255).byte().cpu().numpy()

    if args.auto:
        start, elevation, arc = find_good_view(lambda cam: render_float(cam, 320, 180), centre, up, side, radius, args.fov)
        args.start, args.elevation = start, elevation
        args.arc = arc if args.arc is None else args.arc
        print(f"auto view: --start {start:g} --elevation {elevation:g} --arc {args.arc:g}")
    if args.arc is None:
        args.arc = 120

    path = camera_path(n, centre, up, side, radius, args.fov, args.zoom, args.elevation, args.arc, args.start)
    if args.still:
        frame = render(next(path))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{w}x{h}", "-i", "-", args.still], input=frame.tobytes(), check=True)
        print(f"wrote {args.still}")
        return

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "18", args.output], stdin=subprocess.PIPE)
    with torch.no_grad():
        for i, cam in enumerate(path):
            ff.stdin.write(render(cam).tobytes())
            if (i + 1) % args.fps == 0 or i + 1 == n:
                print(f"\r  frame {i + 1}/{n}", end="", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
