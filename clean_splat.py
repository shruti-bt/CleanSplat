"""Automatic cleanup of a 3D Gaussian Splatting file.

Input:  .ply (standard 3DGS layout, any property types, binary or ascii),
        SuperSplat compressed .ply, or .splat
Output: .ply or .splat, chosen by the output extension. A standard .ply input is
        written back with its original header and properties, only fewer splats.

Filters, in order (each can be disabled):
  1. opacity      - drop near-transparent splats (haze / fog)
  2. scale        - drop huge splats (percentile of the longest axis)
  3. needle       - drop splats much longer than they are wide (streaks)
  4. brightness   - optional: drop splats whose mean colour is above a limit
  5. rules        - optional: --remove / --keep value ranges on any property,
                    like the SuperSplat "Splat Data" histograms (see --stats)
  6. main body    - keep the large connected solid parts plus a margin;
                    removes detached floaters
  7. SOR          - statistical outlier removal on kNN distance

Rules: comma-separated conditions "property:low:high", all of which must hold.
A bound may be empty (no limit), a number, or a percentile of the input such as p99.
  --remove "value:0.85:,saturation::0.15"   remove bright, nearly white splats
  --remove "volume:p99.5:"                  remove the largest 0.5% by volume
  --keep "x:-5:5,y:-5:5,z:-5:5"             keep only splats inside this box
--remove and --keep can be repeated: a splat is removed if it matches any --remove
rule, and kept only if it matches at least one --keep rule.

Usage:
  python clean_splat.py in.ply --stats
  python clean_splat.py in.ply out.ply --preview cmp.png
"""
import argparse
import os
from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814
PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
             "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
SPLAT_DTYPE = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])


# ---------------------------------------------------------------- reading / writing

def read_ply(path):
    """Parse any .ply; returns header lines, {element: structured array} and unparsed tail."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a .ply file")
        header, fmt, elements = [b"ply\n"], None, []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: header has no end_header")
            header.append(line)
            s = line.decode("ascii", "replace").split()
            if not s:
                continue
            if s[0] == "format":
                fmt = s[1]
            elif s[0] == "element":
                elements.append((s[1], int(s[2]), []))
            elif s[0] == "property":
                elements[-1][2].append(None if s[1] == "list" else (s[-1], PLY_TYPES[s[1]]))
            elif s[0] == "end_header":
                break
        body = f.read()

    end = ">" if fmt == "binary_big_endian" else "<"
    lines = body.decode("ascii", "replace").splitlines() if fmt == "ascii" else None
    data, pos, tail = {}, 0, b""
    for name, n, props in elements:
        if None in props:  # list properties (e.g. faces): keep the rest of the file as is
            tail = body[pos:] if lines is None else ("\n".join(lines[pos:]) + "\n").encode()
            break
        if lines is None:
            dt = np.dtype([(p, end + t) for p, t in props])
            data[name] = np.frombuffer(body, dt, n, pos) if n else np.zeros(0, dt)
            pos += n * dt.itemsize
        else:
            arr = np.zeros(n, np.dtype([(p, t) for p, t in props]))
            if n:
                rows = np.loadtxt(lines[pos:pos + n], ndmin=2)
                for k, (p, _) in enumerate(props):
                    arr[p] = rows[:, k]
            data[name] = arr
            pos += n
    if "vertex" not in data:
        raise ValueError(f"{path}: no readable 'vertex' element")
    return SimpleNamespace(header=header, ascii=lines is not None, data=data, tail=tail)


def write_ply(path, ply, vertex):
    """Write a parsed .ply back with a filtered vertex element."""
    with open(path, "wb") as f:
        for line in ply.header:
            if line.split()[:2] == [b"element", b"vertex"]:
                line = f"element vertex {len(vertex)}\n".encode()
            f.write(line)
        for name, arr in ply.data.items():
            arr = vertex if name == "vertex" else arr
            if ply.ascii:
                if len(arr):
                    fmt = ["%d" if arr.dtype[p].kind in "iu" else "%.9g" for p in arr.dtype.names]
                    np.savetxt(f, np.column_stack([arr[p] for p in arr.dtype.names]), fmt=fmt)
            else:
                f.write(np.ascontiguousarray(arr).tobytes())
        f.write(ply.tail)


def write_standard_ply(path, vertex):
    names = vertex.dtype.names
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertex)}\n".encode())
        f.write("".join(f"property float {p}\n" for p in names).encode() + b"end_header\n")
        out = np.zeros(len(vertex), [(p, "<f4") for p in names])
        for p in names:
            out[p] = vertex[p]
        f.write(out.tobytes())


def decompress_ply(ply):
    """SuperSplat compressed .ply -> standard 3DGS vertex array (float32 fields)."""
    ch, v = ply.data["chunk"], ply.data["vertex"]
    n = len(v)
    c = ch[np.arange(n) // 256]

    def bits(x, shift, nbits):
        return ((x.astype(np.uint32) >> shift) & ((1 << nbits) - 1)) / ((1 << nbits) - 1)

    def lerp(lo, hi, t):
        return c[lo] + (c[hi] - c[lo]) * t

    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    sh = ply.data.get("sh")
    if sh is not None:
        names += list(sh.dtype.names)
    out = np.zeros(n, [(p, "f4") for p in names])

    pp, ps, pc, pr = (v[k] for k in ("packed_position", "packed_scale", "packed_color", "packed_rotation"))
    for i, (ax, sh_, nb) in enumerate([("x", 21, 11), ("y", 11, 10), ("z", 0, 11)]):
        out[ax] = lerp(f"min_{ax}", f"max_{ax}", bits(pp, sh_, nb))
        out[f"scale_{i}"] = lerp(f"min_scale_{ax}", f"max_scale_{ax}", bits(ps, sh_, nb))
    for i, (col, sh_) in enumerate([("r", 24), ("g", 16), ("b", 8)]):
        t = bits(pc, sh_, 8)
        if f"min_{col}" in ch.dtype.names:
            t = lerp(f"min_{col}", f"max_{col}", t)
        out[f"f_dc_{i}"] = (t - 0.5) / SH_C0
    a = np.clip(bits(pc, 0, 8), 1e-6, 1 - 1e-6)
    out["opacity"] = np.log(a / (1 - a))

    # rotation: 2 bits = index of the largest component, then the other three at 10 bits each
    norm = np.sqrt(2)
    small = [(bits(pr, s, 10) - 0.5) * norm for s in (20, 10, 0)]
    big = np.sqrt(np.clip(1 - sum(q * q for q in small), 0, None))
    largest = (pr.astype(np.uint32) >> 30).astype(int)
    q = np.zeros((n, 4))
    for k in range(4):
        rows = largest == k
        others = [j for j in range(4) if j != k]
        q[rows, k] = big[rows]
        for j, s in zip(others, small):
            q[rows, j] = s[rows]
    for k in range(4):
        out[f"rot_{k}"] = q[:, k]

    if sh is not None:
        for p in sh.dtype.names:
            t = sh[p].astype(np.float32)
            out[p] = np.where(t == 0, 0, ((t + 0.5) / 256 - 0.5) * 8)
    return out


def read_splat(path):
    s = np.fromfile(path, SPLAT_DTYPE)
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    out = np.zeros(len(s), [(p, "f4") for p in names])
    for i, ax in enumerate("xyz"):
        out[ax] = s["pos"][:, i]
        out[f"scale_{i}"] = np.log(np.maximum(s["scale"][:, i], 1e-12))
        out[f"f_dc_{i}"] = (s["rgba"][:, i] / 255 - 0.5) / SH_C0
    a = np.clip(s["rgba"][:, 3] / 255, 1e-6, 1 - 1e-6)
    out["opacity"] = np.log(a / (1 - a))
    for k in range(4):
        out[f"rot_{k}"] = (s["rot"][:, k].astype(np.float32) - 128) / 128
    return out


def write_splat(path, a, keep):
    s = np.zeros(int(keep.sum()), SPLAT_DTYPE)
    scale = a["scale"][keep]
    s["pos"] = a["xyz"][keep]
    s["scale"][:, :scale.shape[1]] = scale
    s["scale"][:, scale.shape[1]:] = 1e-6  # 2D splats: flat third axis
    s["rgba"][:, :3] = np.round(a["rgb"][keep] * 255)
    s["rgba"][:, 3] = np.round(a["opacity"][keep] * 255)
    q = a["rot"][keep]
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    s["rot"] = np.clip(np.round(q * 128 + 128), 0, 255)
    order = np.argsort(-(np.prod(s["scale"], 1) * s["rgba"][:, 3]))  # .splat viewers expect big/opaque first
    s[order].tofile(path)


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".splat":
        return SimpleNamespace(ply=None, vertex=read_splat(path))
    if ext == ".spz" or ext == ".ksplat" or ext == ".sog":
        raise ValueError(f"{ext} is not supported: export the scene as .ply from SuperSplat first")
    ply = read_ply(path)
    if "chunk" in ply.data and "packed_position" in ply.data["vertex"].dtype.names:
        return SimpleNamespace(ply=None, vertex=decompress_ply(ply))
    return SimpleNamespace(ply=ply, vertex=ply.data["vertex"])


def save(path, sp, a, keep):
    if os.path.splitext(path)[1].lower() == ".splat":
        write_splat(path, a, keep)
    elif sp.ply is not None:
        write_ply(path, sp.ply, sp.vertex[keep])
    else:
        write_standard_ply(path, sp.vertex[keep])


# ---------------------------------------------------------------- splat properties

def attrs(v):
    """Activated values from a 3DGS vertex array: positions, opacity 0-1, linear scales, rgb 0-1."""
    names = v.dtype.names
    scales = sorted(p for p in names if p.startswith("scale_"))
    if not {"x", "y", "z"} <= set(names) or not scales:
        raise ValueError("not a Gaussian splat: needs x, y, z and scale_* properties, found " + ", ".join(names))
    f = lambda p: v[p].astype(np.float64)
    xyz = np.stack([f("x"), f("y"), f("z")], 1)
    opacity = 1 / (1 + np.exp(-f("opacity"))) if "opacity" in names else np.ones(len(v))
    scale = np.exp(np.stack([f(p) for p in scales], 1))
    if "f_dc_0" in names:
        rgb = 0.5 + SH_C0 * np.stack([f(f"f_dc_{i}") for i in range(3)], 1)
    elif "red" in names:
        rgb = np.stack([f(p) for p in ("red", "green", "blue")], 1)
        rgb = rgb / 255 if v.dtype["red"].kind in "iu" else rgb
    else:
        rgb = np.full((len(v), 3), 0.5)
    if "rot_0" in names:
        rot = np.stack([f(f"rot_{i}") for i in range(4)], 1)
    else:
        rot = np.tile([1.0, 0, 0, 0], (len(v), 1))
    return dict(xyz=xyz, opacity=opacity, scale=scale, rgb=np.clip(rgb, 0, 1), rot=rot)


def properties(a):
    """Per-splat values usable in --remove / --keep rules (names follow SuperSplat's Splat Data panel)."""
    xyz, rgb, s = a["xyz"], a["rgb"], np.sort(a["scale"], 1)
    r, g, b = rgb.T
    mx, mn = rgb.max(1), rgb.min(1)
    d = mx - mn
    hue = np.zeros(len(rgb))
    nz = d > 1e-9
    rmax, gmax = nz & (mx == r), nz & (mx == g) & (mx != r)
    bmax = nz & ~rmax & ~gmax
    hue[rmax] = ((g - b)[rmax] / d[rmax]) % 6
    hue[gmax] = (b - r)[gmax] / d[gmax] + 2
    hue[bmax] = (r - g)[bmax] / d[bmax] + 4

    solid = a["opacity"] >= 0.5
    centre = np.median(xyz[solid] if solid.sum() > 10 else xyz, 0)
    if s.shape[1] == 3:
        pw = 1.6075  # Knud Thomsen's ellipsoid surface approximation
        area = 4 * np.pi * (((s[:, 0] * s[:, 1]) ** pw + (s[:, 0] * s[:, 2]) ** pw + (s[:, 1] * s[:, 2]) ** pw) / 3) ** (1 / pw)
    else:
        area = np.pi * np.prod(s, 1)
    p = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "opacity": a["opacity"],
         "red": r, "green": g, "blue": b}
    p.update({f"scale_{ax}": a["scale"][:, i] for i, ax in enumerate("xyz"[:s.shape[1]])})
    p.update({
        "scale_max": s[:, -1], "scale_min": s[:, 0],
        "aniso": s[:, -1] / np.maximum(s[:, -2], 1e-12),  # longest / second-longest: needles high, flat discs not
        "volume": 4 / 3 * np.pi * np.prod(s, 1),
        "surface_area": area,
        "hue": (hue * 60) % 360, "saturation": np.where(mx > 0, d / np.maximum(mx, 1e-12), 0), "value": mx,
        "lum": rgb.mean(1),
        "distance": np.linalg.norm(xyz - centre, axis=1),
    })
    return p, centre


def parse_rule(text, props):
    conds = []
    for part in text.split(","):
        name, lo, hi = (part.split(":") + ["", ""])[:3]
        name = name.strip().lower()
        if name not in props:
            raise SystemExit(f"unknown property '{name}' in rule '{text}'; choose from: {', '.join(props)}")
        v = props[name]

        def bound(t):
            t = t.strip()
            if not t:
                return None
            return float(np.percentile(v, float(t[1:]))) if t[0] in "pP" else float(t)
        conds.append((name, bound(lo), bound(hi)))
    return conds


def rule_mask(conds, props):
    m = np.ones(len(props["x"]), bool)
    for name, lo, hi in conds:
        if lo is not None:
            m &= props[name] >= lo
        if hi is not None:
            m &= props[name] <= hi
    return m


def print_stats(props, centre):
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    print(f"{len(props['x'])} splats; 'distance' is measured from {np.round(centre, 3).tolist()}")
    print(f"{'property':<13}" + "".join(f"{('min' if q == 0 else 'max' if q == 100 else f'p{q}'):>11}" for q in qs))
    for k, v in props.items():
        print(f"{k:<13}" + "".join(f"{x:>11.4g}" for x in np.percentile(v, qs)))


# ---------------------------------------------------------------- spatial filters

def main_body_mask(xyz, opacity, solid_opacity, voxel, dilate, min_frac):
    """26-connected groups of voxels holding a solid splat, keeping every group with at least
    `min_frac` of the largest group's voxels (so an object split in two survives), grown by
    `dilate` voxels. Works on occupied voxels only, so memory does not depend on the scene extent."""
    solid = opacity >= solid_opacity
    if not solid.any():
        return np.ones(len(xyz), bool)
    idx = np.floor((xyz - xyz[solid].min(0)) / voxel).astype(np.int64)
    cells = np.unique(idx[solid], axis=0)
    pairs = cKDTree(cells).query_pairs(1, p=np.inf, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells)))
    _, label = connected_components(graph, directed=False)
    sizes = np.bincount(label)
    body = cells[sizes[label] >= min_frac * sizes.max()]
    dist, _ = cKDTree(body).query(idx, p=np.inf, distance_upper_bound=dilate + 0.5, workers=-1)
    return np.isfinite(dist)


def sor_mask(xyz, k, std_ratio):
    d, _ = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    mean_d = d[:, 1:].mean(1)
    return mean_d < mean_d.mean() + std_ratio * mean_d.std()


def preview(path, a, keep, title_b, title_a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz_b, op_b, rgb_b = a["xyz"], a["opacity"], a["rgb"]
    core = xyz_b[op_b >= np.percentile(op_b, 95)]
    centre = core.mean(0)
    _, _, axes = np.linalg.svd(core - centre, full_matrices=False)
    views = [("top", (0, 1)), ("side", (0, 2)), ("end", (1, 2))]

    fig, ax = plt.subplots(2, 3, figsize=(18, 10), facecolor="black")
    for row, (xyz, op, rgb, title) in enumerate(
        [(xyz_b, op_b, rgb_b, title_b), (xyz_b[keep], op_b[keep], rgb_b[keep], title_a)]
    ):
        p = (xyz - centre) @ axes.T
        order = np.argsort(op)
        col = np.concatenate([rgb[order], op[order, None]], 1)
        for c, (name, (i, j)) in enumerate(views):
            a_ = ax[row, c]
            a_.set_facecolor("black")
            a_.scatter(p[order, i], p[order, j], c=col, s=0.5, linewidths=0)
            a_.set_title(f"{title} - {name}", color="white")
            a_.set_aspect("equal")
            a_.axis("off")
    # share limits per view (0.5-99.5 pct of input) so before/after are directly comparable
    p = (xyz_b - centre) @ axes.T
    lo, hi = np.percentile(p, [0.5, 99.5], axis=0)
    pad = 0.05 * (hi - lo)
    for c, (_, (i, j)) in enumerate(views):
        for r in range(2):
            ax[r, c].set_xlim(lo[i] - pad[i], hi[i] + pad[i])
            ax[r, c].set_ylim(lo[j] - pad[j], hi[j] + pad[j])
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="black")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", help=".ply or .splat (not needed with --stats)")
    ap.add_argument("--stats", action="store_true", help="print the range of every property and exit")
    ap.add_argument("--min-opacity", type=float, default=0.05, help="0 disables")
    ap.add_argument("--max-scale-pct", type=float, default=99.0, help="drop splats above this percentile of longest axis; 100 disables")
    ap.add_argument("--max-aniso", type=float, default=10.0, help="max longest/second-longest axis ratio; 0 disables")
    ap.add_argument("--max-lum", type=float, default=0.0, help="drop splats whose mean colour (0-1) is above this; 0 disables")
    ap.add_argument("--remove", action="append", default=[], metavar="RULE", help='e.g. "value:0.85:,saturation::0.15" (repeatable)')
    ap.add_argument("--keep", action="append", default=[], metavar="RULE", help='e.g. "x:-5:5,y:-5:5,z:-5:5" (repeatable)')
    ap.add_argument("--body-voxel", type=float, default=0.0, help="voxel size for main-body filter, 0 = auto, -1 disables")
    ap.add_argument("--body-opacity", type=float, default=0.3, help="opacity that counts as solid surface")
    ap.add_argument("--body-dilate", type=int, default=2, help="margin in voxels around main body")
    ap.add_argument("--body-min-frac", type=float, default=0.25, help="also keep solid parts at least this fraction of the largest; 1 = largest only")
    ap.add_argument("--sor-k", type=int, default=16, help="0 disables")
    ap.add_argument("--sor-std", type=float, default=2.0)
    ap.add_argument("--preview", help="write a before/after PNG")
    args = ap.parse_args()
    if not args.stats and not args.output:
        ap.error("output is required unless --stats is given")

    sp = load(args.input)
    a = attrs(sp.vertex)
    props, centre = properties(a)
    if args.stats:
        print_stats(props, centre)
        return
    removes = [parse_rule(r, props) for r in args.remove]
    keeps = [parse_rule(r, props) for r in args.keep]

    n = len(sp.vertex)
    keep = np.ones(n, bool)

    def apply(name, mask):
        nonlocal keep
        removed = keep & ~mask
        keep &= mask
        print(f"  {name:<12} -{removed.sum():>8}  ({removed.sum() / max(n, 1):6.1%})  left {keep.sum()}")

    def apply_subset(name, fn):
        idx = np.flatnonzero(keep)
        m = np.zeros(n, bool)
        m[idx[fn(idx)]] = True
        apply(name, m)

    print(f"input: {n} splats")
    if args.min_opacity > 0:
        apply("opacity", props["opacity"] >= args.min_opacity)
    if args.max_scale_pct < 100:
        apply("scale", props["scale_max"] <= np.percentile(props["scale_max"], args.max_scale_pct))
    if args.max_aniso > 0:
        apply("needle", props["aniso"] <= args.max_aniso)
    if args.max_lum > 0:
        apply("brightness", props["lum"] <= args.max_lum)
    for i, (text, conds) in enumerate(zip(args.remove, removes), 1):
        print(f"  remove{i}: {text}")
        apply(f"remove{i}", ~rule_mask(conds, props))
    if keeps:
        apply("keep", np.any([rule_mask(c, props) for c in keeps], 0))
    if args.body_voxel >= 0 and keep.any():
        voxel = args.body_voxel or float(np.median(props["scale_max"][keep]) * 4)
        apply_subset(f"body@{voxel:.3g}", lambda i: main_body_mask(a["xyz"][i], a["opacity"][i], args.body_opacity, voxel, args.body_dilate, args.body_min_frac))
    if args.sor_k > 0 and keep.sum() > args.sor_k:
        apply_subset("SOR", lambda i: sor_mask(a["xyz"][i], args.sor_k, args.sor_std))

    save(args.output, sp, a, keep)
    print(f"wrote {keep.sum()} splats to {args.output}")

    if args.preview:
        preview(args.preview, a, keep, f"before ({n})", f"after ({keep.sum()})")
        print(f"preview: {args.preview}")


if __name__ == "__main__":
    main()
