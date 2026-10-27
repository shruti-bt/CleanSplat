# Rendering a video of a Gaussian splat

`render_video.py` makes an `.mp4` of a splat, for example a file cleaned with `clean_splat.py` (see [README.md](README.md)). It renders the splat for real on the GPU (with [gsplat](https://github.com/nerfstudio-project/gsplat)), not as dots. The camera starts with the whole object in view, swings back and forth around it, and zooms in slightly half way through.

It reads the same formats as `clean_splat.py`: standard `.ply`, SuperSplat compressed `.ply`, and `.splat`. A 15 s 1080p video takes about 10 s on an RTX 4090.

## 1. Install

It needs a CUDA GPU, ffmpeg, and its own conda environment (so the cleaning environment `sgo-test` is not changed):

```bash
conda create -y -n splat-render python=3.10
conda activate splat-render
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy gsplat --extra-index-url https://docs.gsplat.studio/whl/pt24cu124
```

`render_video.py` imports the file reader from `clean_splat.py`, so keep both files in the same folder.

## 2. Run

```bash
conda activate splat-render
cd ~/ShipHull/SplatCleaning

# 1. check the first frame: is the object the right way up?
python render_video.py cleaned.ply cleaned.mp4 --auto --still check.png
python render_video.py cleaned.ply cleaned.mp4 --auto --still check.png --flip      # if it was upside down

# 2. render the video (add --flip if step 1 needed it)
python render_video.py cleaned.ply cleaned.mp4 --auto
```

You can also choose the view yourself instead of using `--auto`:

```bash
python render_video.py cleaned.ply cleaned.mp4 --flip --start 180 --elevation -50
```

## 3. Choosing the view

### Automatically: `--auto`

It renders about 200 angles at low resolution and scores each by how sharp the image is: the sides the cameras saw are sharp, unseen sides look blurry or hazy. The video starts at the best angle and swings around it only as far as the views stay good. It prints the values it chose, for example:

```
auto view: --start 345 --elevation 0 --arc 120
```

Copy them into the command without `--auto` to fine-tune by hand.

Limits: it cannot tell up from down, so check with `--still` and add `--flip` if needed. Bright glare also counts as "sharp", so a scene with a lot of glare may get a view with a lot of glare.

### By hand

Change one option at a time and check with `--still` until the first frame looks right, then render the video with the same options.

| What you see in the still | Add |
|---|---|
| upside down | `--flip` |
| wrong side of the object | `--start 90` (or `180`, `-90`) |
| too much from above | `--elevation 0` |
| want to look from below | negative elevation, e.g. `--elevation -30` |
| object too small | `--fov 35` (smaller number = bigger object) |
| object cut off at the edges | `--fov 65` |
| tilted, and `--flip` does not help | `--up y`, `-y`, `z`, `-z`, `x` or `-x` (COLMAP scenes are often `-y`) |

A good order: first the right way up (`--flip` / `--up`), then the side (`--start`), then the height (`--elevation`), then the size (`--fov`).

The still shows only the **first** frame. The video swings `--arc / 2` degrees to each side of it and zooms in half way.

## 4. All options

| Option (default) | What it does |
|---|---|
| `--auto` | choose `--start`, `--elevation` and `--arc` from the sharpest views |
| `--duration 15` | length, seconds |
| `--fps 30` | frames per second |
| `--size 1920x1080` | resolution, `WIDTHxHEIGHT` |
| `--arc 120` | swing back and forth over this many degrees; `360` = full orbit (only for objects scanned from all sides) |
| `--start 0` | angle around the object at the start (the centre of the swing), degrees |
| `--elevation 20` | camera height above the object's middle plane, degrees; negative = from below |
| `--zoom 0.25` | zoom-in half way through: `0` = none, `0.25` = 1.3×, `0.5` = 2×, `0.75` = 4× |
| `--fov 50` | field of view, degrees |
| `--flip` | turn the view upside down |
| `--up auto` | which world axis is up: `auto`, `x`, `y`, `z`, `-x`, `-y`, `-z` |
| `--bg 0,0,0` | background colour, r,g,b in 0-1 |
| `--still file.png` | render only the first frame, to check the view |

## 5. How it works

- **Object frame.** The centre and size of the object come from its most opaque splats. "Up" is the direction in which the object is thinnest (the vertical for a ship hull). Its sign is a guess, hence `--flip`.
- **Camera path.** The camera stays at a fixed distance that fits the object in view and swings around it (`--arc`). The zoom narrows the field of view like a zoom lens rather than moving the camera, so the camera never ends up inside the object.
- **Why a swing and not a full orbit.** Our scans cover one side of the object. From the uncaptured side, a full orbit shows the inside of the surface and haze.
- **Colours.** View-dependent colour (spherical harmonics) is used when the file has it.

## 6. Limitations

- Only the object's shape is used to choose the view, not the real camera positions from capture. If the training cameras (`cameras.json` from 3DGS, or COLMAP `sparse/0`) are kept, a path along the real ROV trajectory would give the best views; this is not implemented yet.
- Needs an NVIDIA GPU with CUDA.
