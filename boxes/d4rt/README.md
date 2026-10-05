# D4RT Box

4D reconstruction and tracking behind the **shared envelope** interface,
wrapping [OpenD4RT](https://github.com/Lijiaxin0111/Open-d4rt) — a feedforward
video transformer that infers depth, correspondence and camera parameters from
a single video.

The model answers exactly one kind of question: *where is the point at pixel
`(u, v)` of frame `t_src`, at frame `t_tgt`, expressed in the camera frame of
`t_cam`?* Tracking, point clouds and camera poses are all built from that one
query, which is why this box has three commands rather than three models.

| Command | What it does | State |
|---|---|---|
| `track` (default) | follow query points through the clip in 3D | none |
| `reconstruct` | per-frame point cloud from a uv grid, in frame-0 coordinates | none |
| `cameras` | per-frame intrinsics and extrinsics | none |
| `reset` | standard no-op (this box is stateless) | — |

## Build

```bash
cd boxes/d4rt
docker build --tag sipgisr/visionist-d4rt --build-arg SERVICE_NAME=d4rt -f docker/Dockerfile .
```

The build clones OpenD4RT at a **pinned commit** and bakes in the checkpoint
(~4.6 GB), so the image is self-contained and a fleet never waits on a Hugging
Face download. That makes it the heaviest image in the registry, around 12 GB.
For a thin image with the weights mounted instead:

```bash
cd boxes/d4rt
docker build --tag sipgisr/visionist-d4rt --build-arg DOWNLOAD_WEIGHTS=false --build-arg SERVICE_NAME=d4rt -f docker/Dockerfile .
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 \
  -v /path/to/weights:/weights sipgisr/visionist-d4rt
```

`/weights/<variant>/` must then hold `model.yaml` and `opend4rt.ckpt`, from
[Lijiaxin0111/OpenD4RT](https://huggingface.co/Lijiaxin0111/OpenD4RT).
`--build-arg D4RT_VARIANT=OpenD4RT_32CLIP_9Dataset_NoAUG` selects the 32-frame
checkpoint instead of the 48-frame default.

## Run

```bash
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 --ipc=host sipgisr/visionist-d4rt
```

## Service usage

### Request

```jsonc
{
  "d4rt": {
    "command": "track",          // track | reconstruct | cameras | reset
    "parameters": {
      "grid_size": 32,           // track: uv grid when no points are given
      "t_src": 0,                // track: frame the query pixels come from
      "point_grid_size": 64,     // reconstruct: grid side
      "max_points": 4096,        // reconstruct: cap after gridding
      "camera_grid_size": 16,    // cameras: grid for the fits
      "intrinsics": true,        // cameras
      "extrinsics": true,        // cameras
      "chunk_size": 4096,        // queries per decoder call
      "frame_step": 1,
      "max_frames": 48,
      "device": "cuda"
    }
  }
}
```

`data`:

| field | kind | meaning |
|---|---|---|
| `video` | `b` | one video file (mp4/avi/webm/mov) |
| `images` | `bb` | ordered frames, all the same size |
| `points` | `ff` | `track`: flat `[u0, v0, u1, v1, …]`, normalised to `[0, 1]` |

`video` and `images` are mutually exclusive. Without query points, `track`
uses a `grid_size × grid_size` grid.

### Response

```json
{
  "d4rt": {
    "status": "done",
    "command": "track",
    "num_queries": 1024,
    "query_source": "grid 32x32",
    "t_src": 0,
    "num_frames": 48,
    "clip_frames": 48,
    "input_hw": [1080, 1920],
    "model_hw": [256, 256],
    "device": "cuda",
    "variant": "OpenD4RT_48CLIP_9Mix_NoCropAUG",
    "runtime": 7.4,
    "encoding": { "tracks_xyz_ref0": "numpy", "…": "numpy" }
  }
}
```

Status vocabulary per the shared contract: `done`, `empty_request` (no frames),
`error` (reason in `"error"`).

`data` — every field is an `np.save` blob declared `numpy`:

| field | shape | when | meaning |
|---|---|---|---|
| `query_uv` | `(Q, 2)` | track, reconstruct | the normalised pixels actually queried |
| `tracks_xyz_local` | `(Q, T, 3)` | track | each point at frame `t`, in the camera frame of that same `t` |
| `tracks_xyz_ref0` | `(Q, T, 3)` | track | the same points in frame 0's camera frame — **use this for motion** |
| `tracks_uv` | `(Q, T, 2)` | track | 2D reprojection, normalised |
| `tracks_visibility` | `(Q, T)` | track | boolean, already thresholded |
| `tracks_confidence` | `(Q, T)` | track | raw scalar, for ranking |
| `points_xyz` | `(T, P, 3)` | reconstruct | point cloud per frame, frame-0 coordinates; NaN where unresolved |
| `points_visibility` / `points_confidence` | `(T, P)` | reconstruct | |
| `intrinsics` | `(T, 3, 3)` | cameras | per-frame K |
| `extrinsics` | `(T, 4, 4)` | cameras | `T_ref0_cam` per frame |
| `valid_intrinsics` / `valid_extrinsics` | `(T,)` | cameras | boolean; a frame whose fit failed is identity and marked false |

## Semantics worth knowing

- **Predictions are up to scale, not metric.** Every upstream evaluation
  aligns them to ground truth with a similarity transform before measuring, so
  do not read `xyz` as metres. Ratios and shapes are meaningful; absolute
  distances are not.
- **`tracks_xyz_local` vs `tracks_xyz_ref0`.** The local one re-expresses each
  frame in its own camera, so a static point appears to move when the camera
  does. For object motion, use `ref0`.
- **The clip limit is real.** The model sees at most `clip_frames` frames at
  once — 48 for the default checkpoint, 32 for the other. Longer inputs are
  handled by upstream's clip anchoring, but staying within one clip is faster
  and more accurate; `max_frames` defaults to the clip length for that reason.
  Upstream *clamps* out-of-range timesteps rather than rejecting them, so an
  over-long video yields confident wrong answers rather than an error.
- **Frames are resized to 256×256** (what the checkpoints were trained at).
  The true aspect ratio is passed separately, so non-square input is fine.
- **A mismatched checkpoint is refused.** Upstream loads with `strict=False`,
  which turns a config/checkpoint mismatch into a model that runs and returns
  nonsense; this box checks the missing and unexpected key lists and fails to
  serve instead.
- **GPU in practice.** There are no custom CUDA kernels and the code runs on
  CPU, but at ~1.16B parameters in fp32 that is not usable. The model loads
  lazily and a watchdog parks it back on CPU after ~120 s idle, so an idle box
  holds no VRAM.
- **`reconstruct` is a dense *grid*, not every pixel.** Upstream's
  `dense_tracking: occupancy_grid_tracking_all_pixels` config key is never read
  by any code.

## Call with visionist_client

```python
from visionist_client import Visionist
import pathlib, numpy as np

b = Visionist("localhost:8061", timeout=1800)

# --- track a grid of points through a clip -----------------------------
res = b.run(data   = {"video": pathlib.Path("clip.mp4")},
            config = {"d4rt": {"command": "track",
                               "parameters": {"grid_size": 32, "max_frames": 48}}})
xyz = res.tracks_xyz_ref0          # (Q, T, 3) ndarray
vis = res.tracks_visibility        # (Q, T) bool
moved = np.linalg.norm(xyz[:, -1] - xyz[:, 0], axis=1)
print(f"{vis.all(axis=1).sum()} points visible throughout; "
      f"largest displacement {moved.max():.3f}")

# --- track specific pixels --------------------------------------------
res = b.run(data   = {"video": pathlib.Path("clip.mp4"),
                      "points": [0.5, 0.5, 0.25, 0.75]},   # two points, (u, v)
            config = {"d4rt": {"command": "track"}})

# --- point cloud and cameras ------------------------------------------
rec = b.run(data   = {"video": pathlib.Path("clip.mp4")},
            config = {"d4rt": {"command": "reconstruct",
                               "parameters": {"point_grid_size": 64}}})
cams = b.run(data  = {"video": pathlib.Path("clip.mp4")},
             config = {"d4rt": {"command": "cameras"}})
K = cams.intrinsics[0]             # (3, 3)
```

## Testing

```bash
# contract only, no GPU and no weights: the model is stubbed out
cd boxes/d4rt && python3 test/smoke_inprocess.py

# against a running box
python3 test/test_d4rt.py
BOX_HOST=10.0.0.5:8061 python3 test/test_d4rt.py

# the shared contract test every box must pass
python3 ../../contract/conformance/test_conformance.py --box d4rt --host localhost:8061
```

## Upstream

[Lijiaxin0111/Open-d4rt](https://github.com/Lijiaxin0111/Open-d4rt), pinned by
commit in `docker/Dockerfile`. This box imports helpers from the repo's
scripts (`_infer_tracks`, `_infer_point_cloud_ref0`,
`_predict_camera_branches`) — private functions with no stability promise,
which is why the pin exists. Re-test the box when you move it.

Checkpoints: [Lijiaxin0111/OpenD4RT](https://huggingface.co/Lijiaxin0111/OpenD4RT).
Paper and project page: [d4rt-paper.github.io](https://d4rt-paper.github.io/).
Licence: Apache-2.0.
