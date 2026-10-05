"""In-process contract test for the d4rt box, with the model stubbed out.

The real model is ~1.16B parameters behind a 4.6 GB checkpoint, so this does
not test the geometry - it tests everything around it, which is where the
envelope bugs live: command dispatch, input decoding, the images/video
exclusion, query-point parsing and its error cases, the declared encoding,
the namespaced status vocabulary, and that every output is an np.save blob of
the documented shape.

    cd boxes/d4rt && python3 test/smoke_inprocess.py

Needs numpy and opencv-python(-headless). torch is faked, so it runs on a
laptop with no GPU and no weights. For the real thing, build the box and use
test/test_d4rt.py.
"""
import io
import json
import sys
import types

sys.path.insert(0, "src")
sys.path.insert(0, "protos")

import numpy as np
import cv2

# The box ships only pipeline.proto; the generated stubs are produced inside
# the Docker build, so generate them into a temp dir for a local run.
try:
    import pipeline_pb2  # noqa: F401
except ImportError:
    import subprocess, tempfile
    _gen = tempfile.mkdtemp(prefix="d4rt_protos_")
    try:
        subprocess.run([sys.executable, "-m", "grpc_tools.protoc", "-Iprotos",
                        f"--python_out={_gen}", f"--grpc_python_out={_gen}",
                        "protos/pipeline.proto"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit("could not generate the proto stubs: pip install grpcio-tools")
    sys.path.insert(0, _gen)

# ---------------------------------------------------------------- fake torch
# The service does `import torch` inside Process for no_grad, and
# torch_device_default() probes cuda availability. Neither needs real torch.
_torch = types.ModuleType("torch")


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_torch.no_grad = lambda: _NoGrad()
_torch.cuda = types.SimpleNamespace(is_available=lambda: False,
                                    empty_cache=lambda: None)
sys.modules.setdefault("torch", _torch)

import d4rt_service as svc                                   # noqa: E402
import pipeline_pb2                                          # noqa: E402
from aux import wrap_value, unwrap_value                     # noqa: E402

T_FRAMES, CLIP, HW = 6, 48, (256, 256)

# ------------------------------------------------------- stub the model path
svc._MODEL.get = lambda device: ("fake-model", CLIP, HW)
svc._prepare = lambda frames, image_hw: (
    np.zeros((frames.shape[0], image_hw[0], image_hw[1], 3), np.uint8),
    float(frames.shape[2]) / float(frames.shape[1]))

_calls = {}


def _fake_track(model, frames, aspect, uv, chunk, t_src):
    _calls["track"] = {"n": uv.shape[0], "chunk": chunk, "t_src": t_src,
                       "aspect": aspect}
    q, t = uv.shape[0], frames.shape[0]
    return {"tracks_xyz_local": np.zeros((q, t, 3), np.float32),
            "tracks_xyz_ref0": np.ones((q, t, 3), np.float32),
            "tracks_uv_norm": np.full((q, t, 2), 0.5, np.float32),
            "tracks_visibility": np.ones((q, t), bool),
            "tracks_confidence": np.zeros((q, t), np.float32)}


def _fake_reconstruct(model, frames, aspect, uv, chunk):
    _calls["reconstruct"] = {"n": uv.shape[0]}
    p, t = uv.shape[0], frames.shape[0]
    return (np.zeros((t, p, 3), np.float32), np.ones((t, p), bool),
            np.zeros((t, p), np.float32))


def _fake_cameras(model, frames, hw, grid, chunk, want_k, want_t):
    _calls["cameras"] = {"grid": grid, "k": want_k, "t": want_t}
    t = frames.shape[0]
    return {"K": np.tile(np.eye(3, dtype=np.float32), (t, 1, 1)),
            "T_ref0_cam": np.tile(np.eye(4, dtype=np.float32), (t, 1, 1)),
            "valid_intrinsics": np.ones((t,), bool),
            "valid_extrinsics": np.ones((t,), bool)}


svc._run_track = _fake_track
svc._run_reconstruct = _fake_reconstruct
svc._run_cameras = _fake_cameras

# ------------------------------------------------------------------ fixtures
def frame(seed=0, h=120, w=160):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


IMAGES = [frame(i) for i in range(T_FRAMES)]

srv = svc.PipelineService()
failures = []


def call(config=None, data=None):
    req = pipeline_pb2.Envelope(
        config_json=json.dumps(config) if config else "",
        data=data or {})
    return srv.Process(req, None)


def section(resp):
    return json.loads(resp.config_json or "{}").get("d4rt", {})


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def arr(resp, field):
    return np.load(io.BytesIO(bytes(unwrap_value(resp.data[field]))), allow_pickle=False)


print("== 1. track, grid of query points ==")
r = call({"d4rt": {"command": "track", "parameters": {"grid_size": 8}}},
         {"images": wrap_value(IMAGES)})
s = section(r)
print("   ", {k: v for k, v in s.items() if k != "encoding"})
check("status done", s.get("status") == "done", str(s.get("error", "")))
check("num_queries 8x8", s.get("num_queries") == 64, str(s.get("num_queries")))
check("query_source is the grid", s.get("query_source") == "grid 8x8")
for f in ("query_uv", "tracks_xyz_local", "tracks_xyz_ref0", "tracks_uv",
          "tracks_visibility", "tracks_confidence"):
    check(f"field {f}", f in r.data)
check("encoding all numpy",
      all(v == "numpy" for v in s.get("encoding", {}).values()),
      str(s.get("encoding")))
check("tracks_xyz_ref0 is (Q, T, 3)", arr(r, "tracks_xyz_ref0").shape == (64, T_FRAMES, 3),
      str(arr(r, "tracks_xyz_ref0").shape))
check("visibility is boolean", arr(r, "tracks_visibility").dtype == bool)
check("query_uv is (Q, 2)", arr(r, "query_uv").shape == (64, 2))
check("uv inside [0, 1]", float(arr(r, "query_uv").min()) >= 0
      and float(arr(r, "query_uv").max()) <= 1)
check("native aspect reached the model",
      abs(_calls["track"]["aspect"] - 160 / 120) < 1e-6, str(_calls["track"]["aspect"]))
check("frame count reported", s.get("num_frames") == T_FRAMES)
check("clip_frames reported", s.get("clip_frames") == CLIP)

print("== 2. track with explicit points, via data.points ==")
pts = [0.25, 0.25, 0.75, 0.5, 0.1, 0.9]
r = call({"d4rt": {"command": "track"}},
         {"images": wrap_value(IMAGES), "points": wrap_value(pts)})
s = section(r)
check("status done", s.get("status") == "done", str(s.get("error", "")))
check("3 queries", s.get("num_queries") == 3, str(s.get("num_queries")))
check("source is data.points", s.get("query_source") == "data.points")
check("uv round-trips", np.allclose(arr(r, "query_uv"),
                                    np.array(pts, np.float32).reshape(-1, 2)))

print("== 2b. the parameters.points convenience form ==")
r = call({"d4rt": {"command": "track", "parameters": {"points": [0.5, 0.5]}}},
         {"images": wrap_value(IMAGES)})
check("accepted", section(r).get("query_source") == "parameters.points",
      str(section(r).get("query_source")))

print("== 3. t_src is honoured and range-checked ==")
r = call({"d4rt": {"command": "track", "parameters": {"grid_size": 4, "t_src": 3}}},
         {"images": wrap_value(IMAGES)})
check("t_src reaches the model", _calls["track"]["t_src"] == 3,
      str(_calls["track"]["t_src"]))
check("t_src echoed", section(r).get("t_src") == 3)
r = call({"d4rt": {"command": "track", "parameters": {"t_src": 99}}},
         {"images": wrap_value(IMAGES)})
check("t_src past the end is an error", section(r).get("status") == "error",
      str(section(r).get("error"))[:60])

print("== 4. reconstruct ==")
r = call({"d4rt": {"command": "reconstruct",
                   "parameters": {"point_grid_size": 16, "max_points": 100}}},
         {"images": wrap_value(IMAGES)})
s = section(r)
check("status done", s.get("status") == "done", str(s.get("error", "")))
check("capped at max_points", s.get("num_points") == 100, str(s.get("num_points")))
check("points_xyz is (T, P, 3)", arr(r, "points_xyz").shape == (T_FRAMES, 100, 3),
      str(arr(r, "points_xyz").shape))
check("no track fields leaked in", "tracks_xyz_ref0" not in r.data)

print("== 5. cameras ==")
r = call({"d4rt": {"command": "cameras", "parameters": {"camera_grid_size": 8}}},
         {"images": wrap_value(IMAGES)})
s = section(r)
check("status done", s.get("status") == "done", str(s.get("error", "")))
check("intrinsics (T, 3, 3)", arr(r, "intrinsics").shape == (T_FRAMES, 3, 3))
check("extrinsics (T, 4, 4)", arr(r, "extrinsics").shape == (T_FRAMES, 4, 4))
check("valid flags boolean", arr(r, "valid_intrinsics").dtype == bool)
check("grid reached the model", _calls["cameras"]["grid"] == 8)
r = call({"d4rt": {"command": "cameras",
                   "parameters": {"intrinsics": False, "extrinsics": False}}},
         {"images": wrap_value(IMAGES)})
check("asking for neither is an error", section(r).get("status") == "error",
      str(section(r).get("error"))[:60])

print("== 6. video input ==")
import tempfile, os                                          # noqa: E402
with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
    path = fh.name
vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
for i in range(12):
    vw.write(np.random.default_rng(i).integers(0, 255, (120, 160, 3), dtype=np.uint8))
vw.release()
blob = open(path, "rb").read()
os.unlink(path)
r = call({"d4rt": {"command": "track", "parameters": {"grid_size": 4, "max_frames": 5}}},
         {"video": wrap_value(blob)})
s = section(r)
check("video decodes", s.get("status") == "done", str(s.get("error", ""))[:70])
check("max_frames respected", s.get("num_frames") == 5, str(s.get("num_frames")))

print("== 7. input contract ==")
r = call({"d4rt": {"command": "track"}},
         {"images": wrap_value(IMAGES), "video": wrap_value(blob)})
check("images + video is an error", section(r).get("status") == "error",
      str(section(r).get("error"))[:50])
r = call({"d4rt": {"command": "track"}})
check("no input is empty_request", section(r).get("status") == "empty_request",
      str(section(r)))
r = call({"d4rt": {"command": "track"}}, {"images": wrap_value([IMAGES[0]])})
check("a single frame is an error", section(r).get("status") == "error",
      str(section(r).get("error"))[:50])
mixed = [IMAGES[0], frame(9, h=64, w=64)]
r = call({"d4rt": {"command": "track"}}, {"images": wrap_value(mixed)})
check("mismatched frame sizes are an error", section(r).get("status") == "error",
      str(section(r).get("error"))[:50])
r = call({"d4rt": {"command": "track"}}, {"images": wrap_value([b"not an image"] * 2)})
check("undecodable frame is an error", section(r).get("status") == "error",
      str(section(r).get("error"))[:50])

print("== 8. query point validation ==")
r = call({"d4rt": {"command": "track"}},
         {"images": wrap_value(IMAGES), "points": wrap_value([0.1, 0.2, 0.3])})
check("odd number of values is an error", section(r).get("status") == "error",
      str(section(r).get("error"))[:60])
r = call({"d4rt": {"command": "track"}},
         {"images": wrap_value(IMAGES), "points": wrap_value([5.0, 0.2])})
check("out-of-range uv is an error", section(r).get("status") == "error",
      str(section(r).get("error"))[:60])

print("== 9. config contract ==")
check("wrong namespace", section(call({"moge": {"command": "infer"}},
                                      {"images": wrap_value(IMAGES)})).get("status") == "error")
check("no config", section(call(None, {"images": wrap_value(IMAGES)})).get("status") == "error")
r = call({"d4rt": {"command": "explode"}}, {"images": wrap_value(IMAGES)})
check("unknown command", section(r).get("status") == "error"
      and "explode" in section(r).get("error", ""))
r = call({"d4rt": {"command": "track", "parameters": {"grid_size": "lots"}}},
         {"images": wrap_value(IMAGES)})
check("bad numeric parameter", section(r).get("status") == "error",
      str(section(r).get("error"))[:50])
r = call({"d4rt": {"command": "cameras", "parameters": {"intrinsics": "maybe"}}},
         {"images": wrap_value(IMAGES)})
check("bad boolean parameter", section(r).get("status") == "error",
      str(section(r).get("error"))[:50])

print("== 10. reset ==")
s = section(call({"d4rt": {"command": "reset"}}))
check("reset is a no-op done", s.get("status") == "done" and s.get("action") == "reset",
      str(s))

print()
print("FAIL -- " + "; ".join(failures) if failures
      else "PASS -- all in-process contract cases ok")
sys.exit(1 if failures else 0)
