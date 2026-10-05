#!/usr/bin/env python3
"""Test the d4rt box against a running instance.

Needs the real model, so this wants a GPU box with the checkpoint. For the
contract alone, with no weights and no GPU, use test/smoke_inprocess.py.

    python3 test/test_d4rt.py
    BOX_HOST=10.0.0.5:8061 python3 test/test_d4rt.py
    D4RT_VIDEO=/path/clip.mp4 python3 test/test_d4rt.py

Without D4RT_VIDEO a short synthetic clip is generated: a bright square
translating across a textured background. That is enough to check shapes,
the declared encoding and the error contract, and enough for one weak sanity
check on the geometry - the moving square's points should displace more than
the static background's.
"""

import io
import json
import os
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_TEST_DIR, "..", "protos"))

import grpc            # noqa: E402
import numpy as np     # noqa: E402

import aux             # noqa: E402
import pipeline_pb2    # noqa: E402
import pipeline_pb2_grpc  # noqa: E402

FRAMES, SIZE = 8, 192


def synthetic_clip():
    """A moving square over static texture, as JPEG frames."""
    import cv2

    rng = np.random.default_rng(0)
    background = rng.integers(40, 200, (SIZE, SIZE, 3), dtype=np.uint8)
    for _ in range(40):                       # texture, so features exist
        x, y = rng.integers(0, SIZE - 20, 2)
        cv2.rectangle(background, (int(x), int(y)), (int(x) + 18, int(y) + 18),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    out = []
    for t in range(FRAMES):
        img = background.copy()
        x = 20 + t * 12                       # the square translates right
        cv2.rectangle(img, (x, 80), (x + 30, 110), (255, 255, 255), -1)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        out.append(buf.tobytes())
    return out


def make_stub(target):
    channel = grpc.insecure_channel(
        target, options=[("grpc.max_send_message_length", -1),
                         ("grpc.max_receive_message_length", -1)])
    return pipeline_pb2_grpc.PipelineServiceStub(channel), channel


def section(response):
    cfg = json.loads(response.config_json or "{}")
    s = cfg.get("d4rt")
    if not isinstance(s, dict):
        print(f"  ERROR: reply not namespaced under 'd4rt': {cfg}")
        return None
    if s.get("status") == "error":
        print(f"  ERROR: {s.get('error')}")
    return s


def decode(value):
    return np.load(io.BytesIO(bytes(value)), allow_pickle=False)


def main() -> int:
    target = os.getenv("BOX_HOST", "localhost:8061")
    print(f"Target: {target}")
    stub, channel = make_stub(target)
    failures = []

    video_path = os.getenv("D4RT_VIDEO")
    if video_path:
        data_in = {"video": aux.wrap_value(open(video_path, "rb").read())}
        print(f"input: {video_path}")
    else:
        frames = synthetic_clip()
        data_in = {"images": aux.wrap_value(frames)}
        print(f"input: {len(frames)} synthetic frames ({SIZE}x{SIZE})")

    # ---------------------------------------------------------------- track
    print("\n== case 1: track, grid of query points ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"d4rt": {
            "command": "track",
            "parameters": {"grid_size": 12, "max_frames": FRAMES}}}),
        data=dict(data_in)))
    s = section(response)
    if s is None:
        return 1
    if s.get("status") != "done":
        failures.append("case1 status")
    else:
        expected = ("query_uv", "tracks_xyz_local", "tracks_xyz_ref0",
                    "tracks_uv", "tracks_visibility", "tracks_confidence")
        for f in expected:
            if f not in response.data:
                print(f"  missing field: {f}")
                failures.append(f"case1 {f}")
        enc = s.get("encoding", {})
        for f in expected:
            if enc.get(f) != "numpy":
                failures.append(f"case1 encoding {f}")

        if "tracks_xyz_ref0" in response.data:
            xyz = decode(aux.unwrap_value(response.data["tracks_xyz_ref0"]))
            uv = decode(aux.unwrap_value(response.data["query_uv"]))
            vis = decode(aux.unwrap_value(response.data["tracks_visibility"]))
            q, t = uv.shape[0], s.get("num_frames")
            print(f"  query_uv          {uv.shape}")
            print(f"  tracks_xyz_ref0   {xyz.shape} {xyz.dtype}")
            print(f"  visible always    {int(vis.all(axis=1).sum())} of {q}")
            print(f"  runtime {s.get('runtime'):.1f}s on {s.get('device')} "
                  f"({s.get('variant')})")
            if xyz.shape != (q, t, 3):
                print(f"  expected {(q, t, 3)}, got {xyz.shape}")
                failures.append("case1 shape")
            if not np.isfinite(xyz).any():
                print("  every coordinate is non-finite")
                failures.append("case1 all nan")

            # Weak geometry check: the square moves, the background does not.
            # Only meaningful for the synthetic clip.
            if not video_path and xyz.shape[0] == q:
                disp = np.linalg.norm(xyz[:, -1] - xyz[:, 0], axis=1)
                on_square = (uv[:, 1] > 80 / SIZE) & (uv[:, 1] < 110 / SIZE) \
                    & (uv[:, 0] < 0.35)
                if on_square.any() and (~on_square).any():
                    a = np.nanmedian(disp[on_square])
                    b = np.nanmedian(disp[~on_square])
                    print(f"  median displacement: square {a:.4f} vs rest {b:.4f}")
                    if not (np.isfinite(a) and np.isfinite(b)):
                        print("  (non-finite; skipping the comparison)")
                    elif a <= b:
                        print("  NOTE: the moving region did not displace more "
                              "than the static one - advisory, not a failure")

    # ---------------------------------------------------- explicit points
    print("\n== case 2: track, explicit query points ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"d4rt": {"command": "track",
                                         "parameters": {"max_frames": FRAMES}}}),
        data={**data_in, "points": aux.wrap_value([0.25, 0.5, 0.75, 0.5])}))
    s = section(response)
    if s is None or s.get("status") != "done" or s.get("num_queries") != 2:
        failures.append("case2 explicit points")
    else:
        print(f"  {s.get('num_queries')} queries from {s.get('query_source')}")

    # ------------------------------------------------------- reconstruct
    print("\n== case 3: reconstruct ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"d4rt": {
            "command": "reconstruct",
            "parameters": {"point_grid_size": 24, "max_points": 400,
                           "max_frames": FRAMES}}}),
        data=dict(data_in)))
    s = section(response)
    if s is None or s.get("status") != "done":
        failures.append("case3 status")
    else:
        pts = decode(aux.unwrap_value(response.data["points_xyz"]))
        print(f"  points_xyz {pts.shape}, {100*np.isfinite(pts).mean():.0f}% finite")
        if pts.ndim != 3 or pts.shape[2] != 3:
            failures.append("case3 shape")

    # ----------------------------------------------------------- cameras
    print("\n== case 4: cameras ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"d4rt": {
            "command": "cameras",
            "parameters": {"camera_grid_size": 8, "max_frames": FRAMES}}}),
        data=dict(data_in)))
    s = section(response)
    if s is None or s.get("status") != "done":
        failures.append("case4 status")
    else:
        K = decode(aux.unwrap_value(response.data["intrinsics"]))
        E = decode(aux.unwrap_value(response.data["extrinsics"]))
        okk = decode(aux.unwrap_value(response.data["valid_intrinsics"]))
        print(f"  intrinsics {K.shape}, extrinsics {E.shape}, "
              f"{int(okk.sum())}/{okk.size} frames with a valid K")
        if K.shape[1:] != (3, 3) or E.shape[1:] != (4, 4):
            failures.append("case4 shapes")

    # ------------------------------------------------- error contract
    print("\n== case 5: empty_request and errors ==")
    s = section(stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"d4rt": {"command": "track"}}))))
    if s is None or s.get("status") != "empty_request":
        failures.append("case5 empty_request")

    s = section(stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"d4rt": {"command": "explode"}}),
        data=dict(data_in))))
    if s is None or s.get("status") != "error" or "explode" not in str(s.get("error")):
        failures.append("case5 unknown command")

    s = section(stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"d4rt": {"command": "track"}}),
        data={**data_in, "points": aux.wrap_value([9.0, 9.0])})))
    if s is None or s.get("status") != "error":
        failures.append("case5 out-of-range points")

    # -------------------------------------------------------------- reset
    print("\n== case 6: reset ==")
    s = section(stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"d4rt": {"command": "reset"}}))))
    if s is None or s.get("status") != "done" or s.get("action") != "reset":
        failures.append("case6 reset")
    else:
        print(f"  ok: {s}")

    channel.close()
    print("\n" + ("FAIL -- " + "; ".join(failures) if failures
                  else "PASS -- all d4rt test cases."))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
