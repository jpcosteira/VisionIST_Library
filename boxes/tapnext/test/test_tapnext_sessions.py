#!/usr/bin/env python3
"""
In-process multi-session tests for the TAPNext box.

Proves the session layer of `tapnext_service.Process()`:
  * isolation — different `session_id`s hold disjoint state
  * scoped reset — resetting one session leaves others intact
  * back-compat — requests without `session_id` still use the `default` session
  * concurrency — parallel sessions finish with exactly their own frame counts
  * TTL reaper — idle sessions are dropped (and their state is fresh after)
  * `list` operator command
  * declared `encoding` contract preserved

No GPU, no `tapnet` wheel, no Docker: a deterministic stub model is injected
via `PipelineService(model_factory=...)`. The stub encodes each session's
progress in the returned coordinates, so any cross-session leakage is visible
in the numbers.

Run:
    cd images/tapnext_tracker/test && python test_tapnext_sessions.py
"""

import io
import json
import os
import sys
import threading
import time

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BOX_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, os.path.join(BOX_DIR, "protos"))
sys.path.insert(0, os.path.join(BOX_DIR, "src"))

import numpy as np
import cv2
import torch

import pipeline_pb2 as pb2
import aux
import tapnext_service as ts


class StubTAPNext:
    """Mimics the TAPNext call signature used by the service.

    State is a plain dict {"counter", "n"} unique per session. The stub
    returns tracks filled with `counter` (0 on the init call, then incremented
    per step), so the response coordinates directly encode how far *this*
    session has advanced.
    """

    def __init__(self, device="cpu"):
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, video=None, query_points=None, state=None):
        n = int(query_points.shape[1]) if query_points is not None else state["n"]
        counter = 0 if state is None else state["counter"] + 1
        tracks = torch.full((1, 1, n, 2), float(counter), dtype=torch.float32)
        track_logits = torch.zeros((1, 1, n), dtype=torch.float32)
        visible_logits = torch.ones((1, 1, n), dtype=torch.float32)
        return tracks, track_logits, visible_logits, {"counter": counter, "n": n}


def make_service(**kw):
    kw.setdefault("model_factory", lambda device: StubTAPNext(device))
    return ts.PipelineService(**kw)


def make_frame_bytes():
    img = (np.random.rand(80, 100, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def track(svc, sid=None, frames=1, grid=4):
    cfg = {"tapnext": {"command": "track", "parameters": {"grid_size": grid}}}
    if sid is not None:
        cfg["tapnext"]["session_id"] = sid
    req = pb2.Envelope(
        config_json=json.dumps(cfg),
        data={"images": aux.wrap_value([make_frame_bytes() for _ in range(frames)])},
    )
    return svc.Process(req, None)


def reset(svc, sid=None):
    cfg = {"tapnext": {"command": "reset"}}
    if sid is not None:
        cfg["tapnext"]["session_id"] = sid
    return svc.Process(pb2.Envelope(config_json=json.dumps(cfg)), None)


def cfg_of(resp):
    return json.loads(resp.config_json)["tapnext"]


def last_tracks(resp):
    t = torch.load(io.BytesIO(aux.unwrap_value(resp.data["tracks"])), weights_only=False)
    return t.numpy()


# The service scales tracks from the 256x256 net input back to the original
# frame resolution (here 80h x 100w), so the stub's per-frame counter `c`
# comes back as [c * 80/256, c * 100/256].
FRAME_H, FRAME_W = 80, 100
EXPECTED = lambda counter: np.array([counter * FRAME_H / 256.0, counter * FRAME_W / 256.0])


FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  \u2713 {name}")
    else:
        FAILS.append(name)
        print(f"  \u2717 {name}  {detail}")


def test_isolation():
    print("1. session isolation (A=2 frames, B=1 frame)")
    svc = make_service()
    rA1 = track(svc, "alice", frames=2)
    rB1 = track(svc, "bob", frames=1)
    check("A: status done, session echo", cfg_of(rA1)["status"] == "done" and cfg_of(rA1)["session"] == "alice")
    check("A: encoded 'torch' declared", cfg_of(rA1)["encoding"]["tracks"] == "torch")
    check("A: frames_processed == 2", cfg_of(rA1)["frames_processed"] == 2,
          f"got {cfg_of(rA1)['frames_processed']}")
    check("A: last tracks == scaled counter 1 (own progress)",
          np.allclose(last_tracks(rA1)[-1], EXPECTED(1.0), atol=1e-6),
          f"got {last_tracks(rA1)[-1][:4]}")
    check("B: frames_processed == 1 (not A's history)", cfg_of(rB1)["frames_processed"] == 1,
          f"got {cfg_of(rB1)['frames_processed']}")
    check("B: last tracks == scaled counter 0 (own init, not A's state)",
          np.allclose(last_tracks(rB1)[-1], EXPECTED(0.0), atol=1e-9),
          f"got {last_tracks(rB1)[-1][:4]}")


def test_scoped_reset():
    print("2. scoped reset (reset alice; bob must continue)")
    svc = make_service()
    track(svc, "alice", frames=2)
    track(svc, "bob", frames=1)
    r = reset(svc, "alice")
    check("reset: scoped to alice", cfg_of(r)["action"] == "reset" and cfg_of(r)["session"] == "alice")

    rA2 = track(svc, "alice", frames=1)
    check("A: restarted fresh (frames_processed == 1)", cfg_of(rA2)["frames_processed"] == 1,
          f"got {cfg_of(rA2)['frames_processed']}")
    check("A: re-initialized (tracks == scaled counter 0)",
          np.allclose(last_tracks(rA2)[-1], EXPECTED(0.0), atol=1e-9))

    rB2 = track(svc, "bob", frames=1)
    check("B: intact across A's reset (frames_processed == 2)", cfg_of(rB2)["frames_processed"] == 2,
          f"got {cfg_of(rB2)['frames_processed']}")
    check("B: state kept (tracks == scaled counter 1, not re-init)",
          np.allclose(last_tracks(rB2)[-1], EXPECTED(1.0), atol=1e-6),
          f"got {last_tracks(rB2)[-1][:4]}")


def test_default_session_backcompat():
    print("3. back-compat: no session_id -> 'default' session")
    svc = make_service()
    r1 = track(svc, None, frames=1)
    check("default: works, echo 'default'", cfg_of(r1)["session"] == "default")
    r2 = track(svc, None, frames=1)
    check("default: accumulates (frames_processed == 2)", cfg_of(r2)["frames_processed"] == 2,
          f"got {cfg_of(r2)['frames_processed']}")
    rB = track(svc, "bob", frames=1)
    check("named session unaffected by default's history", cfg_of(rB)["frames_processed"] == 1)
    r3 = track(svc, None, frames=1)
    check("default: still accumulating after bob's request (== 3)", cfg_of(r3)["frames_processed"] == 3,
          f"got {cfg_of(r3)['frames_processed']}")


def test_concurrency():
    print("4. concurrency: 8 sessions x 5 frames in parallel")
    svc = make_service()
    outcomes = {}
    errors = []

    def worker(sid):
        try:
            last = None
            for _ in range(5):
                last = track(svc, sid, frames=1)
            outcomes[sid] = last
        except Exception as e:  # noqa: BLE001 - test harness
            errors.append(f"{sid}: {e!r}")

    threads = [threading.Thread(target=worker, args=(f"stu{i:02d}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    check("no exceptions during concurrent tracking", not errors, str(errors[:3]))
    check("all 8 sessions present", set(outcomes) == {f"stu{i:02d}" for i in range(8)},
          f"sizes={ {k: cfg_of(v)['frames_processed'] for k, v in outcomes.items()} }")
    ok_counts = all(cfg_of(r)["frames_processed"] == 5 for r in outcomes.values())
    check("every session ended at exactly its own 5 frames", ok_counts,
          str({k: cfg_of(v)["frames_processed"] for k, v in outcomes.items() if cfg_of(v)["frames_processed"] != 5}))
    ok_vals = all(np.allclose(last_tracks(r)[-1], EXPECTED(4.0), atol=1e-6) for r in outcomes.values())
    check("every session's final progress == scaled counter 4 (no shared counters)", ok_vals)


def test_reaper():
    print("5. TTL reaper (ttl=0.3s, watchdog=0.1s)")
    svc = make_service(session_ttl=0.3, watchdog_interval=0.1)
    r1 = track(svc, "temp", frames=1)
    check("session alive right after use", cfg_of(r1)["status"] == "done" and "temp" in svc._sessions)
    time.sleep(1.2)
    check("session reaped after TTL", "temp" not in svc._sessions)
    r2 = track(svc, "temp", frames=1)
    check("recreated fresh (frames_processed == 1)", cfg_of(r2)["frames_processed"] == 1,
          f"got {cfg_of(r2)['frames_processed']}")


def test_list():
    print("6. list (operator view)")
    svc = make_service()
    track(svc, "alpha", frames=2)
    track(svc, "beta", frames=1)
    req = pb2.Envelope(config_json=json.dumps({"tapnext": {"command": "list"}}))
    r = svc.Process(req, None)
    c = json.loads(r.config_json)["tapnext"]
    by_sid = {s["session"]: s for s in c.get("sessions", [])}
    check("list: status done, action list", c.get("status") == "done" and c.get("action") == "list")
    check("list: sees alpha with 2 frames", by_sid.get("alpha", {}).get("frames_processed") == 2,
          str(c.get("sessions")))
    check("list: sees beta with 1 frame", by_sid.get("beta", {}).get("frames_processed") == 1)
    check("list: config-only response (no data leak)", not r.data)


def test_grpc_end_to_end():
    print("7. real gRPC round-trip (the path students use)")
    import concurrent.futures
    import grpc

    svc = make_service()
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=4),
        options=[("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)],
    )
    ts.tapnext_pb2_grpc.add_PipelineServiceServicer_to_server(svc, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        channel = grpc.insecure_channel(f"127.0.0.1:{port}", options=[
            ("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)])
        stub = ts.tapnext_pb2_grpc.PipelineServiceStub(channel)

        def call(cfg, data=None):
            return stub.Process(pb2.Envelope(config_json=json.dumps(cfg), data=data or {}))

        rA = call({"tapnext": {"command": "track", "parameters": {"grid_size": 4},
                               "session_id": "g-a"}},
                  {"images": aux.wrap_value([make_frame_bytes(), make_frame_bytes()])})
        rB = call({"tapnext": {"command": "track", "parameters": {"grid_size": 4},
                               "session_id": "g-b"}},
                  {"images": aux.wrap_value([make_frame_bytes()])})
        cA, cB = cfg_of(rA), cfg_of(rB)
        check("gRPC: A done, 2 frames", cA["status"] == "done" and cA["frames_processed"] == 2, str(cA))
        check("gRPC: B done, 1 frame (isolated)", cB["status"] == "done" and cB["frames_processed"] == 1, str(cB))
        check("gRPC: tracks payload is real torch bytes",
              torch.load(io.BytesIO(aux.unwrap_value(rB.data["tracks"])), weights_only=False).shape[0] == 1)

        call({"tapnext": {"command": "reset", "session_id": "g-a"}})
        rB2 = call({"tapnext": {"command": "track", "parameters": {"grid_size": 4},
                                "session_id": "g-b"}},
                   {"images": aux.wrap_value([make_frame_bytes()])})
        check("gRPC: B survives A's reset over the wire (2 frames)",
              cfg_of(rB2)["frames_processed"] == 2, f"got {cfg_of(rB2)['frames_processed']}")

        rl = call({"tapnext": {"command": "list"}})
        sids = {s["session"] for s in json.loads(rl.config_json)["tapnext"]["sessions"]}
        check("gRPC: list sees both sessions", {"g-a", "g-b"} <= sids, str(sids))
    finally:
        server.stop(0)


def test_video_input():
    print("8. data.video input (the box decodes the video server-side, like yolo)")
    video_path = os.path.join(TEST_DIR, "apple.mp4")
    if not os.path.exists(video_path):
        print("  (apple.mp4 fixture not present — skipped)")
        return
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    svc = make_service()

    def call(data, params, sid):
        cfg = {"tapnext": {"command": "track", "session_id": sid, "parameters": params}}
        return svc.Process(pb2.Envelope(config_json=json.dumps(cfg), data=data), None)

    r = call({"video": aux.wrap_value(video_bytes)},
             {"grid_size": 4, "frame_step": 2, "max_frames": 3}, "v-sampled")
    c = cfg_of(r)
    check("video: status done + session echo", c.get("status") == "done" and c.get("session") == "v-sampled", str(c))
    check("video: frames_processed == max_frames (3)", c.get("frames_processed") == 3, f"got {c.get('frames_processed')}")
    check("video: encoding still declares 'torch'", c.get("encoding", {}).get("tracks") == "torch")
    check("video: tracks + visibles returned", "tracks" in r.data and "visibles" in r.data)
    t = last_tracks(r)
    check("video: tracks shape (frames, points, 2)", t.ndim == 3 and t.shape[2] == 2 and t.shape[0] == 3, f"shape={t.shape}")

    frame = make_frame_bytes()
    rb = call({"video": aux.wrap_value(video_bytes), "images": aux.wrap_value([frame])},
              {"grid_size": 4}, "v-both")
    cb = cfg_of(rb)
    check("video+images rejected (mutually exclusive)", cb.get("status") == "error" and "both" in cb.get("error", ""), str(cb))

    rn = call({}, {"grid_size": 4}, "v-none")
    check("neither input -> empty_request", cfg_of(rn).get("status") == "empty_request")

    ri = call({"images": aux.wrap_value([frame, frame])}, {"grid_size": 4}, "v-img")
    check("legacy images path unaffected (2 frames)", cfg_of(ri).get("frames_processed") == 2,
          f"got {cfg_of(ri).get('frames_processed')}")


def main():
    tests = [
        test_isolation,
        test_scoped_reset,
        test_default_session_backcompat,
        test_concurrency,
        test_reaper,
        test_list,
        test_grpc_end_to_end,
        test_video_input,
    ]
    for t in tests:
        t()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        sys.exit(1)
    print("All session tests passed.")


if __name__ == "__main__":
    main()
