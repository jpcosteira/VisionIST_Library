#!/usr/bin/env python3
"""
Equivalence check: two DIFFERENT session_ids are fed the EXACT same frame
sequence (same order, same grid). If the box's session isolation is sound and
inference is deterministic, their tracked coordinates must be element-for-
element equal (up to float16 non-determinism). Any state leakage or cross-talk
shows up as a large coordinate difference.

Runs the two sessions INTERLEAVED (A0, B0, A1, B1, ...) so it also exercises
the concurrent path, not just sequential.

Run (from images/tapnext_tracker/test/):
    python equivalence_check.py
"""

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "protos"))

import cv2            # noqa: E402
import grpc           # noqa: E402
import numpy as np    # noqa: E402
import torch          # noqa: E402

import pipeline_pb2 as pb2            # noqa: E402
import pipeline_pb2_grpc as pb2_grpc  # noqa: E402
import aux                            # noqa: E402

BOX_HOST = os.getenv("BOX_HOST", "localhost:9063")
N = int(os.getenv("EQ_FRAMES", "8"))   # same frames for both sessions
GRID = 16


def frames_cache():
    import glob
    v = [c for c in sorted(glob.glob(os.path.join(HERE, "*.mp4"))) if "output" not in c][0]
    cache = os.path.splitext(v)[0] + "_frames.npy"
    if os.path.exists(cache):
        return list(np.load(cache))
    cap = cv2.VideoCapture(v)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    return fr


FRAMES = frames_cache()
OPTS = [("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)]
STUB = pb2_grpc.PipelineServiceStub(grpc.insecure_channel(BOX_HOST, options=OPTS))


def rpc(cfg, images=None):
    data = {"images": aux.wrap_value(images)} if images else {}
    env = pb2.Envelope(config_json=json.dumps(cfg), data=data)
    return STUB.Process(env, timeout=120)


def encode(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def main():
    a_sid, b_sid = "eq-ALPHA-01", "eq-BETA-02"
    print(f"box={BOX_HOST}  grid={GRID}  same {N} frames -> two different sessions")

    for sid in (a_sid, b_sid):
        c = json.loads(rpc({"tapnext": {"command": "reset", "session_id": sid}}).config_json)["tapnext"]
        assert c.get("status") == "done", c

    lastA = lastB = None
    for i in range(N):
        f = FRAMES[i]
        lastA = rpc({"tapnext": {"command": "track", "parameters": {"grid_size": GRID},
                                 "session_id": a_sid}}, [encode(f)])
        lastB = rpc({"tapnext": {"command": "track", "parameters": {"grid_size": GRID},
                                 "session_id": b_sid}}, [encode(f)])

    def tracks_of(resp):
        t = torch.load(io.BytesIO(aux.unwrap_value(resp.data["tracks"])), weights_only=False)
        return t.numpy()

    A = tracks_of(lastA)
    B = tracks_of(lastB)
    ca = json.loads(lastA.config_json)["tapnext"]
    cb = json.loads(lastB.config_json)["tapnext"]

    print(f"A: session={ca['session']} frames={ca['frames_processed']} shape={A.shape}")
    print(f"B: session={cb['session']} frames={cb['frames_processed']} shape={B.shape}")

    same_shape = A.shape == B.shape
    max_diff = float(np.max(np.abs(A - B))) if same_shape else float("inf")
    mean_diff = float(np.mean(np.abs(A - B))) if same_shape else float("inf")
    equal = same_shape and np.allclose(A, B, rtol=1e-3, atol=2e-3)

    print()
    print(f"coordinate comparison (A vs B, {N} identical input frames):")
    print(f"  equal shape      : {same_shape}")
    print(f"  max |A - B|      : {max_diff:.6g}")
    print(f"  mean|A - B|      : {mean_diff:.6g}")
    print(f"  allclose(rtol1e-3 atol2e-3): {equal}")
    print()
    print("EQUIVALENCE " + ("PASSED — same input, two sessions, identical results."
                            if equal else "FAILED — outputs diverged (see diffs above)."))
    sys.exit(0 if equal else 1)


if __name__ == "__main__":
    main()
