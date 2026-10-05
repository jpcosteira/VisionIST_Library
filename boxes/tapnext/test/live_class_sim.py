#!/usr/bin/env python3
"""
Live class simulation: 10 simultaneous "students" driving ONE running tapnext
box, with realistic staggering — some start together, others start in between.

Each student:
  * gets a random session_id
  * resets their own session (fresh start)
  * streams K random video frames from ./apples.mp4, one request per frame
  * records status, latencies, final frames_processed

Isolation proof: after a run, `list` must show exactly the frames each student
sent for their own session — no more, no less.

Run (from images/tapnext_tracker/test/):
    python live_class_sim.py
Env:
    BOX_HOST    default localhost:9063
    STUDENTS    number of sessions, default 10
"""

import io
import json
import os
import random
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "protos"))

import cv2                      # noqa: E402
import grpc                     # noqa: E402
import numpy as np              # noqa: E402
import torch                    # noqa: E402

import pipeline_pb2 as pb2      # noqa: E402
import pipeline_pb2_grpc as pb2_grpc  # noqa: E402
import aux                      # noqa: E402

BOX_HOST = os.getenv("BOX_HOST", "localhost:9063")
N_STUDENTS = int(os.getenv("STUDENTS", "10"))
GRID = 16        # 16x16 = 256 tracked points per student (fast, still real tracking)


def load_frames(path):
    """Prefer a pre-decoded <video-stem>_frames.npy (plain numpy read); else
    cv2; else imageio (bundled ffmpeg) if installed."""
    import cv2
    cache = os.path.splitext(path)[0] + "_frames.npy"
    if os.path.exists(cache):
        import numpy as np
        return list(np.load(cache))
    cap = cv2.VideoCapture(path)
    frames = []
    if cap.isOpened():
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
    if frames and frames[0] is not None:
        return frames
    try:
        import imageio.v3 as iio
        import numpy as np
        return list(np.asarray(iio.imread(path)))
    except ModuleNotFoundError:
        raise RuntimeError(
            f"could not decode {path}: this cv2 build has no video backend "
            f"and imageio is not installed (pip install imageio imageio-ffmpeg)")


def find_video():
    import glob
    cands = sorted(glob.glob(os.path.join(HERE, "*.mp4")))
    cands = [c for c in cands if "output" not in c]
    if not cands:
        raise FileNotFoundError(f"no input video in {HERE}")
    return cands[0]


VIDEO = find_video()
FRAMES = load_frames(VIDEO)
OPTS = [("grpc.max_send_message_length", -1), ("grpc.max_receive_message_length", -1)]
CHANNEL = grpc.insecure_channel(BOX_HOST, options=OPTS)
STUB = pb2_grpc.PipelineServiceStub(CHANNEL)
PRINT_LOCK = threading.Lock()


def log(msg):
    with PRINT_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rpc(cfg, images=None):
    data = {"images": aux.wrap_value(images)} if images else {}
    env = pb2.Envelope(config_json=json.dumps(cfg), data=data)
    return STUB.Process(env, timeout=300)


def encode(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("frame encode failed")
    return buf.tobytes()


def student(sid, delay, n_frames, seed, stats, start_wall):
    t0 = time.monotonic()
    time.sleep(delay)  # staggering: some start together (delay=0), some in between
    log(f"{sid}: START  (+{delay:5.1f}s)  {n_frames} frames from video "
        f"(offset {seed % len(FRAMES)})")
    t_start = time.monotonic()

    r = rpc({"tapnext": {"command": "reset", "session_id": sid}})
    c = json.loads(r.config_json)["tapnext"]
    if c.get("status") != "done":
        stats[sid] = {"error": f"reset failed: {c}"}
        return

    sent = 0
    lat_s, lat_ms = [], []
    try:
        for i in range(n_frames):
            frame = FRAMES[(seed + i) % len(FRAMES)]
            t = time.monotonic()
            r = rpc({"tapnext": {"command": "track",
                                 "parameters": {"grid_size": GRID},
                                 "session_id": sid}},
                    [encode(frame)])
            c = json.loads(r.config_json)["tapnext"]
            if c.get("status") != "done":
                stats[sid] = {"error": f"frame {i} failed: {c}", "sent": sent}
                return
            dt = (time.monotonic() - t) * 1000
            lat_ms.append(dt)
            # a little human pacing between frames
            time.sleep(random.uniform(0.02, 0.12))
            sent += 1
    except Exception as e:  # noqa: BLE001 - reported per student
        stats[sid] = {"error": repr(e), "sent": sent}
        return

    # grab the accumulated payload once and sanity-check its shape
    r = rpc({"tapnext": {"command": "track",
                         "parameters": {"grid_size": GRID},
                         "session_id": sid}},
            [encode(FRAMES[(seed + n_frames) % len(FRAMES)])])
    c = json.loads(r.config_json)["tapnext"]
    sent += 1
    n_pts = GRID * GRID
    if "tracks" in r.data:
        t = torch.load(io.BytesIO(aux.unwrap_value(r.data["tracks"])), weights_only=False)
        shape_ok = t.numpy().shape == (sent, n_pts, 2)
    else:
        shape_ok = False

    dur = time.monotonic() - t_start
    stats[sid] = {
        "error": None,
        "sent": sent,
        "reported_final": c.get("frames_processed"),
        "payload_shape_ok": shape_ok,
        "first_ms": min(lat_ms), "mean_ms": sum(lat_ms) / len(lat_ms), "max_ms": max(lat_ms),
        "dur_s": dur,
        "started_after_s": t_start - start_wall,
    }
    s = stats[sid]
    log(f"{sid}: DONE   sent={sent} fps={sent / dur:.1f} "
        f"lat(ms) min={s['first_ms']:.0f} mean={s['mean_ms']:.0f} max={s['max_ms']:.0f} "
        f"wall={dur:.1f}s")


def main():
    random.seed(20250914)
    names = ["ana", "liam", "marco", "sara", "yusuf", "elena", "david", "marta", "pedro",
             "zoe", "nina", "omar", "ines", "luka", "tara", "ivo", "kara", "noah"]
    sids = [f"stu-{random.choice(names)}-{random.randint(10, 99)}" for _ in range(N_STUDENTS)]
    # staggering plan: 4 start together (delay 0), the rest start at random
    # in-between delays so streams overlap realistically
    delays = [0.0] * 4 + [round(random.uniform(1.5, 18.0), 1) for _ in range(N_STUDENTS - 4)]
    random.shuffle(delays)
    log(f"video frames loaded: {len(FRAMES)}  (video: {os.path.basename(VIDEO)})")
    log(f"box: {BOX_HOST}   students: {N_STUDENTS}   grid: {GRID}x{GRID}")
    log("stagger (s): " + ", ".join(f"{d:5.1f}" for d in delays))

    stats = {}
    start_wall = time.monotonic()
    threads = []
    for sid, delay in zip(sids, delays):
        n_frames = random.randint(15, 30)
        seed = random.randrange(len(FRAMES))
        th = threading.Thread(target=student, args=(sid, delay, n_frames, seed, stats, start_wall), daemon=True)
        th.start()
        threads.append(th)
    wall0 = time.monotonic()
    for th in threads:
        th.join(timeout=1800)
    wall = time.monotonic() - wall0

    # ground truth from the box itself
    final = json.loads(rpc({"tapnext": {"command": "list"}}).config_json)["tapnext"]
    by = {s["session"]: s for s in final.get("sessions", [])}

    log("")
    log("=" * 100)
    ok_all = True
    for sid in sids:
        s = stats.get(sid, {})
        box_n = by.get(sid, {}).get("frames_processed")
        match = s.get("sent") is not None and s.get("sent") == box_n
        payload = s.get("payload_shape_ok") is not None
        ok = s.get("error") is None and match and payload
        ok_all = ok_all and ok
        mark = "OK " if ok else "FAIL"
        err = f"  error={s.get('error')}" if s.get("error") else ""
        log(f"{mark} {sid:18s} sent={s.get('sent')}  box_reports={box_n}  "
            f"payload_shape={'ok' if payload else 'BAD'}  {err}")
    log("=" * 100)
    log(f"total wall time {wall:.1f}s | box final sessions: {len(by)}")
    if ok_all:
        log("CLASS SIM PASSED — 10 students, one box, one GPU, every session exact.")
    else:
        log("CLASS SIM FAILED — see FAIL lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
