#!/usr/bin/env python3
"""Test script for the YOLO gRPC service (shared envelope interface).

Connects to a running yolo box and:
  1. builds an Envelope with the two bundled test images
     (``dog.jpg`` - the dog in the grass, and ``car.jpg`` - the car)
  2. calls stub.Process(request)
  3. decodes the declared-``json`` ``detections`` field and prints, per
     frame, the detection count and each box/label/score
  4. checks the ``annotated`` JPEG frames (magic + count)
  5. exercises the tapnext-style multi-session tracking contract
     (``track_id`` per box, stable within a session, independent across
     sessions, ``reset`` scoped to the session, ``list``)
  6. optionally, if a video is available (``YOLO_TEST_VIDEO`` env var, or
     the repo's ``cozinha.mp4`` at the repo root), runs the same tour on
     the video input with frame sampling
  7. optionally, if ``YOLO_TEST_WEIGHTS`` names a fetchable checkpoint,
     exercises the ``parameters.weights`` switch (and the switch back)

Mirrors the style of images/clip/test/test_clip.py.

Run (from the repo or image root, server already up):
    python images/yolo/test/test_yolo.py
    BOX_HOST=10.0.0.5:8061 python images/yolo/test/test_yolo.py
    YOLO_TEST_VIDEO=cozinha.mp4 python images/yolo/test/test_yolo.py
"""

import json
import os
import sys

# Make the protos/ folder importable (same pattern as the clip test).
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_TEST_DIR, "..", "protos"))

import grpc  # noqa: E402
import pipeline_pb2  # noqa: E402
import pipeline_pb2_grpc  # noqa: E402
import aux  # noqa: E402


def load_local_image(path: str) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    print(f"loaded {os.path.basename(path)}: {len(data) / (1024*1024):.2f} MB")
    return data


def make_stub(target: str):
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    return pipeline_pb2_grpc.PipelineServiceStub(channel), channel


def check_status(response):
    cfg = json.loads(response.config_json or "{}")
    print(f"config: {cfg}")
    section = cfg.get("yolo")
    if not isinstance(section, dict) or section.get("status") == "error":
        print(f"  ERROR: {section.get('error') if section else cfg}")
        return None
    return section


def print_detections(detections):
    total = 0
    for d in detections:
        n = len(d["boxes"])
        total += n
        tid = d.get("track_id")
        print(f"  frame {d['frame_index']:>3d} ({d['width']}x{d['height']}): {n} detection(s)"
              + (f"  track_ids={tid}" if tid else ""))
        for box, label, cid, score in zip(d["boxes"], d["labels"], d["class_ids"], d["scores"]):
            print(f"      {label:>10s} (class {cid})  score={score:.4f}  xyxy=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    return total


def main():
    target = os.getenv("BOX_HOST", "localhost:8061")
    print(f"Target: {target}")
    stub, channel = make_stub(target)

    failures = []

    # ------------------------------------------------------------------ #
    # Case 1: images                                                      #
    # ------------------------------------------------------------------ #
    print("\n== case 1: images (dog.jpg + car.jpg) ==")
    image_bytes_list = [
        load_local_image(os.path.join(_TEST_DIR, "dog.jpg")),
        load_local_image(os.path.join(_TEST_DIR, "car.jpg")),
    ]
    request = pipeline_pb2.Envelope(
        config_json=json.dumps({
            "yolo": {
                "command": "detect",
                "parameters": {"conf": 0.25, "iou": 0.7},
            }
        }),
        data={"images": aux.wrap_value(image_bytes_list)},
    )
    response = stub.Process(request)
    section = check_status(response)
    if section is None:
        return 1

    if "detections" not in response.data:
        print("  missing field: detections")
        failures.append("detections missing")
    else:
        raw = aux.unwrap_value(response.data["detections"])
        detections = json.loads(bytes(raw).decode("utf-8"))
        if not isinstance(detections, list) or len(detections) != len(image_bytes_list):
            print(f"  wrong detections shape: {type(detections)} len={len(detections) if isinstance(detections, list) else '-'}")
            failures.append("detections shape")
        else:
            for d in detections:
                for key in ("frame_index", "width", "height", "boxes", "labels",
                            "class_ids", "scores", "track_id"):
                    if key not in d:
                        print(f"  detection record missing key {key!r}")
                        failures.append(f"record key {key}")
            total = print_detections(detections)
            print(f"  total: {total} detections over {len(detections)} frames; declared encoding: {section.get('encoding')}")
            if total == 0:
                print("  (no detections — expected on odd photos, not a failure)")

    if "annotated" in response.data:
        ann = aux.unwrap_value(response.data["annotated"])
        if not isinstance(ann, list) or len(ann) != len(image_bytes_list):
            print(f"  annotated: wrong shape {type(ann)} len={len(ann) if isinstance(ann, list) else '-'}")
            failures.append("annotated shape")
        else:
            non_jpeg = [i for i, f in enumerate(ann) if not bytes(f).startswith(b"\xff\xd8")]
            if non_jpeg:
                print(f"  annotated: frames {non_jpeg} are not JPEG")
                failures.append("annotated not jpeg")
            else:
                print(f"  annotated: {len(ann)} JPEG frame(s), first {len(ann[0])} bytes")
    else:
        print("  annotated: (absent — box declared save_annotated on; expect this field)")
        failures.append("annotated missing")

    # ------------------------------------------------------------------ #
    # Case 2: reset (scoped to the calling session)                       #
    # ------------------------------------------------------------------ #
    print("\n== case 2: reset (scoped to session) ==")
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"yolo": {"command": "reset", "session_id": "case2"}})))
    section = check_status(response)
    if section is None:
        return 1
    print(f"  action: {section.get('action')}  session: {section.get('session')}")
    if section.get("action") != "reset":
        failures.append("reset action")
    if section.get("session") != "case2":
        failures.append("reset session echo")

    # ------------------------------------------------------------------ #
    # Case 2b: multi-session tracking (tapnext-style contract)            #
    # ------------------------------------------------------------------ #
    print("\n== case 2b: multi-session tracking (track_id) ==")
    dog = image_bytes_list[0]  # dog.jpg — reliably has a COCO object

    def call_session(sid):
        resp = stub.Process(pipeline_pb2.Envelope(
            config_json=json.dumps({
                "yolo": {
                    "command": "detect",
                    "session_id": sid,
                    "parameters": {"conf": 0.25, "save_annotated": False},
                }
            }),
            data={"images": aux.wrap_value([dog])},
        ))
        sec = check_status(resp)
        if sec is None:
            return None
        dets = json.loads(bytes(aux.unwrap_value(resp.data["detections"])).decode("utf-8"))
        return sec, dets

    ids = {}
    out_a1 = call_session("t-a")
    out_a2 = call_session("t-a")   # same session again: ids must be stable
    out_b = call_session("t-b")    # different session: independent sequence
    for label, out in (("a1", out_a1), ("a2", out_a2), ("b", out_b)):
        if out is None:
            failures.append(f"session call {label}")
            continue
        sec, dets = out
        if sec.get("session") != ("t-a" if label.startswith("a") else "t-b"):
            failures.append(f"session echo {label}")
        box_ids = dets[0].get("track_id")
        if dets[0]["boxes"] and (not isinstance(box_ids, list)
                                 or len(box_ids) != len(dets[0]["boxes"])
                                 or not all(isinstance(t, int) for t in box_ids)):
            failures.append(f"track_id shape {label}: {box_ids!r}")
        ids[label] = box_ids

    if ids.get("a1") and ids.get("a2"):
        if ids["a1"] == ids["a2"]:
            print(f"  stable ids across calls in t-a: {ids['a1']}")
        else:
            print(f"  UNSTABLE ids: a1={ids['a1']} a2={ids['a2']}")
            failures.append("session id stability")

    # fresh tracker = numbering from the base, for EVERY new session
    # (regenerate / new key behaves like a reset — per-session namespaces)
    for label in ("a1", "b"):
        if ids.get(label) and ids[label][0] != 1:
            print(f"  session {label} did NOT start at base: {ids[label]}")
            failures.append(f"fresh session {label} base id")
    if ids.get("a1") and ids.get("b"):
        print(f"  each session starts at base: t-a={ids['a1']} t-b={ids['b']}")

    # reset scoped to t-a only; t-b is untouched
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"yolo": {"command": "reset", "session_id": "t-a"}})))
    section = check_status(response)
    if section is None or section.get("session") != "t-a":
        failures.append("scoped reset t-a")
    out_a3 = call_session("t-a")
    if out_a3:
        box_ids3 = out_a3[1][0].get("track_id")
        print(f"  t-a after reset: ids={box_ids3} (fresh tracker, numbering from base)")
        if out_a3[1][0]["boxes"] and not box_ids3:
            failures.append("post-reset track ids")
        if box_ids3 and box_ids3[0] != 1:
            print(f"  t-a after reset did NOT restart at base: {box_ids3}")
            failures.append("post-reset base id")
        # t-b must still hold its id from before (scoped reset left it alone)
        out_b2 = call_session("t-b")
        if out_b2 and ids.get("b") and out_b2[1][0].get("track_id") == ids["b"]:
            print(f"  t-b untouched by t-a's reset: ids still {ids['b']}")
        elif out_b2 and ids.get("b"):
            print(f"  t-b RE-NUMBERED after t-a reset: {ids['b']} -> {out_b2[1][0].get('track_id')}")
            failures.append("scoped reset leaked to t-b")

    # list: active sessions, no state content
    response = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"yolo": {"command": "list"}})))
    section = check_status(response)
    if section is None:
        failures.append("list")
    else:
        names = {s.get("session") for s in section.get("sessions", [])}
        print(f"  list: {sorted(names)}")
        if not {"t-a", "t-b"}.issubset(names):
            failures.append(f"list missing sessions: {names}")
        for s in section.get("sessions", []):
            for key in ("session", "frames_processed", "has_tracker", "idle_seconds"):
                if key not in s:
                    failures.append(f"list entry key {key}")

    # ------------------------------------------------------------------ #
    # Case 3: video (optional — needs a real video file)                  #
    # ------------------------------------------------------------------ #
    print("\n== case 3: video (optional) ==")
    video_path = os.getenv("YOLO_TEST_VIDEO") or os.path.join(
        _TEST_DIR, "..", "..", "..", "cozinha.mp4")
    if not os.path.isfile(video_path):
        print(f"  SKIPPED — no video (set YOLO_TEST_VIDEO; tried {video_path})")
    else:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        print(f"loaded {os.path.basename(video_path)}: {len(video_bytes) / (1024*1024):.2f} MB")
        request = pipeline_pb2.Envelope(
            config_json=json.dumps({
                "yolo": {
                    "command": "detect",
                    "parameters": {"conf": 0.25, "frame_step": 30, "max_frames": 8},
                }
            }),
            data={"video": aux.wrap_value(video_bytes)},
        )
        response = stub.Process(request)
        section = check_status(response)
        if section is None:
            failures.append("video status")
        else:
            if section.get("source") != "video":
                print(f"  source not 'video': {section.get('source')}")
                failures.append("video source")
            if "detections" in response.data:
                detections = json.loads(aux.unwrap_value(response.data["detections"]).decode("utf-8"))
                total = print_detections(detections)
                print(f"  {section.get('frames_sampled')}/{section.get('frames_in_video')} frames sampled, "
                      f"{total} detections")
                idxs = [d["frame_index"] for d in detections]
                if idxs != [i * 30 for i in range(len(idxs))]:
                    print(f"  frame_index spacing wrong: {idxs}")
                    failures.append("video frame_index")
            else:
                print("  missing field: detections")
                failures.append("video detections missing")

            # save_annotated (default on): JPEG list + one annotated mp4
            ann = aux.unwrap_value(response.data["annotated"]) if "annotated" in response.data else None
            av = aux.unwrap_value(response.data["annotated_video"]) if "annotated_video" in response.data else None
            if not ann or not isinstance(ann, list) or len(ann) != len(detections if "detections" in response.data else []):
                failures.append("video annotated missing/wrong shape")
            codec = section.get("annotated_video_codec")
            if not isinstance(av, (bytes, bytearray)) or bytes(av)[4:8] != b"ftyp":
                print(f"  annotated_video: not an mp4 ({type(av)})")
                failures.append("video annotated_video")
            else:
                # when PyAV is available the box must have encoded browser-
                # playable H.264 — verify by decoding the artifact
                codec_ok = True
                try:
                    import av as avlib
                    import tempfile, os as _os
                    _p = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    with open(_p, "wb") as _f:
                        _f.write(bytes(av))
                    _c = avlib.open(_p)
                    got = _c.streams.video[0].codec_context.name
                    _n = sum(1 for _ in _c.decode(video=0))
                    _c.close()
                    _os.unlink(_p)
                    print(f"  annotated: {len(ann)} JPEGs; annotated_video: {len(av)} bytes, "
                          f"codec={got} (declared {codec}), {_n} frames decoded")
                    if got != "h264" or codec != "h264":
                        codec_ok = False
                except ImportError:
                    print(f"  annotated: {len(ann)} JPEGs; annotated_video: {len(av)} bytes, "
                          f"codec={codec} (av not available on test host, skipped verify)")
                except Exception as e:
                    codec_ok = False
                    print(f"  annotated_video decode check failed: {e}")
                if not codec_ok:
                    failures.append("video annotated_video codec")

            # save_annotated=false drops both
            r2 = stub.Process(pipeline_pb2.Envelope(
                config_json=json.dumps({
                    "yolo": {"command": "detect",
                             "parameters": {"frame_step": 30, "max_frames": 1,
                                            "save_annotated": False}},
                }),
                data={"video": aux.wrap_value(video_bytes)},
            ))
            s2 = check_status(r2)
            if s2 is None:
                failures.append("video save_annotated=fail status")
            elif "annotated" in r2.data or "annotated_video" in r2.data:
                print("  save_annotated=false did not drop the annotated fields")
                failures.append("video save_annotated=false")
            else:
                print("  save_annotated=false: no annotated fields (ok)")

    # ------------------------------------------------------------------ #
    # Case 4: weights switch (optional — needs a fetchable checkpoint)    #
    # ------------------------------------------------------------------ #
    print("\n== case 4: weights switch (optional) ==")
    test_weights = os.getenv("YOLO_TEST_WEIGHTS")
    if not test_weights:
        print("  SKIPPED — set YOLO_TEST_WEIGHTS (e.g. yolov8s.pt)")
    else:
        request = pipeline_pb2.Envelope(
            config_json=json.dumps({
                "yolo": {
                    "command": "detect",
                    "parameters": {"weights": test_weights},
                }
            }),
            data={"images": aux.wrap_value([image_bytes_list[0]])},
        )
        response = stub.Process(request)
        section = check_status(response)
        if section is None:
            failures.append("weights status")
        else:
            if section.get("weights") != test_weights:
                print(f"  weights not switched: {section.get('weights')} != {test_weights}")
                failures.append("weights switch")
            else:
                print(f"  active checkpoint: {section.get('weights')}")
        # switch back to the startup default (also proves the reverse)
        response = stub.Process(pipeline_pb2.Envelope(
            config_json=json.dumps({"yolo": {"command": "detect", "parameters": {}}}),
            data={"images": aux.wrap_value([image_bytes_list[0]])},
        ))
        section = check_status(response)
        if section is not None and section.get("weights") == "yolov8n.pt":
            print(f"  switched back to: {section.get('weights')}")
        else:
            print(f"  back-switch: {section.get('weights') if section else 'error'}")
            failures.append("weights back-switch")

    channel.close()

    print("\n" + ("FAIL -- " + "; ".join(failures) if failures else "PASS -- all yolo test cases."))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
