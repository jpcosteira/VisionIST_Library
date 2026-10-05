#!/usr/bin/env python3
"""Standalone smoke test for the vggt box (shared `Process` Envelope).

Connects to a running vggt box and:
  1. sends the bundled test frames (test/images/00.jpg, 01.jpg)
  2. checks the "vggt" status section in the response config
  3. decodes the tensors via the declared "torch" codec and prints shapes
  4. writes the .glb to /tmp/vggt_smoke.glb

Run (server already up on :8061):
    python images/vggt/test/test_vggt.py
    BOX_HOST=10.0.0.5:8061 python images/vggt/test/test_vggt.py
"""

import io
import json
import os
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_TEST_DIR, "..", "protos"))

import grpc  # noqa: E402
import pipeline_pb2  # noqa: E402
import pipeline_pb2_grpc  # noqa: E402
import aux  # noqa: E402
import torch  # noqa: E402


def main():
    host = os.getenv("BOX_HOST", "localhost:8061")
    print(f"calling vggt box at {host} ...")

    frame_paths = [os.path.join(_TEST_DIR, "images", n) for n in ("00.jpg", "01.jpg")]
    missing = [p for p in frame_paths if not os.path.isfile(p)]
    if missing:
        sys.exit(f"missing test frames: {missing}")
    images = [open(p, "rb").read() for p in frame_paths]
    print(f"loaded {len(images)} test frames "
          f"({sum(len(i) for i in images) / (1024 * 1024):.1f} MB total)")

    channel = grpc.insecure_channel(host, options=[
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
    ])
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

    def call(config):
        return stub.Process(pipeline_pb2.Envelope(
            config_json=json.dumps(config),
            data={"images": aux.wrap_value(images)},
        ))

    # --- 1. reset is a stateless no-op ------------------------------------
    res = call({"vggt": {"command": "reset"}})
    cfg = json.loads(res.config_json)
    print("reset:", cfg)
    assert cfg["vggt"]["status"] == "done", cfg

    # --- 2. empty_request when images are missing -------------------------
    res = stub.Process(pipeline_pb2.Envelope(
        config_json=json.dumps({"vggt": {"command": "reconstruct"}})))
    cfg = json.loads(res.config_json)
    print("empty_request:", cfg)
    assert cfg["vggt"]["status"] == "empty_request", cfg

    # --- 3. the reconstruction itself --------------------------------------
    res = call({"vggt": {"command": "reconstruct",
                         "parameters": {"conf_threshold": 30}}})
    cfg = json.loads(res.config_json)
    print("result:", cfg, "| data fields:", sorted(res.data.keys()))
    assert cfg["vggt"]["status"] == "done", cfg

    enc = cfg["vggt"].get("encoding", {})
    tensor_fields = [f for f in ("world_points", "world_points_conf", "depth",
                                 "depth_conf", "extrinsic", "intrinsic", "images")
                     if f in res.data]
    for f in tensor_fields:
        t = torch.load(io.BytesIO(aux.unwrap_value(res.data[f])),
                       map_location="cpu", weights_only=False)
        print(f"  {f:20s} torch tensor {tuple(t.shape)} {t.dtype}")
        if f in enc:
            assert enc[f] == "torch", (f, enc[f])

    glb_bytes = aux.unwrap_value(res.data["glb_file"]) if "glb_file" in res.data else b""
    glb_bytes = bytes(glb_bytes)
    _check_glb(glb_bytes)

    # --- 4. CPU round trip --------------------------------------------------
    # the default call above ran on CUDA (when visible); a CPU call right
    # after must also work (catches device-sensitive caches upstream)
    if torch.cuda.is_available():
        call({"vggt": {"command": "reconstruct",
                       "parameters": {"device": "cpu", "conf_threshold": 30}}})
        res = call({"vggt": {"command": "reconstruct",
                             "parameters": {"device": "cpu"}}})
        cfg = json.loads(res.config_json)
        print("cpu round trip:", {k: cfg["vggt"][k] for k in ("status", "device")})
        assert cfg["vggt"]["status"] == "done" and cfg["vggt"]["device"] == "cpu", cfg
        glb2 = bytes(aux.unwrap_value(res.data["glb_file"]))
        _check_glb(glb2)

        # and back: cpu -> cuda must survive too
        res = call({"vggt": {"command": "reconstruct",
                             "parameters": {"device": "cuda"}}})
        cfg = json.loads(res.config_json)
        assert cfg["vggt"]["status"] == "done" and cfg["vggt"]["device"].startswith("cuda"), cfg
        print("back to cuda:", cfg["vggt"]["device"])

    print("OK — vggt box smoke test passed")


def _check_glb(glb_bytes: bytes):
    print(f"  glb_file: {len(glb_bytes) / (1024 * 1024):.2f} MB, header={glb_bytes[:4]!r}")
    assert glb_bytes[:4] == b"glTF", "glb_file is not a glTF binary"
    out = "/tmp/vggt_smoke.glb"
    with open(out, "wb") as f:
        f.write(glb_bytes)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
