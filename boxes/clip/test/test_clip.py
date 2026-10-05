#!/usr/bin/env python3
"""Test script for the CLIP gRPC service (shared envelope interface).

Connects to a running clip box and:
  1. builds an Envelope with the two bundled test images
     (``dog.jpg`` - the dog in the grass, and ``car.jpg`` - the car)
     plus sample texts
  2. calls stub.Process(request)
  3. loads and prints the shape of image_emb / text_emb / similarity
     and the per-image similarity logits

Mirrors the style of images/tapnext_tracker/test/test_tapnext.py.

Run (from the repo or image root, server already up):
    python images/clip/test/test_clip.py
    BOX_HOST=10.0.0.5:8061 python images/clip/test/test_clip.py
"""

import json
import os
import sys

# Make the protos/ folder importable (same pattern as the tapnext test).
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


def bytes_to_tensor(b: bytes):
    import torch
    import io as bio
    return torch.load(bio.BytesIO(b), weights_only=False)


def main():
    target = os.getenv("BOX_HOST", "localhost:8061")
    print(f"Target: {target}")

    # The two bundled test images (dog in the grass, car).
    image_bytes_list = [
        load_local_image(os.path.join(_TEST_DIR, "dog.jpg")),
        load_local_image(os.path.join(_TEST_DIR, "car.jpg")),
    ]
    texts = ["a diagram", "a dog", "a cat", "garden", "grass", "dirt road", "race car"]

    config = {
        "clip": {
            "command": "encode",
            "parameters": {},
        }
    }

    request = pipeline_pb2.Envelope(
        config_json=json.dumps(config),
        data={
            "images": aux.wrap_value(image_bytes_list),
            "texts": aux.wrap_value(texts),
        },
    )

    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

    response = stub.Process(request)

    cfg = json.loads(response.config_json or "{}")
    print(f"config: {cfg}")

    if "clip" in cfg and cfg["clip"].get("status") == "error":
        print(f"  ERROR: {cfg['clip'].get('error')}")
        return 1

    shapes = {}
    for key in ("image_emb", "text_emb", "similarity"):
        if key not in response.data:
            print(f"missing field: {key}")
            continue
        val = aux.unwrap_value(response.data[key])
        if not isinstance(val, (bytes, bytearray)):
            print(f"{key}: (not bytes) {type(val)}")
            continue
        t = bytes_to_tensor(val)
        shapes[key] = t
        print(f"{key}: shape={tuple(t.shape)} dtype={t.dtype}")

    # Show the cross-modal logits so the expected pairings are visible
    # (dog -> "a dog", car -> "race car").
    if "similarity" in shapes:
        import torch
        logits = shapes["similarity"].softmax(dim=-1).detach().cpu().numpy()
        labels = [os.path.basename(p) for p in ("dog.jpg", "car.jpg")]
        print("\nper-image similarity (softmax over texts):")
        for i, lab in enumerate(labels):
            row = logits[i]
            top3 = sorted(range(len(texts)), key=lambda j: -row[j])[:3]
            top3_str = ", ".join(f"{texts[j]}={row[j]:.4f}" for j in top3)
            print(f"  {lab:>8s}: {top3_str}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
